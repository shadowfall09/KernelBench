"""NKI 0.7 native Gather input witnesses; no SDK/device import at module load."""

from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import sys
import types
from pathlib import Path

from ..common import fingerprint
from ..hardware.neuron_tables import integer
from .work_counters import sha

PROTOCOL = "kernelbench_pro_neuron_native_ast_v1"
SCOPE = "native_gather_definitions_to_actual_inspected_frozen_inputs; not_output_causality_or_formal_timing"
LIMIT = 4 * 1024**2


def decode(raw):
    import msgpack

    if type(raw) is not bytes or not 0 < len(raw) <= LIMIT:
        raise ValueError("Native NKI AST exceeds its 4 MiB envelope")
    try:
        value = msgpack.unpackb(raw, raw=False, strict_map_key=False, max_str_len=LIMIT,
            max_bin_len=LIMIT, max_array_len=200000, max_map_len=0, max_ext_len=0)
    except (ValueError, TypeError, msgpack.exceptions.UnpackException) as exc:
        raise ValueError("Invalid native NKI MessagePack") from exc
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > 1000000 or depth > 128:
            raise ValueError("Native NKI AST exceeds its decoded envelope")
        if isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif type(item) not in {str, bytes, int, float, bool, type(None)}:
            raise ValueError("Unsupported native NKI AST value")
    if (not isinstance(value, list) or len(value) != 10 or not isinstance(value[0], str)
            or not all(isinstance(value[i], list) for i in (1, 2, 3, 4, 5, 9))
            or not all(integer(value[i]) for i in (6, 7, 8))
            or len(value[1]) + len(value[2]) > 4096):
        raise ValueError("Unsupported native NKI 0.7 Gather envelope")
    return value


def definitions(value):
    result = []
    for kind, entries in (("function", value[1]), ("class", value[2])):
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != (5 if kind == "function" else 6) or not isinstance(entry[0], str):
                raise ValueError("Unsupported native NKI definition")
            result.append({"kind": kind, "name": entry[0], "location": entry[-1]})
            if kind == "class":
                if not isinstance(entry[4], list):
                    raise ValueError("Unsupported native NKI class methods")
                for method in entry[4]:
                    if not isinstance(method, list) or len(method) != 5 or not isinstance(method[0], str):
                        raise ValueError("Unsupported native NKI method")
                    result.append({"kind": "function", "name": method[0], "location": method[-1]})
    if len(result) > 20000:
        raise ValueError("Native NKI definitions exceed their envelope")
    return result


class Inputs:
    """Bind real inspect returns to sources frozen before candidate execution."""

    def __init__(self, registry, language=None):
        self.registry, self.language = registry, language
        self.classes = {}
        self.trees = {}
        if language is not None:
            for name, module in list(sys.modules.items()):
                if not name.startswith("nki") or type(module) is not types.ModuleType:
                    continue
                for value in list(vars(module).values()):
                    if isinstance(value, type):
                        try:
                            path = str(Path(inspect.getsourcefile(value)).resolve())
                        except (OSError, TypeError):
                            continue
                        if path in language.paths:
                            self.classes[id(value)] = (value, path, tuple((k, id(v)) for k, v in vars(value).items()))

    def read(self, obj, returned, *, direct=False):
        if (not isinstance(returned, tuple) or len(returned) != 2 or not isinstance(returned[0], list)
                or any(type(line) is not str for line in returned[0]) or not integer(returned[1], minimum=1)):
            return {"status": "unresolved", "reason": "invalid_inspected_source"}
        if type(obj) is types.FunctionType:
            owner = self.registry.source_for(obj)
            if owner:
                path, digest, text, role = str(Path(owner["path"]).resolve()), owner["source_sha256"], self.registry.texts[owner["source_sha256"]], "registered_code"
            elif self.language is not None and self.language.python_sources(obj):
                path = str(Path(obj.__code__.co_filename).resolve())
                if path not in self.language.paths:
                    return {"status": "unresolved", "reason": "non_source_callable"}
                digest, text, role = self.language.paths[path]["file_sha256"], self.language.texts[path], "approved_primitive"
            else:
                return {"status": "unresolved", "reason": "unregistered_inspected_callable"}
        elif id(obj) in self.classes and self.classes[id(obj)][0] is obj:
            _, path, state = self.classes[id(obj)]
            if state != tuple((k, id(v)) for k, v in vars(obj).items()):
                return {"status": "unresolved", "reason": "approved_class_changed"}
            digest, text, role = self.language.paths[path]["file_sha256"], self.language.texts[path], "approved_class_source"
        else:
            return {"status": "unresolved", "reason": "unapproved_inspected_object"}
        start, lines = returned[1], returned[0]
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        body = "".join(lines)
        expected = "".join(normalized.splitlines(keepends=True)[start - 1:start - 1 + len(lines)])
        if body != expected or not body or len(body.encode()) > LIMIT:
            return {"status": "unresolved", "reason": "inspected_body_differs_from_frozen_source"}
        if digest not in self.trees:
            self.trees[digest] = ast.parse(normalized)
        tree = self.trees[digest]
        spans = [{"name": n.name, "kind": "class" if isinstance(n, ast.ClassDef) else "function",
                  "span": [n.lineno, n.col_offset + 1, n.end_lineno, n.end_col_offset + 1]}
                 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and start <= n.lineno and n.end_lineno < start + len(lines)]
        return {"status": "observed", "name": str(obj.__module__) + "." + obj.__qualname__, "path": path,
                "role": role, "source_sha256": digest, "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "direct_native_read": direct,
                "callable": self.registry.callable_identity(obj) if type(obj) is types.FunctionType else None,
                "start_line": start, "line_count": len(lines), "definitions": spans}


def bind(raw, reads, config, *, simulated=False, module_handoff=False):
    value = decode(raw)
    if not isinstance(reads, list) or len(reads) > 4096 or not isinstance(config, list) or len(config) != 3:
        raise ValueError("Invalid native NKI witness envelope")
    for record in reads:
        if not isinstance(record, dict) or record.get("status") not in {"observed", "unresolved"}:
            raise ValueError("Invalid native NKI source witness")
        if record["status"] == "observed" and (
                any(not isinstance(record.get(k), str) or len(record[k]) > 4096 for k in ("name", "path"))
                or record.get("role") not in {"registered_code", "approved_primitive", "approved_class_source"}
                or not sha(record.get("source_sha256")) or not sha(record.get("body_sha256"))
                or type(record.get("direct_native_read")) is not bool
                or record.get("role") == "registered_code" and (
                    not isinstance(record.get("callable"), dict) or record["callable"].get("kind") != "registered_code"
                    or record["callable"].get("source_sha256") != record["source_sha256"]
                    or not integer(record["callable"].get("code_index")))
                or not integer(record.get("start_line"), minimum=1) or not integer(record.get("line_count"), minimum=1)
                or not isinstance(record.get("definitions"), list) or len(record["definitions"]) > 2048
                or any(not isinstance(n, dict) or not isinstance(n.get("name"), str) or n.get("kind") not in {"function", "class"}
                       or not isinstance(n.get("span"), list) or len(n["span"]) != 4
                       or any(not integer(v, minimum=1) for v in n["span"]) for n in record["definitions"])):
            raise ValueError("Invalid frozen native NKI source witness")
    entries, matches, unresolved = definitions(value), [], []
    observed = [r for r in reads if r.get("status") == "observed"]
    for index, entry in enumerate(entries):
        location = entry["location"]
        if (not isinstance(location, list) or len(location) != 5 or not isinstance(location[0], str)
                or any(not integer(x, minimum=1) for x in location[1:])):
            raise ValueError("Invalid native NKI definition source span")
        candidates = []
        for read_index, record in enumerate(reads):
            if record.get("status") != "observed" or location[0] != record["path"]:
                continue
            if not simulated and record["direct_native_read"] is not True:
                continue
            if entry["name"] != record["name"] and not entry["name"].startswith(record["name"] + "."):
                continue
            if any(n["kind"] == entry["kind"] and n["name"] == entry["name"].split(".")[-1]
                   and n["span"] == location[1:] for n in record["definitions"]):
                candidates.append(read_index)
        if not candidates:
            unresolved.append(index)
        else:
            # Multiple identical inspect reads are the same input witness.
            identities = {(reads[i]["source_sha256"], reads[i]["body_sha256"]) for i in candidates}
            if len(identities) != 1:
                unresolved.append(index)
            else:
                matches.append({"definition": index, "read": candidates[0]})
    roots = [m for m in matches if entries[m["definition"]]["kind"] == "function" and entries[m["definition"]]["name"] == value[0]]
    complete = bool(entries) and not unresolved and len(roots) == 1 and not value[3] and value[6:9] == config
    result = {"protocol": PROTOCOL, "scope": SCOPE, "simulated": simulated, "formal_origin_qualified": False,
              "ast_sha256": hashlib.sha256(raw).hexdigest(), "ast_bytes": len(raw),
              "artifact": "compiler_ast/" + hashlib.sha256(raw).hexdigest() + ".msgpack",
              "messagepack_b64": base64.b64encode(raw).decode(), "root": value[0],
              "definitions": entries, "reads": reads, "matches": matches, "unresolved_definitions": unresolved,
              "config": config, "serialized_config": value[6:9], "unsupported_external_entries": bool(value[3]),
              "global_binding_count": len(value[5]), "all_definition_sources_matched": complete,
              "root_source_sha256": reads[roots[0]["read"]]["source_sha256"] if len(roots) == 1 else None,
              "native_definition_sources_bound": complete and not simulated,
              "module_handoff_observed": module_handoff,
              "status": "observed" if complete else "partial" if observed else "not_measured"}
    result["receipt_sha256"] = fingerprint(result)
    return result


def validate(packet, files, sizes, *, allow_simulated=False):
    if (not isinstance(files, dict) or not isinstance(sizes, dict)
            or not isinstance(packet, dict) or packet.get("protocol") != PROTOCOL or packet.get("scope") != SCOPE
            or packet.get("simulated") is not False and not allow_simulated
            or packet.get("formal_origin_qualified") is not False or type(packet.get("module_handoff_observed")) is not bool
            or not isinstance(packet.get("reads"), list) or len(packet["reads"]) > 4096
            or not isinstance(packet.get("config"), list) or len(packet["config"]) != 3
            or not isinstance(packet.get("messagepack_b64"), str) or len(packet["messagepack_b64"]) > ((LIMIT + 2) // 3) * 4):
        raise ValueError("Invalid native NKI source-input packet")
    try:
        raw = base64.b64decode(packet["messagepack_b64"], validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid native NKI MessagePack encoding") from exc
    rebuilt = bind(raw, packet["reads"], packet["config"], simulated=packet["simulated"],
                   module_handoff=packet["module_handoff_observed"])
    if packet != rebuilt or files.get(packet["artifact"]) != packet["ast_sha256"] or sizes.get(packet["artifact"]) != len(raw):
        raise ValueError("Native NKI AST packet differs from its original artifact")
    for record in packet["reads"]:
        if record.get("status") != "observed":
            continue
        path = "compiler_sources/" + record["source_sha256"] + ".py"
        if not sha(record["source_sha256"]) or files.get(path) != record["source_sha256"]:
            raise ValueError("Native NKI inspected source snapshot missing or changed")
    return packet


class KernelProxy:
    """Freeze this native instance's specialized AST immediately before lowering."""

    def __init__(self, native, compile_method, before):
        self.native, self.compile_method, self.before = native, compile_method, before

    def __getattr__(self, name):
        return getattr(self.native, name)

    def compile(self, *args, **kwargs):
        self.before(self.native)
        return self.compile_method(self.native, *args, **kwargs)
