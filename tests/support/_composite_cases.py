"""Explicit small constructor configurations for CPU arithmetic unit tests.

Only benchmark globals change; class/function algorithm bodies stay intact.
These configurations are not default-task calibration or device evidence.
"""

import ast

import torch

from kb_workflows.common import problem_paths


def small_composite_case(pid, dtype=torch.float64):
    path = problem_paths(2)[pid]
    source = path.read_text()
    tree = ast.parse(source)
    overrides = {"batch_size": 2, "in_features": 8, "out_features": 8, "input_size": 8, "hidden_size": 8, "output_size": 8,
        "in_channels": 4, "out_channels": 4, "groups": 2, "num_groups": 2, "depth": 8, "height": 8, "width": 8,
        "D": 8, "H": 8, "W": 8, "pool_kernel_size": 2}
    # These frozen LayerNorms normalize the last spatial axis, rather than the
    # channel axis. Retain that actual formula and its width/channel coupling.
    if pid == 3:
        overrides["width"] = 2
    if pid == 34:
        overrides["W"] = 2
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name) and name_node.id.endswith("_shape") and isinstance(node.value, (ast.Tuple, ast.List)):
                        for item in node.value.elts:
                            if isinstance(item, ast.Constant) and type(item.value) is int and item.value >= 8:
                                item.value = 4
            body.append(node)
            names = {item.id for target in node.targets for item in ast.walk(target) if isinstance(item, ast.Name)}
            for name in sorted(names & overrides.keys()):
                body.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(overrides[name])))
        else:
            body.append(node)
    tree.body = body
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    torch.manual_seed(123)
    model = namespace["Model"](*namespace["get_init_inputs"]()).to(dtype=dtype)
    args = [value.to(dtype=dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for value in namespace["get_inputs"]()]
    return source, model, args
