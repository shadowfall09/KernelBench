"""Offline Hugging Face adapter with exact submitted/generated token IDs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .common import fingerprint

_FILE_HASHES = {}


def checkpoint_digest(directory):
    """Hash local weights/tokenizer assets; reuse bytes only when file metadata is unchanged."""
    directory = Path(directory).resolve()
    records = []
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if identity not in _FILE_HASHES:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    digest.update(block)
            after = path.stat()
            if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity[1:]:
                raise ValueError("Local checkpoint changed while hashing")
            _FILE_HASHES[identity] = digest.hexdigest()
        records.append({"path": str(path.relative_to(directory)), "sha256": _FILE_HASHES[identity]})
    return fingerprint({"assets": records})


class LocalTransformersModel:
    def __init__(self, config, *, tokenizer=None, model=None):
        self.config, self.tokenizer, self.model = config, tokenizer, model
        self.prepared, self.prepared_sha = None, None
        self.model_sha = None

    def _tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, local_files_only=True, trust_remote_code=False)
        return self.tokenizer

    def count_tokens(self, messages):
        cleaned = [{"role": message["role"], "content": message["content"]} for message in messages]
        if any(not isinstance(message["content"], str) for message in cleaned):
            raise ValueError("The controlled local adapter only accepts text messages")
        self.prepared = self._tokenizer().apply_chat_template(cleaned, tokenize=True, add_generation_prompt=True,
                                                            return_tensors="pt", return_dict=True)
        self.prepared_sha = fingerprint({"messages": cleaned})
        ids = self.prepared["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("The tokenizer must return one nonempty, exact prompt sequence")
        return ids.shape[1]

    def _model(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM

            directory = Path(self.config.model)
            self.model_sha = checkpoint_digest(directory)
            self.model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                                            use_safetensors=True, dtype="auto")
            self.model.to(self.config.local_model_device)
            self.model.eval()
        return self.model

    def __call__(self, messages, config, directory, timeout, remaining_cost=None, *, max_output_tokens):
        import torch

        cleaned = [{"role": message["role"], "content": message["content"]} for message in messages]
        if self.prepared_sha != fingerprint({"messages": cleaned}):
            self.count_tokens(messages)
        model = self._model()
        inputs = {key: value.to(self.config.local_model_device) for key, value in self.prepared.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        options = {"max_new_tokens": max_output_tokens, "max_time": timeout, "num_beams": 1,
                   "num_return_sequences": 1, "return_dict_in_generate": False,
                   "do_sample": self.config.agent_temperature > 0}
        if self.config.agent_temperature > 0:
            options["temperature"] = self.config.agent_temperature
        with torch.inference_mode():
            sequences = model.generate(**inputs, **options)
        if (sequences.ndim != 2 or sequences.shape[0] != 1 or sequences.shape[1] < prompt_tokens
                or not torch.equal(sequences[:, :prompt_tokens], inputs["input_ids"])):
            raise ValueError("Local generation returned an unexpected prefix or batch")
        output = sequences[0, prompt_tokens:]
        count = output.numel()
        return {"text": self._tokenizer().decode(output.tolist(), skip_special_tokens=True), "cost": None,
                "input_tokens": prompt_tokens, "output_tokens": count, "total_tokens": prompt_tokens + count,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "models": [str(config.model)],
                "model_snapshot_sha256": self.model_sha, "provider_error": count > max_output_tokens,
                "token_scope": "exact_prompt_and_returned_generated_ids; one_sequence_no_beam_search"}
