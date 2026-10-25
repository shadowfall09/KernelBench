"""Versioned production-family contracts, importable without Torch."""

from __future__ import annotations

BASE = [
    ("online_attention", "attention", "attention", "design"),
    ("linear_cross_entropy", "linear_cross_entropy", "cross_entropy", "design"),
    ("welford_normalization", "layer_norm", "normalization", "design"),
    ("partial_topk", "partial_topk", "topk", "design"),
    ("chunked_linear_attention", "linear_attention", "attention_linear", "design"),
    ("chunked_state_space", "state_space", "state_space", "design"),
    ("persistent_recurrent", "recurrent", "recurrent", "design"),
    ("fft_long_convolution", "long_convolution", "convolution", "design"),
    ("grouped_moe", "grouped_moe", "moe", "design"),
    ("dequantization_gemm", "dequant_gemm", "int4_gemm", "design"),
    ("paged_attention", "paged_attention", "paged", "design"),
    ("split_k_gemm", "split_k_gemm", "matmul", "design"),
    ("accurate_gemm_emulation", "accurate_gemm", "matmul", "numerical"),
    ("format_budget_gemm", "accurate_gemm", "matmul", "numerical"),
    ("format_budget_attention", "budget_attention", "attention", "numerical"),
    ("format_budget_moe", "budget_moe", "moe", "numerical"),
    ("block_fp4_quantization_gemm", "block_fp4_gemm", "fp4_gemm", "numerical"),
    ("token_fp8_quantization_attention", "token_fp8_attention", "attention", "numerical"),
    ("bf16_gemm", "fixed_gemm", "fixed_gemm", "implementation"),
    ("fp16_gemm", "fixed_gemm", "fixed_gemm", "implementation"),
    ("fp8_scaled_gemm", "fp8_scaled_gemm", "fp8_gemm", "implementation"),
    ("w4a16_gemm", "dequant_gemm", "int4_gemm", "implementation"),
    ("batched_grouped_gemm", "grouped_gemm", "grouped_gemm", "implementation"),
    ("paged_decode", "paged_attention", "paged", "implementation"),
    ("batched_topk", "partial_topk", "topk", "implementation"),
    ("swiglu_projection", "swiglu", "swiglu", "implementation"),
    ("residual_rmsnorm", "residual_rmsnorm", "rmsnorm", "implementation"),
    ("rotary_embedding", "rotary", "rotary", "implementation"),
    ("quantized_moe", "quantized_moe", "quantized_moe", "implementation"),
    ("gqa_decode", "gqa_decode", "gqa", "implementation"),
]
BACKWARDS = [(1, [0, 1, 2]), (2, [0, 1]), (3, [0, 1, 2]), (5, [0, 1, 2]),
             (6, [0, 1, 2, 3, 4]), (7, [0, 1, 2, 3, 4]), (8, [0, 1]),
             (9, [0, 1, 3]), (10, [0]), (11, [0, 1, 2]), (26, [0, 1, 2])]

# shape expressions only allow a symbol or integer division of two symbols.
# domain controls generation and role-aware perturbation; storage controls dtype.
ROLES = {
    "attention": [("q", "B H S D", "activation", "signed"), ("k", "B H S D", "activation", "signed"),
                  ("v", "B H S D", "activation", "signed"), ("mask", "S S", "bool", "causal_mask")],
    "cross_entropy": [("x", "M K", "activation", "signed"), ("weight", "V K", "activation", "weight"),
                      ("targets", "M", "int64", "class_index")],
    "normalization": [("x", "M D", "activation", "signed"), ("weight", "D", "fp32", "unit_weight"),
                      ("bias", "D", "fp32", "signed"), ("eps", "", "scalar", "epsilon")],
    "topk": [("x", "M N", "fp32", "distinct"), ("k", "", "scalar", "topk_count")],
    "attention_linear": [("q", "B H S D", "activation", "signed"), ("k", "B H S D", "activation", "signed"),
                         ("v", "B H S D", "activation", "signed")],
    "state_space": [("u", "B H S D", "activation", "signed"), ("delta", "B H S", "fp32", "positive_step"),
                    ("decay", "H", "fp32", "negative_decay"), ("b", "B H S R", "activation", "signed"),
                    ("c", "B H S R", "activation", "signed")],
    "recurrent": [("x", "B S D", "activation", "signed"), ("h", "B D", "activation", "signed"),
                  ("input_weight", "D D", "activation", "weight"), ("hidden_weight", "D D", "activation", "weight"),
                  ("bias", "D", "fp32", "signed")],
    "convolution": [("x", "B D S", "activation", "signed"), ("kernel", "D L", "activation", "weight")],
    "moe": [("x", "M K", "activation", "signed"), ("weight", "E N K", "activation", "weight"),
            ("experts", "M Top", "int64", "expert_index"), ("gates", "M Top", "fp32", "probabilities")],
    "int4_gemm": [("x", "M K", "activation", "signed"), ("packed", "N K/2", "uint8", "packed_int4"),
                  ("scale", "N K/G", "fp32", "scale"), ("group_size", "", "scalar", "group_size")],
    "paged": [("q", "B H D", "activation", "signed"), ("k_cache", "P Hkv Block D", "activation", "signed"),
              ("v_cache", "P Hkv Block D", "activation", "signed"), ("block_table", "B Pages", "int64", "page_index"),
              ("lengths", "B", "int64", "sequence_length")],
    "matmul": [("x", "M K", "activation", "signed"), ("weight", "K N", "activation", "weight")],
    "fp4_gemm": [("x", "M K", "fp32", "signed"), ("weight", "N K", "fp32", "weight"),
                 ("block", "", "scalar", "group_size")],
    "fixed_gemm": [("x", "M K", "activation", "signed"), ("weight", "N K", "activation", "weight")],
    "fp8_gemm": [("x", "M K", "fp8_e4m3fn", "signed"), ("weight", "N K", "fp8_e4m3fn", "weight"),
                 ("x_scale", "M 1", "fp32", "scale"), ("weight_scale", "N 1", "fp32", "scale")],
    "grouped_gemm": [("x", "E M K", "activation", "signed"), ("weight", "E N K", "activation", "weight")],
    "swiglu": [("x", "M K", "activation", "signed"), ("up", "N K", "activation", "weight"),
               ("gate", "N K", "activation", "weight")],
    "rmsnorm": [("x", "M D", "activation", "signed"), ("residual", "M D", "activation", "signed"),
                ("weight", "D", "fp32", "unit_weight"), ("eps", "", "scalar", "epsilon")],
    "rotary": [("x", "B H S D", "activation", "signed"), ("cosine", "S D/2", "fp32", "cosine"),
               ("sine", "S D/2", "fp32", "sine")],
    "quantized_moe": [("x", "M K", "activation", "signed"), ("packed", "E N K/2", "uint8", "packed_int4"),
                      ("scale", "E N K/G", "fp32", "scale"), ("experts", "M Top", "int64", "expert_index"),
                      ("gates", "M Top", "fp32", "probabilities"), ("group_size", "", "scalar", "group_size")],
    "gqa": [("q", "B H 1 D", "activation", "signed"), ("k", "B Hkv S D", "activation", "signed"),
            ("v", "B Hkv S D", "activation", "signed")],
}

FIXED_FORMAT_COMPARISON = {"norm_rtol_by_activation_dtype": {"fp32": 1e-3, "fp16": 2e-3, "bf16": 1e-2},
                           "norm_atol": 1e-8, "element_atol": 1e-8,
                           "element_rtol_multiplier": 2, "element_global_rms_multiplier": 1}


def comparison_contract(spec, primary):
    if spec["precision_contract"]:
        return spec["precision_contract"]
    relative = FIXED_FORMAT_COMPARISON["norm_rtol_by_activation_dtype"][primary]
    return {"norm_rtol": relative, "norm_atol": FIXED_FORMAT_COMPARISON["norm_atol"],
            "element_rtol": relative * 2, "element_atol": FIXED_FORMAT_COMPARISON["element_atol"],
            "element_scale_rtol": relative}


def task(pid):
    if type(pid) is not int or not 1 <= pid <= 41:
        raise ValueError("Production task ID must be 1–41")
    base_id, gradient_indices = (pid, None) if pid <= 30 else BACKWARDS[pid - 31]
    name, operation, family, track = BASE[base_id - 1]
    fixed = {19: "bf16", 20: "fp16", 21: "bf16", 22: "bf16", 29: "bf16"}.get(base_id)
    if track == "numerical":
        fixed = "fp32"
    budget = {13: 1e-5, 14: 1e-2, 15: 5e-3, 16: 5e-3, 17: 2e-4, 18: 2e-3}.get(base_id)
    return {"problem_id": pid, "base_id": base_id, "name": name + ("_backward" if gradient_indices else ""),
            "operation": operation, "family": family, "track": "backward" if gradient_indices else track,
            "gradient_indices": gradient_indices, "roles": ROLES[family], "fixed_activation_dtype": fixed,
            "upstream_role": ({"name": "upstream", "shape": "forward_output_shape", "domain": "signed",
                               "dtype": "fp32" if family in {"cross_entropy", "attention_linear", "state_space", "recurrent", "convolution", "moe"}
                               else "activation"} if gradient_indices else None),
            "dtype_families": [fixed] if fixed else ["fp32", "bf16", "fp16"],
            "contract_kind": "error_budget" if track == "numerical" else "fixed_format",
            "precision_contract": ({"norm_rtol": budget, "norm_atol": 1e-7, "element_rtol": budget,
                                    "element_atol": 1e-7, "element_scale_rtol": budget} if budget else None),
            "fixed_format_comparison": FIXED_FORMAT_COMPARISON if not budget else None,
            "determinism": "deterministic_reference; candidate_must_meet_numeric_contract",
            "layer": "D" if track in {"design", "numerical"} else "I",
            "bound_certificate": None, "reachability_witness": None}
