"""INT3 (group 128) routed experts of DeepSeek-V4.1-Flash (32 of 40 layers), v3 decode path.

The original is #54's TKDSL `apply` (fused_humming_moe_tkdsl.py), which the engine runs today at
HummingIndexedExperts.apply. m <= 64: routing prep, INT3 gate/up grouped GEMV (gemm_gu, 5120 -> 4608), clamped
SwiGLU (_swiglu_k), INT3 down grouped GEMV (gemm_dn, 2304 -> 5120) and f32 -> bf16 (_cvt_k). m > 64: #54's
tcgen05 prefill path (s_gemm1 / s_gemm2), which the new code must call unchanged.
"""
import torch
from lattice_kernel_harness import Case, close_within_noise
from lattice_kernel_harness.module import ModuleTask, setup_vllm
from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe_tkdsl as _v2

E, TOPK, HIDDEN, INTER, GROUP = 384, 6, 5120, 2304, 128
SWIGLU_LIMIT = 10.0
MAX_TOKENS = 16384  # engine max_num_batched_tokens

CALL_SITE = "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe:HummingIndexedExperts.apply"

# Profile token counts: 64 (decode, 71.6% of GPU time), 16384 (mixed chunk, 19.5%), the partial chunks 8281-8283
# and 11075-11799, 3580/3697, and the decode ramp 24-56. Plus 1, 8 and 17 (odd) for the tails, and the
# mixed-step capture sizes 128/512.
TOKEN_COUNTS = (1, 8, 17, 24, 32, 40, 48, 56, 64, 128, 512, 3580, 8283, 11799, 16384)
WEIGHTS = {1: 0.003, 8: 0.002, 17: 0.002, 24: 0.0005, 32: 0.0005, 40: 0.0005, 48: 0.0005, 56: 0.0006,
           64: 0.70, 128: 0.004, 512: 0.004, 3580: 0.0011, 8283: 0.012, 11799: 0.003, 16384: 0.195}
# Padded batches as the engine sends them: rows padded up to a CUDA-graph size get expert id -1 and weight 0
# (VLLM_MOE_SKIP_PADDING); capture warm-up runs batches that are all padding. (m, padded rows, weight).
PADDED = ((1, 1, 0.0005), (64, 8, 0.02), (64, 64, 0.0005), (512, 512, 0.0005), (3580, 100, 0.0005))

_CACHE = {}


def _build_experts(seed):
    """The INT3 experts exactly as CompressedTensorsWNA16MoEMethod builds them on the Humming backend."""
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import _humming_wna16_weight_schema
    from vllm.model_executor.layers.quantization.utils.humming import (
        convert_to_humming_moe_kernel_format, get_humming_moe_quant_config)

    moe = FusedMoEConfig(
        num_experts=E, experts_per_token=TOPK, hidden_dim=HIDDEN, intermediate_size=INTER,
        num_local_experts=E, num_logical_experts=E, activation=MoEActivation.SILU, device="cuda",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16, max_num_tokens=MAX_TOKENS, swiglu_limit=SWIGLU_LIMIT)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    layer = torch.nn.Module()
    layer.moe_config = moe
    layer.params_dtype = torch.bfloat16
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.swiglu_limit = SWIGLU_LIMIT

    def param(name, t):
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))

    with torch.device("cuda"):
        # compressed-tensors pack-quantized INT3: [E, N, K*3/32] int32; any bit pattern is a valid code.
        for name, n, k in (("w13", 2 * INTER, HIDDEN), ("w2", HIDDEN, INTER)):
            param(f"{name}_weight_packed", torch.randint(
                -(2 ** 31), 2 ** 31 - 1, (E, n, k * 3 // 32), generator=g, dtype=torch.int32))
            scale = 0.005 + 0.01 * torch.rand(E, n, k // GROUP, generator=g)
            param(f"{name}_weight_scale", scale.to(torch.bfloat16))
            param(f"{name}_weight_shape", torch.tensor([float(n), float(k)]).expand(E, 2).contiguous())
    qargs = QuantizationArgs(num_bits=3, type="int", symmetric=True, strategy="group", group_size=GROUP)
    convert_to_humming_moe_kernel_format(layer, quant_config=_humming_wna16_weight_schema(qargs))
    quant_config = get_humming_moe_quant_config(layer, gemm1_clamp_limit=SWIGLU_LIMIT)
    experts = HummingIndexedExperts(moe_config=moe, quant_config=quant_config)
    return experts, layer.w13_weight, layer.w2_weight, layer


def make_inputs(case, seed):
    """The experts module (built once per seed) and what FusedMoEKernel passes apply() at m tokens: a separate
    (m, 5120) bf16 output, bf16 hidden states, top-6 routing (fp32 weights, int32 ids) over 384 experts, and
    workspaces sized by workspace_shapes(m). The last `pad` rows are padding: expert id -1 and weight 0."""
    setup_vllm()
    import vllm.forward_context as fc
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if not fc.is_forward_context_available():
        fc._forward_context = fc.ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    if seed not in _CACHE:
        _CACHE.clear()
        _CACHE[seed] = _build_experts(seed)
    experts, w1, w2, _layer = _CACHE[seed]
    m = case.params["m"]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed * 1000003 + m)
    with torch.device("cuda"):
        x = torch.randn(m, HIDDEN, generator=g, dtype=torch.bfloat16)
        scores = torch.randn(m, E, generator=g, dtype=torch.float32)
        topk_weights, topk_ids = torch.softmax(scores, dim=-1).topk(TOPK, dim=-1)
        topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        pad = min(case.params.get("pad", 0), m)
        if pad:
            topk_ids[m - pad:] = -1
            topk_weights[m - pad:] = 0
        ws1_shape, ws2_shape, out_shape = experts.workspace_shapes(
            m, INTER, HIDDEN, TOPK, E, E, None, MoEActivation.SILU)
        workspace13 = torch.zeros(ws1_shape, dtype=torch.bfloat16)
        workspace2 = torch.zeros(ws2_shape, dtype=torch.bfloat16)
        output = torch.zeros(out_shape, dtype=torch.bfloat16)
    args = (experts, output, x, w1, w2, topk_weights, topk_ids, MoEActivation.SILU, E, None, None,
            None, workspace13, workspace2, None, False)
    return args, {}


def state(args, kwargs):
    """What a call must leave right: only `output` (workspace contents are scratch, free to differ)."""
    return (args[1],)


def _tensors(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for y in x for t in _tensors(y)]
    return []


def compare(candidate, reference, case):
    """`output` (bf16, m x 5120) must match #54's original within that original's own run-to-run noise.

    The original's decode GEMVs accumulate split-K partials with f32 atomics, so it is not deterministic run to
    run: two identical calls differ by up to 0.0625 at 56 tokens and 0.25 at 8,283 tokens, with 4-20 elements
    outside rtol/atol 2e-2. So: close_within_noise with rtol = atol = 2e-2, at most 0.1% of elements outside
    that, no element off by more than 0.5, and relative L2 error of the whole output at most 1e-2. A
    systematic error (wrong expert, wrong scale group, missing clamp, a padded row counted) fails the L2 bound.
    Only real rows are compared: the engine drops the padded rows' output.
    """
    m = case.params["m"]
    real = m - min(case.params.get("pad", 0), m)
    c = [t[:real] for t in _tensors(candidate)]
    r = [t[:real] for t in _tensors(reference)]
    if len(c) != len(r):
        return [f"expected {len(r)} tensors, got {len(c)}"]
    failures = []
    for i, (a, b) in enumerate(zip(c, r)):
        if real == 0:
            continue  # all padding: nothing to compare; the case checks that the call runs
        failures += close_within_noise(a, b, f"output[{i}]", rtol=2e-2, atol=2e-2, max_outlier_fraction=1e-3,
                                       max_outlier_error=0.5, max_relative_error=1e-2)
    return failures


# The engine's apply today: #54's TKDSL apply, installed at CALL_SITE by its wiring.
baseline = _v2.apply


def wire(function, original, call_site):
    """The new apply for this model's INT3 layers; anything else goes back to the original."""
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        args = (self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                apply_router_weight_on_input)
        handled = (
            hidden_states.dtype == torch.bfloat16
            and hidden_states.size(-1) == HIDDEN
            and activation == MoEActivation.SILU
            and expert_map is None
            and a1q_scale is None
            and not apply_router_weight_on_input
            and topk_ids.size(1) == TOPK
            and self.num_experts == E
            and w1.size(0) == E
            and expert_tokens_meta is None
        )
        return (function if handled else original)(*args)
    return apply


TASK = ModuleTask(
    name="humming_int3_moe_experts_v3",
    source="vllm/model_executor/layers/fused_moe/experts/fused_humming_moe_tkdsl.py",
    entry="apply",
    cases=tuple(Case(f"m{m}", {"m": m}, weight=WEIGHTS[m]) for m in TOKEN_COUNTS)
    + tuple(Case(f"m{m}p{pad}", {"m": m, "pad": pad}, weight=w) for m, pad, w in PADDED),
    dev_cases=("m64", "m64p8", "m17"),
    make_inputs=make_inputs,
    baseline=baseline,
    compare=compare,
    state=state,
    regime="both",
    eager_fraction=0.2,
    deterministic=False,
    token_count="m",
    wire=wire,
    call_sites={CALL_SITE: "apply"},
)
