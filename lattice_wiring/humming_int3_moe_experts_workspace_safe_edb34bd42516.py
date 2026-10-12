"""INT3 (group 128) Humming routed experts of Geometric-AI/DeepSeek-V4.1-Flash (32 of 40 layers).

HummingIndexedExperts.apply: moe_align_block_size, gate/up grouped GEMM (5120 -> 4608, INT3 weights,
bf16 activations), clamped SiLU(gate) * up (swiglu_limit 10), down grouped GEMM (2304 -> 5120), then the
top-6 weighted sum into `output`. Intermediates live in the engine's workspace13 / workspace2.
"""
import torch
from lattice_kernel_harness import Case, close_within_noise
from lattice_kernel_harness.module import ModuleTask, setup_vllm
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts

E, TOPK, HIDDEN, INTER, GROUP = 384, 6, 5120, 2304, 128
SWIGLU_LIMIT = 10.0
MAX_TOKENS = 16384  # engine max_num_batched_tokens

CALL_SITE = "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe:HummingIndexedExperts.apply"

# Token counts the profile puts time in (64: decode, 0.567; 16384: mixed chunk, 0.347), the decode ramp,
# the mixed-step CUDA-graph capture sizes 128/256/512, plus 1 and an odd count for the tails.
TOKEN_COUNTS = (1, 24, 32, 40, 48, 56, 64, 128, 256, 512, 777, 3580, 3697, 8222, 8281, 8282, 8283,
                11075, 11682, 11799, 16384)
WEIGHTS = {1: 0.002, 24: 0.0004, 32: 0.0004, 40: 0.0004, 48: 0.0005, 56: 0.0005, 64: 0.567,
           128: 0.01, 256: 0.01, 512: 0.01, 777: 0.002, 3580: 0.0007, 3697: 0.0007, 8222: 0.0013,
           8281: 0.0054, 8282: 0.0054, 8283: 0.0081, 11075: 0.0016, 11682: 0.0017, 11799: 0.0017,
           16384: 0.347}

_CACHE = {}


def _build_experts(seed):
    """The INT3 experts exactly as CompressedTensorsWNA16MoEMethod builds them on the Humming backend."""
    from compressed_tensors.quantization import QuantizationArgs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
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
            param(f"{name}_weight_shape",
                  torch.tensor([float(n), float(k)]).expand(E, 2).contiguous())
    qargs = QuantizationArgs(num_bits=3, type="int", symmetric=True, strategy="group",
                             group_size=GROUP)
    convert_to_humming_moe_kernel_format(layer, quant_config=_humming_wna16_weight_schema(qargs))
    quant_config = get_humming_moe_quant_config(layer, gemm1_clamp_limit=SWIGLU_LIMIT)
    experts = HummingIndexedExperts(moe_config=moe, quant_config=quant_config)
    return experts, layer.w13_weight, layer.w2_weight, layer


def make_inputs(case, seed):
    """The experts module (built once per seed) and what FusedMoEKernel passes apply() at m tokens:
    a separate (m, 5120) bf16 output (the engine aliases its own output buffer), bf16 hidden states,
    top-6 routing (fp32 weights, int32 ids) over 384 experts, and workspaces sized by workspace_shapes(m)."""
    setup_vllm()
    import vllm.forward_context as fc
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if not fc.is_forward_context_available():
        fc._forward_context = fc.ForwardContext(no_compile_layers={}, attn_metadata={},
                                                slot_mapping={})
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
    """`output` (bf16, m x 5120) must match the original within its own run-to-run noise.

    Humming's INT3 grouped GEMMs are not deterministic run to run (split/atomic accumulation across
    blocks): on this B300 two identical calls differ by up to 0.0625 at 56 tokens and 0.25 at 8,283
    tokens, with 4-20 elements outside rtol/atol 2e-2. So: close_within_noise with rtol = atol = 2e-2,
    at most 0.1% of elements outside that, no element off by more than 0.5, and relative L2 error of
    the whole output at most 1e-2. A systematic error (wrong expert, wrong scale group, missing clamp)
    fails the L2 bound.
    """
    c, r = _tensors(candidate), _tensors(reference)
    if len(c) != len(r):
        return [f"expected {len(r)} tensors, got {len(c)}"]
    failures = []
    for i, (a, b) in enumerate(zip(c, r)):
        failures += close_within_noise(a, b, f"output[{i}]", rtol=2e-2, atol=2e-2,
                                       max_outlier_fraction=1e-3, max_outlier_error=0.5,
                                       max_relative_error=1e-2)
    return failures


baseline = HummingIndexedExperts.apply


def wire(function, original, call_site):
    """The new apply for this model's INT3 layers; anything else goes back to the original."""
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        args = (self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                expert_tokens_meta, apply_router_weight_on_input)
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
            and (expert_tokens_meta is None or expert_tokens_meta.psum_recv_per_rank is None)
        )
        return (function if handled else original)(*args)
    return apply


TASK = ModuleTask(
    name="humming_int3_moe_experts",
    source="vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py",
    entry="apply",
    cases=tuple(Case(f"m{m}", {"m": m}, weight=WEIGHTS[m]) for m in TOKEN_COUNTS),
    dev_cases=("m64", "m16384"),
    make_inputs=make_inputs,
    baseline=baseline,
    compare=compare,
    state=state,
    regime="both",
    eager_fraction=0.41,
    deterministic=False,
    token_count="m",
    wire=wire,
    call_sites={CALL_SITE: "apply"},
)