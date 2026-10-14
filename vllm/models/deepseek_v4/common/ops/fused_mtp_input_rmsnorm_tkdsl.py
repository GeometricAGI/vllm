from tkdsl import nvidia as tk
import torch

HIDDEN = 4096
VEC = 16
LANES = 256


def _compile(T, hc_mult, eps):
    rows_e = T
    rows_h = T * hc_mult
    fspec = tk.vec.bf(VEC)
    vw = 8 if VEC == 16 else None
    e_spec = tk.gl("bf16", (rows_e, HIDDEN))
    h_spec = tk.gl("bf16", (rows_h, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    p_spec = tk.gl("s32", (2 * T,))
    warps = LANES // 32
    red_w = 1 << (warps - 1).bit_length()
    scratch_spec = tk.tiles.sv("f32", red_w)
    inp_cache = tk.gmem.CachePolicy(l1="evict_first", l2="evict_first", fraction=1.0)

    @tk.inline
    def norm_row(src, dst, wt, row, column, scratch, zero):
        x = tk.load(fspec, src, (row, column), cache=inp_cache, vector_words=vw)
        w = tk.load(fspec, wt, (0, column), vector_words=vw)
        s = tk.sum_squares([x])
        tot = tk.sum(s, scratch=scratch, count=warps, width=red_w, base=0,
                     policy=tk.SumPolicy.LOCAL_SERIAL_VECTOR)
        scale = tk.ops.rsqrt(tk.ops.fma(tot, 1.0 / HIDDEN, eps), ftz=True)
        if zero is not None:
            scale = tk.ops.select(zero, 0.0, scale)
        tk.store(dst, (x * w).scale(scale), (row, column), vector_words=vw)

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(eout_ptr, hout_ptr, emb_ptr, pos_ptr, prev_ptr, ew_ptr, hw_ptr):
        tid = tk.arch.thread_idx_x()
        column = tid * VEC
        b = tk.arch.block_idx_x()
        eout = tk.tensor(e_spec, eout_ptr)
        hout = tk.tensor(h_spec, hout_ptr)
        emb = tk.tensor(e_spec, emb_ptr)
        prev = tk.tensor(h_spec, prev_ptr)
        ew = tk.tensor(w_spec, ew_ptr)
        hw = tk.tensor(w_spec, hw_ptr)
        pos = tk.tensor(p_spec, pos_ptr)
        scratch = tk.shared_allocator().allocate(scratch_spec)
        if b < rows_e:
            lo = tk.load(pos, (2 * b,), mask=True, other=0)
            hi = tk.load(pos, (2 * b + 1,), mask=True, other=0)
            zero = (lo | hi) == 0
            norm_row(emb, eout, ew, b, column, scratch, zero)
        else:
            norm_row(prev, hout, hw, b - rows_e, column, scratch, None)

    @tk.jit
    def launch(eout, hout, emb, pos, prev, ew, hw, stream):
        kernel(eout, hout, emb, pos, prev, ew, hw).launch(
            grid=rows_e + rows_h, block=LANES, stream=stream)

    pointer = tk.compiler.KernelArgument.pointer(0)
    target = tk.profile(tk.current_specialization(required=True).target)
    return tk.compile(launch, pointer, pointer, pointer, pointer, pointer, pointer, pointer, 0, target=target)


@tk.specialize(key=lambda emb, pos, prev, ew, hw, eps, hc: (emb.shape[0], hc, float(eps)))
def _compiled(T, hc_mult, eps):
    return _compile(T, hc_mult, eps)


@torch.no_grad()
def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult):
    T = inputs_embeds.shape[0]
    eout = torch.empty_like(inputs_embeds)
    hout = torch.empty_like(previous_hidden_states)
    prog = _compiled(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult)
    prog(eout, hout, inputs_embeds, positions.view(torch.int32), previous_hidden_states.view(T * hc_mult, HIDDEN),
         enorm_weight, hnorm_weight, torch.cuda.current_stream(inputs_embeds.device))
    return eout, hout
