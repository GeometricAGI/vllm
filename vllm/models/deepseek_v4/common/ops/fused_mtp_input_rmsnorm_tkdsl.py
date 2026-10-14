from tkdsl import nvidia as tk
import torch

H = 4096
VECTOR = 8


def _build(T, eps, hc):
    LANES = 512 if T <= 8 else 128
    WARPS = LANES // 32
    FR = H // (VECTOR * LANES)
    rows_h = T * hc
    fragment_spec = tk.vec.bf(VECTOR)
    emb_spec = tk.gl("bf16", (T, H))
    prev_spec = tk.gl("bf16", (rows_h, H))
    w_spec = tk.gl("bf16", (H,))
    pos_spec = tk.gl("u32", (2 * T,))
    scratch_spec = tk.tiles.sv("f32", 32)

    @tk.inline
    def norm_row(src, dst, weight, row, column, scratch, keep):
        xs = [tk.load(fragment_spec, src, (row, column + f * LANES * VECTOR)) for f in range(FR)]
        ws = [tk.load(fragment_spec, weight, (column + f * LANES * VECTOR,)) for f in range(FR)]
        ss = tk.sum_squares(xs) * (keep * keep)
        total = tk.sum(
            ss, scratch=scratch, count=WARPS, width=WARPS, base=0,
            policy=tk.SumPolicy.LOCAL_SERIAL_VECTOR,
        )
        scale = tk.ops.rsqrt(tk.ops.fma(total, 1.0 / H, eps), ftz=True)
        scale = scale * keep
        for f in range(FR):
            outs = []
            for xp, wp in zip(xs[f].values, ws[f].values):
                xl, xh = tk.ops.unpack(xp)
                wl, wh = tk.ops.unpack(wp)
                outs.append(tk.ops.pack((xl * scale * wl, xh * scale * wh), dtype="bf16"))
            tk.store(dst, xs[f]._new(tuple(outs)), (row, column + f * LANES * VECTOR))

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(eout_ptr, hout_ptr, emb_ptr, prev_ptr, pos_ptr, ew_ptr, hw_ptr):
        tid = tk.arch.thread_idx_x()
        column = tid * VECTOR
        bid = tk.arch.block_idx_x()
        eout = tk.tensor(emb_spec, eout_ptr)
        hout = tk.tensor(prev_spec, hout_ptr)
        emb = tk.tensor(emb_spec, emb_ptr)
        prev = tk.tensor(prev_spec, prev_ptr)
        pos = tk.tensor(pos_spec, pos_ptr)
        ew = tk.tensor(w_spec, ew_ptr)
        hw = tk.tensor(w_spec, hw_ptr)
        scratch = tk.shared_allocator().allocate(scratch_spec)
        if bid < rows_h:
            norm_row(prev, hout, hw, bid, column, scratch, 1.0)
        else:
            t = bid - rows_h
            p0 = tk.load(pos, (2 * t,))
            p1 = tk.load(pos, (2 * t + 1,))
            keep = tk.ops.select((p0 | p1) != 0, 1.0, 0.0)
            norm_row(emb, eout, ew, t, column, scratch, keep)

    @tk.jit
    def launch(eout, hout, emb, prev, pos, ew, hw, stream):
        kernel(eout, hout, emb, prev, pos, ew, hw).launch(
            grid=rows_h + T, block=LANES, stream=stream)

    pointer = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, pointer, pointer, pointer, pointer, pointer, pointer, pointer, 0)


@tk.specialize(key=lambda emb, pos, prev, ew, hw, eps, hc: (emb.shape[0], float(eps), int(hc)))
def _compiled(T, eps, hc):
    return _build(T, eps, hc)


def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult):
    eout = torch.empty_like(inputs_embeds)
    hout = torch.empty_like(previous_hidden_states)
    pos32 = positions.view(torch.int32)
    prog = _compiled(inputs_embeds, pos32, previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult)
    prog(eout, hout, inputs_embeds, previous_hidden_states, pos32, enorm_weight, hnorm_weight,
         torch.cuda.current_stream(inputs_embeds.device))
    return eout, hout
