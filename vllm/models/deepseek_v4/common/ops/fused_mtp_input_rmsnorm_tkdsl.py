from tkdsl import nvidia as tk
import torch

HIDDEN = 4096
VECTOR = 8
LANES = 128
FRAGMENTS = HIDDEN // (VECTOR * LANES)
WARPS = LANES // 32

_FRAG = tk.vec.bf(VECTOR)
_CACHE = {}


def _compile(T, hc, eps):
    emb_spec = tk.gl("bf16", (T, HIDDEN))
    prev_spec = tk.gl("bf16", (T * hc, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    scratch_spec = tk.tiles.sv("f32", WARPS)
    tasks = hc + 1

    @tk.inline
    def load_row(tensor, row, column):
        vals = []
        for f in range(FRAGMENTS):
            off = f * LANES * VECTOR
            col = column if off == 0 else tk.ops.add(column, off, dtype='u32')
            vals.append(tk.load(_FRAG, tensor, (row, col), vector_words=8))
        return vals

    @tk.inline
    def norm_row(src, dst, wt, row, column, lane, warp, scratch):
        vals = load_row(src, row, column)
        weights = []
        for f in range(FRAGMENTS):
            off = f * LANES * VECTOR
            col = column if off == 0 else tk.ops.add(column, off, dtype='u32')
            weights.append(tk.load(_FRAG, wt, (0, col), vector_words=8))
        xs = []
        ws = []
        for f in range(FRAGMENTS):
            for i in range(4):
                xs.append(tk.ops.unpack(vals[f].values[i]))
                ws.append(tk.ops.unpack(weights[f].values[i]))
        lane_sum = tk.ops.zero_f32()
        for lo, hi in xs:
            lane_sum = tk.ops.fma(lo, lo, lane_sum)
            lane_sum = tk.ops.fma(hi, hi, lane_sum)
        row_sum = tk.sum(
            lane_sum, scratch=scratch, index=warp, count=WARPS,
            width=WARPS, lane=lane, base=0,
        )
        scale = tk.ops.rsqrt(row_sum * (1.0 / HIDDEN) + eps, ftz=True)
        for f in range(FRAGMENTS):
            off = f * LANES * VECTOR
            col = column if off == 0 else tk.ops.add(column, off, dtype='u32')
            packed = []
            for i in range(4):
                xl, xh = xs[f * 4 + i]
                wl, wh = ws[f * 4 + i]
                packed.append(tk.ops.pack(((xl * scale) * wl, (xh * scale) * wh), dtype="bf16"))
            tk.store(dst, vals[f]._new(tuple(packed)), (row, col), vector_words=4)

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(enorm_out, hnorm_out, emb_p, pos_p, prev_p, ew_p, hw_p):
        tid = tk.arch.thread_idx_x()
        lane = tk.arch.lane_id()
        warp = tid // 32
        column = tid * VECTOR
        b = tk.arch.block_idx_x()
        token = b // tasks
        task = b - token * tasks
        scratch = tk.shared_allocator().allocate(scratch_spec)
        emb = tk.tensor(emb_spec, emb_p)
        prev = tk.tensor(prev_spec, prev_p)
        eo = tk.tensor(emb_spec, enorm_out)
        ho = tk.tensor(prev_spec, hnorm_out)
        ew = tk.tensor(w_spec, ew_p)
        hw = tk.tensor(w_spec, hw_p)
        if task == 0:
            pos = tk.tensor(tk.gl("u32", (1, 2 * T)), pos_p)[(0, token * 2)]
            if pos != 0:
                norm_row(emb, eo, ew, token, column, lane, warp, scratch)
            else:
                for f in range(FRAGMENTS):
                    off = f * LANES * VECTOR
                    col = column if off == 0 else tk.ops.add(column, off, dtype='u32')
                    z = tk.load(_FRAG, ew, (0, col), vector_words=8)
                    zp = []
                    for i in range(4):
                        zp.append(tk.ops.pack((tk.ops.zero_f32(), tk.ops.zero_f32()), dtype="bf16"))
                    tk.store(eo, z._new(tuple(zp)), (token, col), vector_words=4)
        else:
            row = token * hc + (task - 1)
            norm_row(prev, ho, hw, row, column, lane, warp, scratch)

    @tk.jit
    def launch(enorm_out, hnorm_out, emb, pos, prev, ew, hw, stream):
        kernel(enorm_out, hnorm_out, emb, pos, prev, ew, hw).launch(
            grid=T * tasks, block=LANES, stream=stream)

    return launch


def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, *, eps, hc_mult):
    T, H = inputs_embeds.shape
    assert H == HIDDEN
    eo = torch.empty_like(inputs_embeds)
    ho = torch.empty_like(previous_hidden_states)
    key = (T, hc_mult, eps)
    prog = _CACHE.get(key)
    stream = torch.cuda.current_stream()
    if prog is None:
        prog = tk.compile(_compile(T, hc_mult, eps), eo, ho, inputs_embeds, positions,
                          previous_hidden_states, enorm_weight, hnorm_weight, stream)
        _CACHE[key] = prog
    prog(eo, ho, inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, stream)
    return eo, ho
