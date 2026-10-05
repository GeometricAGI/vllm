from tkdsl import nvidia as tk
import torch

HIDDEN = 4096
VEC = 8
LANES = 256
FRAGS = HIDDEN // (VEC * LANES)
STRIDE = LANES * VEC
WARPS = LANES // 32

_ACT = tk.gmem.CachePolicy(l1="evict_first", l2="evict_first", fraction=1.0)
_WGT = tk.gmem.CachePolicy(l2="evict_last", fraction=1.0)


def _compile(T: int):
    frag = tk.vec.bf(VEC)
    row_spec = tk.gl("bf16", (T, HIDDEN))
    hid_spec = tk.gl("bf16", (T * 4, HIDDEN))
    w_spec = tk.gl("bf16", (HIDDEN,))
    pos_spec = tk.gl("s32", (T * 2,))
    scratch_spec = tk.tiles.sv("f32", 32)

    def load_frags(tensor, row, column, cache):
        vals = []
        for f in range(FRAGS):
            col = column if f == 0 else tk.ops.add(column, f * STRIDE, dtype="u32")
            vals.append(tk.load(frag, tensor, (row, col), cache=cache))
        return vals

    @tk.inline
    def norm_row(out, src, w, row, column, lane, tid, scratch, keep):
        vals = load_frags(src, row, column, _ACT)
        ss = tk.sum_squares(vals, policy=tk.vec.ReductionPolicy.ACCURATE)
        total = tk.sum(ss, scratch=scratch, index=tid // 32, count=WARPS, width=8,
                       lane=lane, base=0, policy=tk.SumPolicy.LOCAL_SERIAL_VECTOR)
        mean = total * (1.0 / HIDDEN)
        scale = tk.ops.rsqrt(mean + 1e-6, ftz=True)
        scale = scale * keep
        for f, (v, wv) in enumerate(zip(vals, w)):
            col = column if f == 0 else tk.ops.add(column, f * STRIDE, dtype="u32")
            regs = []
            for rv, rw in zip(v.values, wv.values):
                a, b = tk.ops.unpack(rv)
                c, d = tk.ops.unpack(rw)
                regs.append(tk.ops.pack(((a * scale) * c, (b * scale) * d), dtype="bf16"))
            tk.store(out, v._new(tuple(regs)), (row, col))

    @tk.kernel(reqntid=(LANES, 1, 1))
    def kernel(eo_ptr, ho_ptr, emb_ptr, prev_ptr, pos_ptr, ew_ptr, hw_ptr):
        tid = tk.arch.thread_idx_x()
        lane = tk.arch.lane_id()
        column = tid * VEC
        b = tk.arch.block_idx_x()
        scratch = tk.shared_allocator().allocate(scratch_spec)
        if b < T:
            eo = tk.tensor(row_spec, eo_ptr)
            emb = tk.tensor(row_spec, emb_ptr)
            pos = tk.tensor(pos_spec, pos_ptr)
            ew = tk.tensor(w_spec, ew_ptr)
            p = tk.load(pos, (b * 2,))
            keep = tk.ops.select(p != 0, 1.0, 0.0)
            w = load_frags(ew, 0, column, _WGT)
            norm_row(eo, emb, w, b, column, lane, tid, scratch, keep)
        else:
            ho = tk.tensor(hid_spec, ho_ptr)
            prev = tk.tensor(hid_spec, prev_ptr)
            hw = tk.tensor(w_spec, hw_ptr)
            w = load_frags(hw, 0, column, _WGT)
            norm_row(ho, prev, w, b - T, column, lane, tid, scratch, 1.0)

    @tk.jit
    def launch(eo, ho, emb, prev, pos, ew, hw, stream):
        kernel(eo, ho, emb, prev, pos, ew, hw).launch(grid=T * 5, block=LANES, stream=stream)

    ptr = tk.compiler.KernelArgument.pointer(0)
    return tk.compile(launch, ptr, ptr, ptr, ptr, ptr, ptr, ptr, 0)


@tk.specialize(key=lambda *a: a[0])
def _compiled(T: int):
    return _compile(T)


@torch.no_grad()
def fused_mtp_input_rmsnorm(inputs_embeds, positions, previous_hidden_states, enorm_weight, hnorm_weight, *, eps, hc_mult):
    T = inputs_embeds.shape[0]
    eo = torch.empty_like(inputs_embeds)
    ho = torch.empty_like(previous_hidden_states)
    if T == 0:
        return eo, ho
    prog = _compiled(T)
    prog(eo, ho, inputs_embeds, previous_hidden_states.view(T * hc_mult, HIDDEN),
         positions.view(torch.int32), enorm_weight, hnorm_weight,
         torch.cuda.current_stream(inputs_embeds.device))
    return eo, ho
