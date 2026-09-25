#pragma once

#include "custom_collective_common.cuh"

namespace vllm {

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

#if !defined(USE_ROCM)
/**
 * Barrier-free one-shot "push" allreduce, after "Every us Matters: Achieving
 * Near Speed-of-Light Latency in GPU Collectives" (arXiv:2607.16100).
 *
 * Unlike cross_device_reduce_1stage, which pulls from peer input buffers
 * between a start and an end barrier, every rank pushes its local input into
 * each peer's scratch buffer and polls its own scratch until all peers' data
 * has landed. Data arrival is detected per element, so there are no
 * cross-GPU barriers and no peer input pointers (hence no graph buffer
 * registration or eager staging copy).
 *
 * Two ways of detecting arrival are supported:
 * - LL: every 4 bytes of payload travel with a 4-byte epoch flag in the same
 *   8-byte word, NCCL-LL style ({d0, flag, d1, flag} per 16B line). It needs
 *   only 8-byte store atomicity, so it is also correct over PCIe. It halves
 *   the effective bandwidth and needs no buffer resets.
 * - Sentinel: raw payloads are written into a buffer pre-filled with a
 *   sentinel word 0x80000000 (a -0.0 float, or a -0.0/+0.0 pair of 16-bit
 *   floats); senders replace that word by zeros and receivers reset slots
 *   after reading.
 *
 * Buffer reuse without barriers: scratch is double-buffered by the parity of
 * a per-block epoch kept in device memory (so CUDA graph replays advance it).
 * The grid size is fixed, hence all blocks of all ranks see the same epoch
 * sequence and every element is always handled by the same block. A rank can
 * only start epoch e+2 (same parity as e) after finishing e+1, which needs
 * data each peer sends only after it finished epoch e, so a slot is never
 * overwritten before it has been consumed.
 */
// Tuned on 2x H100 PCIe: larger grids only help messages above ~512KB, where
// other allreduce kernels are as fast, at the expense of small messages.
constexpr int kPushBlocks = 36;
constexpr int kPushThreads = 256;
static_assert(kPushBlocks <= kMaxPushBlocks);

struct __align__(16) PushBuffers {
  uint4* ptrs[kMaxCustomCollectiveRanks];
};

static DINLINE void st_volatile_v4(uint4* addr, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

static DINLINE uint4 ld_volatile_v4(const uint4* addr) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

static constexpr uint32_t kPushSentinel = 0x80000000U;

static DINLINE bool push_sentinel_ready(const uint4& v) {
  return v.x != kPushSentinel && v.y != kPushSentinel && v.z != kPushSentinel &&
         v.w != kPushSentinel;
}

static DINLINE uint4 push_sanitize(uint4 v) {
  v.x = v.x == kPushSentinel ? 0 : v.x;
  v.y = v.y == kPushSentinel ? 0 : v.y;
  v.z = v.z == kPushSentinel ? 0 : v.z;
  v.w = v.w == kPushSentinel ? 0 : v.w;
  return v;
}

// Scratch layout per rank, in 16B units: [parity][src_rank][pack][line] where
// LL uses 2 lines per 16B pack and sentinel uses 1.
template <typename T, int ngpus, bool Sentinel>
__global__ void __launch_bounds__(kPushThreads, 1)
    cross_device_reduce_push(PushBuffers bufs, Signal* self_sg,
                             const T* __restrict__ input,
                             T* __restrict__ result, int rank, int size,
                             int max_packs) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  static_assert(sizeof(P) == sizeof(uint4));
  constexpr int kLines = Sentinel ? 1 : 2;

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // With PDL, wait for the upstream grid (which may still be writing the
  // input or, if it is a push allreduce, the epochs and scratch), then let
  // the next kernel start its launch. It in turn waits for our completion,
  // so kernels still run in stream order.
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  #endif
  FlagType* epoch_ptr = &self_sg->push_epoch[Sentinel][blockIdx.x];
  // Epochs start at 1 so a zero-initialized LL buffer never looks ready.
  const uint32_t epoch = *epoch_ptr + 1;
  const int parity = epoch & 1;
  // Spread consecutive warps over blocks so small messages use many SMs,
  // while keeping each warp's accesses contiguous.
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int stride = gridDim.x * blockDim.x;
  const size_t region = static_cast<size_t>(max_packs) * kLines;
  uint4* self_buf = bufs.ptrs[rank] + parity * ngpus * region;

  const int first = (warp * gridDim.x + blockIdx.x) * 32 + lane;

  // Phase 1: push everything first, so that all stores are in flight at once
  // instead of paying one interconnect round trip per loop iteration.
  for (int idx = first; idx < size; idx += stride) {
    uint4 mine = reinterpret_cast<const uint4*>(input)[idx];
    if constexpr (Sentinel) mine = push_sanitize(mine);
    const size_t dst = (parity * ngpus + rank) * region + idx * kLines;
  #pragma unroll
    for (int i = 1; i < ngpus; i++) {
      int peer = (rank + i) % ngpus;
      if constexpr (Sentinel) {
        st_volatile_v4(bufs.ptrs[peer] + dst, mine);
      } else {
        st_volatile_v4(bufs.ptrs[peer] + dst,
                       make_uint4(mine.x, epoch, mine.y, epoch));
        st_volatile_v4(bufs.ptrs[peer] + dst + 1,
                       make_uint4(mine.z, epoch, mine.w, epoch));
      }
    }
  }

  // Phase 2: poll for the peers' contributions and reduce.
  for (int idx = first; idx < size; idx += stride) {
    uint4 got[ngpus];
    got[rank] = reinterpret_cast<const uint4*>(input)[idx];
    if constexpr (Sentinel) got[rank] = push_sanitize(got[rank]);
    // Poll all peers at once, re-reading until every slot is ready.
    bool ready;
    do {
      ready = true;
  #pragma unroll
      for (int r = 0; r < ngpus; r++) {
        if (r == rank) continue;
        const uint4* src = self_buf + r * region + idx * kLines;
        if constexpr (Sentinel) {
          got[r] = ld_volatile_v4(src);
          ready &= push_sentinel_ready(got[r]);
        } else {
          uint4 l0 = ld_volatile_v4(src), l1 = ld_volatile_v4(src + 1);
          ready &=
              l0.y == epoch && l0.w == epoch && l1.y == epoch && l1.w == epoch;
          got[r] = make_uint4(l0.x, l0.z, l1.x, l1.z);
        }
      }
    } while (!ready);

    if constexpr (Sentinel) {
      // Re-arm the slots for the next epoch with this parity.
      const uint4 s = make_uint4(kPushSentinel, kPushSentinel, kPushSentinel,
                                 kPushSentinel);
  #pragma unroll
      for (int r = 0; r < ngpus; r++) {
        if (r != rank) self_buf[r * region + idx] = s;
      }
    }

    // Reduce in rank order so that all ranks produce bitwise identical
    // results.
    A acc = upcast(*reinterpret_cast<P*>(&got[0]));
  #pragma unroll
    for (int r = 1; r < ngpus; r++) {
      packed_assign_add(acc, upcast(*reinterpret_cast<P*>(&got[r])));
    }
    reinterpret_cast<P*>(result)[idx] = downcast<P>(acc);
  }

  // Every thread has read the epoch above; publish the next one.
  __syncthreads();
  if (threadIdx.x == 0) *epoch_ptr = epoch;
}
#endif  // !defined(USE_ROCM)

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

#if !defined(USE_ROCM)
  // Scratch buffers of the push allreduce (LL and sentinel regions), or
  // max_push_packs_ == 0 if they were never registered.
  PushBuffers push_ll_{}, push_sentinel_{};
  int max_push_packs_ = 0;

  static size_t push_buffer_size(int world_size, size_t max_size) {
    // [parity][rank][pack] with 2 x 16B lines per pack for LL, 1 for sentinel.
    return 2 * world_size * max_size * (2 + 1);
  }

  /**
   * Register the IPC scratch buffers used by push_allreduce, one per rank,
   * each at least push_buffer_size(world_size, max_size) bytes and zeroed.
   * The caller must make sure that all ranks registered before any rank
   * calls push_allreduce.
   */
  void register_push_buffers(void** ptrs, size_t max_size) {
    if (max_size % 16 != 0)
      throw std::runtime_error("push allreduce max size must be 16B aligned");
    size_t max_packs = max_size / 16;
    size_t ll_units = 2 * world_size_ * max_packs * 2;
    for (int i = 0; i < world_size_; i++) {
      push_ll_.ptrs[i] = reinterpret_cast<uint4*>(ptrs[i]);
      push_sentinel_.ptrs[i] = reinterpret_cast<uint4*>(ptrs[i]) + ll_units;
    }
    // The LL region is zeroed by the allocator, fill the sentinel region.
    size_t sentinel_words = 2 * world_size_ * max_packs * 4;
    if (cuMemsetD32(reinterpret_cast<CUdeviceptr>(push_sentinel_.ptrs[rank_]),
                    kPushSentinel, sentinel_words) != CUDA_SUCCESS)
      throw std::runtime_error("failed to initialize push allreduce buffer");
    CUDACHECK(cudaDeviceSynchronize());
    max_push_packs_ = static_cast<int>(max_packs);
  }

  /**
   * Barrier-free one-shot push allreduce, see cross_device_reduce_push.
   * input needs no IPC registration.
   */
  template <typename T>
  void push_allreduce(cudaStream_t stream, const T* input, T* output, int size,
                      bool sentinel) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "push allreduce requires input length to be multiple of " +
          std::to_string(d));
    size /= d;
    if (size > max_push_packs_)
      throw std::runtime_error("push allreduce input of " +
                               std::to_string(size * 16) +
                               " bytes exceeds the registered scratch size");
    if (size == 0) return;
    // The grid must never change, see cross_device_reduce_push.
    cudaLaunchAttribute attributes[1]{};
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{.gridDim = dim3(kPushBlocks),
                              .blockDim = dim3(kPushThreads),
                              .dynamicSmemBytes = 0,
                              .stream = stream,
                              .attrs = attributes,
                              .numAttrs = 1};
  #define KL(ngpus)                                                           \
    if (sentinel) {                                                           \
      CUDACHECK(cudaLaunchKernelEx(                                           \
          &config, &cross_device_reduce_push<T, ngpus, true>, push_sentinel_, \
          self_sg_, input, output, rank_, size, max_push_packs_));            \
    } else {                                                                  \
      CUDACHECK(cudaLaunchKernelEx(                                           \
          &config, &cross_device_reduce_push<T, ngpus, false>, push_ll_,      \
          self_sg_, input, output, rank_, size, max_push_packs_));            \
    }
    switch (world_size_) {
      case 2:
        KL(2);
        break;
      case 4:
        KL(4);
        break;
      case 6:
        KL(6);
        break;
      case 8:
        KL(8);
        break;
      default:
        throw std::runtime_error(
            "push allreduce only supports num gpus in (2,4,6,8). Actual num "
            "gpus = " +
            std::to_string(world_size_));
    }
  #undef KL
  }
#endif  // !defined(USE_ROCM)

  void allgather(cudaStream_t stream, void* input, void* output, int size_bytes,
                 int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_allgather(cudaStream_t stream, T* input, T* output,
                               void* local_buffer, void* multicast_buffer,
                               uint32_t* epochs, int size_bytes,
                               int stage_size_bytes);
  template <typename T>
  void reduce_scatter(cudaStream_t stream, T* input, T* output, int size,
                      int threads = 512, int block_limit = defaultBlockLimit);
  template <typename T>
  void mnnvl_lamport_reduce_scatter(cudaStream_t stream, T* input, T* output,
                                    void* local_buffer, uint32_t* epochs,
                                    int size, int stage_size_bytes);

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 * add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 *                                                       half *, int, int, int);
 */
}  // namespace vllm
