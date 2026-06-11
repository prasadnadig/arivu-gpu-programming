# CUDA Programming Model: Grids, Blocks, Threads, and Warps

This document gives a clear definition of **grid**, alongside blocks, threads, and warps, and ties them to the GPU hardware hierarchy.

---

## 1. Core concepts in the CUDA programming model

In CUDA, when you launch a kernel, you specify:

```cpp
kernel<<<gridDim, blockDim>>>(...);
```

This defines a **grid** of **blocks**, where each block contains multiple **threads**. [web:35][web:39]

- **Thread**  
  The smallest unit of execution. Each thread runs the kernel code on one piece of data (for example, one pixel or one array element). Threads have indices such as `threadIdx.x`, `threadIdx.y`, `threadIdx.z`. [web:38][web:39]

- **Block (thread block)**  
  A group of threads that:
  - Run on the same SM.
  - Can synchronize via `__syncthreads()`.
  - Can cooperate via fast on‑chip **shared memory**.  
  Blocks have indices `blockIdx.x`, `blockIdx.y`, `blockIdx.z`. [web:35][web:38][web:39]

- **Warp**  
  A group of 32 threads that the hardware executes together in lockstep (SIMT).  
  Inside each block, threads are automatically grouped into warps of 32. The warp is the unit the SM’s warp scheduler actually issues to the cores. [web:35][web:38][web:39]

- **Grid** (your requested definition)  
  A **grid** is the full collection of thread blocks launched by a single kernel call. [web:35][web:36][web:39]  
  More precisely:
  - One kernel launch → one grid.
  - A grid is an ordered set of blocks with indices from `(0,0,0)` up to `(gridDim.x-1, gridDim.y-1, gridDim.z-1)`. [web:36][web:38]
  - Different grids can be active concurrently (for example, when using streams), but each grid is still “one launch = one kernel function”. [web:39]

You can think of it as:

```text
Grid
 ├─ Block (0,0,0)
 │   └─ Threads in that block
 ├─ Block (1,0,0)
 │   └─ Threads in that block
 └─ ...
```

---

## 2. How grid, blocks, threads map to hardware

### 2.1 GPU, SMs, and grids

On the hardware side, the GPU has multiple **Streaming Multiprocessors (SMs)**. [web:35][web:46]

- When you launch a kernel, you create a **grid** of blocks.
- The GPU runtime schedules those blocks on the available SMs.
- Each SM can run multiple blocks at the same time, up to limits like “max blocks per SM” and “max threads per SM”. [web:38][web:39]

You do **not** assign blocks to specific SMs; the hardware does that.

### 2.2 Blocks and warps inside an SM

Inside an SM:

- Every block assigned to that SM is split into **warps** of 32 threads. [web:35][web:39]
- Warps are stored in the SM’s scheduler queues.
- Each cycle, a warp scheduler in an SM partition picks a ready warp and issues its instruction to the core pipelines. [web:35][web:46]

So the mapping looks like:

```text
Grid (software)
 └─ Blocks
     └─ Threads

GPU (hardware)
 └─ SMs
     └─ Warps (32 threads each)
         └─ Executed on cores
```

---

## 3. Example: 1D vector add

Kernel:

```cpp
__global__ void vecAdd(const float* a,
                       const float* b,
                       float* c,
                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```

Launch:

```cpp
int N = 1'000'000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vecAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);
```

Here:

- **Grid**: `blocksPerGrid` 1D blocks (e.g., 3907 blocks). [web:24][web:42]
- **Each block**: 256 threads.
- **Within each block**: 256 / 32 = 8 warps. [web:35][web:39]

On the GPU:

- Blocks get distributed across SMs.
- Each SM holds some subset of those blocks at once.
- Inside an SM, all the threads of its resident blocks are scheduled as warps and executed on the SM’s cores. [web:35][web:38][web:39]

---

## 4. Summary mental model

- A **grid** is the whole “wave” of work for a single kernel launch.
- That grid is made up of **blocks**, each of which is a team of cooperating threads.
- The GPU executes those blocks on its SMs; inside each SM, threads run in groups of 32 called **warps**.
- Performance tuning typically means:
  - Choosing grid and block sizes to fully use SM resources.
  - Ensuring each block has enough threads (warps) and good memory access patterns. [web:35][web:39][web:42]