# Multi‑GPU Data‑Parallel Processing with Batches

This document assumes you already understand grids, blocks, threads, and warps, and focuses on **how work is split across multiple GPU devices** when you have batched data.

---

## 1. High‑level idea: data parallelism

With multiple GPUs, the most common strategy is **data parallelism**: [web:40][web:43]

- Each GPU runs **the same kernel** (same grid configuration).
- The **input data is partitioned** into disjoint chunks or batches.
- Each GPU processes its own chunk independently.
- Results are combined or reduced on the host (or via collective communication if doing distributed training).

In other words:

```text
Full dataset --> split into N chunks
Chunk 0 -> GPU 0 (runs kernel on its chunk)
Chunk 1 -> GPU 1 (runs kernel on its chunk)
...
Chunk N-1 -> GPU N-1
```

Inside each GPU, the usual single‑GPU story applies: blocks are scheduled on that device’s SMs.

---

## 2. Practical single‑process, multi‑GPU pattern

A common pattern in CUDA C/C++ is: [web:41][web:44]

1. Query the number of devices:

   ```cpp
   int numDevices = 0;
   cudaGetDeviceCount(&numDevices);
   ```

2. Decide how to split your global batch:

   ```cpp
   int totalN = ...;                  // total elements or samples
   int N_per_gpu = (totalN + numDevices - 1) / numDevices;
   ```

3. Loop over devices and launch the same kernel on each:

   ```cpp
   for (int dev = 0; dev < numDevices; ++dev) {
       cudaSetDevice(dev);           // select GPU dev

       int offset = dev * N_per_gpu;
       int N_local = std::min(N_per_gpu, totalN - offset);

       // Allocate device buffers of size N_local on this GPU
       // Copy this GPU's slice of the data (host[offset .. offset+N_local-1])

       int threadsPerBlock = 256;
       int blocksPerGrid = (N_local + threadsPerBlock - 1) / threadsPerBlock;

       vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N_local);

       // Optionally async: cudaMemcpyAsync results back to host[offset..]
   }
   ```

4. Synchronize and gather results.

This uses **one CPU process and one thread**, which selects each GPU in turn and launches a separate grid for each device. [web:41][web:44]

---

## 3. How grids look across devices

For a vector addition example:

- Suppose `totalN = 1,000,000` and you have 2 GPUs.  
- You choose `N_per_gpu = 500,000` and `threadsPerBlock = 256`.

On each GPU:

- GPU 0:
  - Data slice: elements `[0 .. 499,999]`.
  - `blocksPerGrid = ceil(500000 / 256) ≈ 1954`.
  - Launch `vecAdd<<<1954, 256>>>(...)` on GPU 0.
- GPU 1:
  - Data slice: elements `[500,000 .. 999,999]`.
  - `blocksPerGrid = 1954`.
  - Launch the same kernel with an offset or with pointers already pointing to the right slice, on GPU 1.

Key points:

- **Each GPU has its own grid**, independent of the others.
- Inside each device, that grid’s blocks are scheduled on that device’s SMs in the usual way.
- From the programmer’s perspective, you’re just launching **the same kernel many times**, once per GPU, with different data and sometimes different grid sizes. [web:41][web:44]

---

## 4. Batching strategies

How you “batch” data across GPUs depends on your workload.

### 4.1 Equal-size data batches per GPU

Simplest case (homogeneous GPUs): [web:40][web:43]

- Split the global batch into equal chunks: `global_batch / num_gpus`.
- Launch the same grid configuration on each GPU for its local batch.

This works well when:

- GPUs are the same model.
- Data is uniform and each element costs about the same to process.

### 4.2 Load‑balanced or performance‑aware batching

When GPUs differ (e.g., one A100 and one smaller GPU), or data chunks have different runtimes, you may:

- Assign more data to the faster GPU. [web:40][web:44]
- Use a **work queue**:
  - Host maintains a queue of tasks or mini‑batches.
  - Each GPU pulls new work when it finishes the previous batch.
  - This can be done with multiple host threads or with asynchronous streams and callbacks. [web:44]

### 4.3 Mini‑batches inside each GPU

Even on a single GPU, you may break its local slice into smaller **mini‑batches**:

- For example, GPU 0 gets 500k elements, but you process them in 10 chunks of 50k each.
- Each mini‑batch corresponds to a kernel launch (a grid).
- This can help:
  - Overlap data transfer and compute.
  - Reduce peak memory needs.

At the multi‑GPU level, you then have:

```text
Global batch
 ├─ GPU 0 chunk
 │   └─ mini-batches processed by multiple grids
 ├─ GPU 1 chunk
 │   └─ mini-batches processed by multiple grids
 └─ ...
```

---

## 5. Memory movement and streams

### 5.1 Host–device transfers per GPU

For each GPU you typically:

1. `cudaSetDevice(dev)`
2. Allocate `d_input`, `d_output` on that device.
3. Copy that device’s slice of data from host to device.
4. Launch kernels (possibly in streams).
5. Copy results back. [web:41][web:44]

These transfers happen **per device**, and you can often overlap copies and computation using **streams**:

- Use `cudaMemcpyAsync` into a stream on GPU `dev`.
- Launch kernels in the same or another stream.
- Use events to synchronize when needed. [web:41]

### 5.2 Peer‑to‑peer and inter‑GPU communication

For purely data‑parallel vector add, GPUs don’t need to talk to each other during compute. But in some workloads:

- You might use **peer‑to‑peer (P2P)** copies (`cudaMemcpyPeer`) to move data between GPUs.
- In deep learning, you may use libraries (NCCL, frameworks) for all‑reduce of gradients across devices. [web:40][web:43]

At the abstraction level of grids/blocks/threads, each GPU still sees only **its own grid**; synchronization across devices is handled at a higher level (host code or communication libraries).

---

## 6. Common multi‑GPU patterns

Two common patterns in practice: [web:40][web:41][web:43][web:44]

1. **One host thread/process controls all GPUs**  
   - Single main loop:
     - For each GPU:
       - `cudaSetDevice`
       - Manage memory, launch kernels for this batch.
   - Simpler to implement; good when per‑GPU work is large.

2. **One process (or thread) per GPU**  
   - Popular in MPI or distributed training setups:
     - Each process is bound to a specific device.
     - Input data is already partitioned across processes.
   - Fits naturally with frameworks that expect “one rank per GPU”. [web:41][web:43]

In both patterns, **the per‑GPU logic is identical**; you just give each instance its own portion of data and possibly its own seed or configuration.

---

## 7. How this relates to the single‑GPU execution hierarchy

Even in a multi‑GPU, batched setup:

- Each **GPU** runs its own **grids**.
- Inside each grid, **blocks and threads** behave exactly as in the single‑GPU case.
- Each device’s SMs, SM partitions, and cores are only aware of **their own work**; they don’t see other GPUs. [web:35][web:39][web:45]

Multi‑GPU programming is therefore **layered**:

- **Inside one device**: think about grids, blocks, threads, warps, memory hierarchy.
- **Across devices**: think about how to:
  - Split data across GPUs.
  - Launch one or more grids per device.
  - Coordinate data transfers and result aggregation.

If you keep these two layers conceptually separate, it’s much easier to scale a single‑GPU kernel to multiple devices while reusing the same kernel code.