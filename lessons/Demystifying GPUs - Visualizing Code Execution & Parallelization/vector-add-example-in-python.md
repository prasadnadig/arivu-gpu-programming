# Vector Addition on GPU in Python (with Multi‑GPU Notes)

This document shows how to implement the earlier **vector addition** example in Python using CUDA via **Numba**, and explains how this fits into the “Python view” of single‑ and multi‑GPU usage. The focus is on:

- A clear Python implementation of vector addition on one GPU.
- How this maps to grids, blocks, and threads.
- How to extend the same idea to **multiple GPUs with batched data**.

---

## 1. Tools and environment

We will use **Numba CUDA** because it lets us write CUDA kernels directly in Python: [web:48][web:49][web:55]

- `@cuda.jit` to define a GPU kernel.
- Python code on the host to:
  - Allocate device arrays.
  - Launch kernels with `kernel[grid_dim, block_dim](...)`.
  - Copy results back.

You will need:

```bash
pip install numba numpy
# plus a working NVIDIA driver + CUDA toolkit
```

---

## 2. Single‑GPU vector addition in Python

### 2.1 Kernel: one thread per element

```python
import numpy as np
from numba import cuda

@cuda.jit
def vec_add(a, b, c, N):
    # 1D global thread index (like blockIdx.x * blockDim.x + threadIdx.x)
    idx = cuda.grid(1)

    if idx < N:
        c[idx] = a[idx] + b[idx]
```

Key points: [web:48][web:49][web:55]

- `cuda.grid(1)` computes a 1D global index from the grid and block indices (same formula you saw in the C++ example).
- Each **thread** handles at most one element `idx`.
- The kernel corresponds to the earlier C++ `vecAdd` kernel, but written in Python.

### 2.2 Host code: configure grid and launch

```python
def run_single_gpu_vector_add(N=1_000_000):
    # ----- 1. Create input arrays on host (CPU) -----
    a_host = np.random.rand(N).astype(np.float32)
    b_host = np.random.rand(N).astype(np.float32)

    # Output array on host
    c_host = np.empty_like(a_host)

    # ----- 2. Allocate device arrays -----
    a_dev = cuda.to_device(a_host)
    b_dev = cuda.to_device(b_host)
    c_dev = cuda.device_array_like(a_host)

    # ----- 3. Configure blocks and grid -----
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # ----- 4. Launch kernel on the current GPU -----
    vec_add[blocks_per_grid, threads_per_block](a_dev, b_dev, c_dev, N)

    # Wait for the kernel to complete
    cuda.synchronize()

    # ----- 5. Copy result back to host -----
    c_dev.copy_to_host(c_host)

    # ----- 6. Verify correctness -----
    expected = a_host + b_host
    max_err = np.max(np.abs(c_host - expected))
    print(f"Max error: {max_err:.6e}")

    return c_host
```

How this maps to the CUDA model: [web:48][web:49]

- `threads_per_block = 256`  
  - Each block has 256 threads.
  - Each block is internally divided into 8 warps (256 / 32).
- `blocks_per_grid` is chosen so that `blocks_per_grid * threads_per_block >= N`.  
Together, all blocks form the **grid** of the kernel launch.
- `vec_add[blocks_per_grid, threads_per_block](...)` is equivalent to `vecAdd<<<blocksPerGrid, threadsPerBlock>>>(...)` in CUDA C++.

---

## 3. Relating this to grids, blocks, and threads

In this Python version, NumPy + Numba give you a high‑level interface, but under the hood the same layout applies: [web:48][web:49][web:55]

- **Grid**: `blocks_per_grid` blocks.
- **Block**: `threads_per_block` threads.
- **Thread**: each one computes a global index `idx` and performs `c[idx] = a[idx] + b[idx]`.

Conceptual picture (1D):

```text
Grid
 ├─ Block 0
 │   ├─ Thread 0  -> idx = 0
 │   ├─ Thread 1  -> idx = 1
 │   └─ ...
 ├─ Block 1
 │   ├─ Thread 0  -> idx = blockDim + 0
 │   └─ ...
 └─ Block (blocks_per_grid - 1)
```

The GPU hardware then takes that grid and:

- Assigns blocks to SMs.
- Splits each block into warps of 32 threads.
- Schedules warps on SM partitions and cores, as discussed in your earlier documents.

---

## 4. How this looks “in Python terms”

Python code is executing on the **host (CPU)**:

- It prepares data using NumPy arrays.
- It moves data to the device with `cuda.to_device`.
- It launches a kernel by indexing the `@cuda.jit` function with `[grid, block]`.

From the Python point of view:

- You do **not** manually manage SMs or warps.
- You choose the grid and block sizes; Numba and CUDA handle the mapping to hardware. [web:48][web:49]

So the **Python mental model** is:

```text
Python host code
 ├─ Prepare arrays
 ├─ cuda.to_device(...)        (host → GPU memory)
 ├─ kernel[grid, block](...)   (tell GPU to run a grid of blocks)
 └─ result = device_array.copy_to_host()
```

---

## 5. Extending the vector add to multiple GPUs (batched data)

Now let’s connect this to your **multi‑GPU, batched data** question, still staying in Python.

### 5.1 Strategy: data parallelism over GPUs

We follow a data‑parallel approach, similar to libraries and training frameworks: [web:40][web:41][web:44][web:57]

- Split the global vector into **chunks**, one per GPU.
- On each GPU:
  - Copy its chunk to that GPU.
  - Launch `vec_add` on that chunk.
  - Copy its chunk of results back.
- Optionally use Python threads or async streams to run those steps in parallel.

### 5.2 Multi‑GPU vector add example (Numba)

```python
from numba import cuda
import threading

def vec_add_multi_gpu(a_host, b_host):
    """
    a_host, b_host: 1D NumPy arrays of the same length (float32).
    Returns: c_host = a_host + b_host, using all visible GPUs.
    """
    assert a_host.shape == b_host.shape
    N_total = a_host.size

    num_devices = len(cuda.gpus)
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    # Split indices into chunks (one per GPU)
    # Use nearly-equal partitioning
    chunk_sizes = []
    base = N_total // num_devices
    remainder = N_total % num_devices
    start = 0
    chunks = []

    for dev_id in range(num_devices):
        size = base + (1 if dev_id < remainder else 0)
        end = start + size
        chunks.append((start, end))
        start = end

    # Output array on host
    c_host = np.empty_like(a_host)

    # Worker function for one GPU
    def worker(dev_id, start, end):
        cuda.select_device(dev_id)  # choose GPU dev_id
        N_local = end - start

        # Host slices for this GPU
        a_slice = a_host[start:end]
        b_slice = b_host[start:end]

        # Copy to this device
        a_dev = cuda.to_device(a_slice)
        b_dev = cuda.to_device(b_slice)
        c_dev = cuda.device_array_like(a_slice)

        # Configure grid for local slice
        threads_per_block = 256
        blocks_per_grid = (N_local + threads_per_block - 1) // threads_per_block

        # Launch kernel on this device
        vec_add[blocks_per_grid, threads_per_block](a_dev, b_dev, c_dev, N_local)

        # Synchronize and copy results back into correct host slice
        cuda.synchronize()
        c_dev.copy_to_host(c_host[start:end])

        # Reset device context if desired (optional)
        cuda.close()

    # Launch one thread per GPU
    threads = []
    for dev_id, (start, end) in enumerate(chunks):
        t = threading.Thread(target=worker, args=(dev_id, start, end))
        t.start()
        threads.append(t)

    # Wait for all GPUs to finish
    for t in threads:
        t.join()

    return c_host
```

What is happening here: [web:41][web:44][web:48][web:49]

- We query `cuda.gpus` to find all available devices.
- We partition the 1D input range `[0, N_total)` into `num_devices` contiguous subranges.
- For each device:
  - A Python thread selects that device (`cuda.select_device(dev_id)`).
  - Copies its slice of `a_host` and `b_host` to that device.
  - Launches **the same grid** (with size based on the local slice).
  - Copies back its slice of the result into `c_host[start:end]`.

From each GPU’s perspective:

- It only sees its own local data and its own grid of blocks.
- The implementation inside the GPU is exactly the same as in the single‑GPU case.

### 5.3 Verifying multi‑GPU correctness

You can test it like this:

```python
def test_multi_gpu():
    N = 10_000_000
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)

    c_multi = vec_add_multi_gpu(a, b)
    c_ref = a + b

    max_err = np.max(np.abs(c_multi - c_ref))
    print(f"Multi-GPU max error: {max_err:.6e}")
```

---

## 6. Mental model summary (Python + multi‑GPU)

- **Single GPU (Numba)**:
  - You write a kernel with `@cuda.jit` and use `cuda.grid(1)` inside.
  - You launch it with `[grid_dim, block_dim]`.
  - NumPy/Numba manage memory copies and kernel invocation. [web:48][web:49][web:55]
- **Multiple GPUs**:
  - You partition your data (e.g., a vector) into slices.
  - For each GPU:
    - Select device.
    - Copy its slice.
    - Launch the same kernel with grid sized for that slice.
    - Copy back its part of the result.
  - At the end, the host array holds the combined result.

All the previously discussed GPU internals (SMs, partitions, warps, cores) still apply **inside each device**, but Python lets you think primarily in terms of:

- Arrays on host vs device.
- Grid/block sizes for each kernel launch.
- Data partitioning across devices for multi‑GPU work.

