# Beginner Guide: Vector Addition on GPU with Numba and `cuda.jit`

This guide walks through a **minimal but realistic** example of using Numba’s CUDA support to run a **vector addition** on your GPU using Python. It focuses on short code snippets and explains each concept from a developer’s point of view.

---

## 1. What you need and what you get

### 1.1 Why Numba + CUDA?

Numba can compile selected Python functions into GPU kernels that run on NVIDIA GPUs. You write Python, decorate with `@cuda.jit`, and Numba generates CUDA code under the hood. [web:48][web:49][web:61]

We will:

- Add two vectors `a` and `b` to produce `c`.
- Map **one element per GPU thread**.
- Control the number of **threads per block** and **blocks per grid**.

### 1.2 Install dependencies

```bash
pip install numba numpy
```

You also need:

- A supported NVIDIA GPU.
- Working CUDA driver (and usually CUDA toolkit).

---

## 2. CPU baseline: simple vector add

Start with pure NumPy so you know what you are trying to accelerate.

```python
import numpy as np

def vec_add_cpu(a, b):
    return a + b

N = 1_000_000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = vec_add_cpu(a, b)
```

This runs on the **CPU** only.

---

## 3. First CUDA kernel with `@cuda.jit`

### 3.1 Basic kernel

```python
from numba import cuda

@cuda.jit
def vec_add_gpu(a, b, c, N):
    # 1D global thread index: like blockIdx.x * blockDim.x + threadIdx.x
    idx = cuda.grid(1)

    # Guard against going out of bounds
    if idx < N:
        c[idx] = a[idx] + b[idx]
```

Concepts: [web:48][web:42][web:60]

- `@cuda.jit`  
Tells Numba “compile this as a CUDA kernel”. The function will run in **parallel on the GPU**.
- `cuda.grid(1)`  
Computes a unique 1D index for each thread based on:
  - `blockIdx.x`, `blockDim.x`, `threadIdx.x` (the CUDA execution model).
- `if idx < N:`  
Threads are launched in blocks of 32/64/128/etc. The guard ensures that threads with an index beyond the vector size do nothing (avoids out‑of‑bounds access).

Think of `vec_add_gpu` as the GPU version of `vec_add_cpu`, but written in CUDA‑style Python.

---

## 4. Launching the kernel from Python

### 4.1 Setting up device memory

```python
import numpy as np
from numba import cuda

def run_vec_add_on_gpu(N=1_000_000):
    # ----- 1. Host (CPU) arrays -----
    a_host = np.random.rand(N).astype(np.float32)
    b_host = np.random.rand(N).astype(np.float32)
    c_host = np.empty_like(a_host)

    # ----- 2. Copy inputs to device (GPU) -----
    a_dev = cuda.to_device(a_host)          # alloc + copy
    b_dev = cuda.to_device(b_host)          # alloc + copy
    c_dev = cuda.device_array_like(a_host)  # alloc only
```

Concepts: [web:64][web:61]

- `cuda.to_device(array)`  
Allocates GPU memory and copies the NumPy array from host to device.
- `cuda.device_array_like(array)`  
Allocates a device array with the same shape and dtype as a given host array (no copy).

### 4.2 Choosing blocks and grid size

```python
    # ----- 3. Configure blocks and grid -----
    threads_per_block = 256              # typical starting point
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
```

Concepts: [web:42][web:60]

- **Threads per block** (`threads_per_block`):
  - Number of threads in each block.
  - Common choices: 128, 256, 512.
  - Must not exceed the GPU's max threads/block (often 1024).
- **Blocks per grid** (`blocks_per_grid`):
  - Number of blocks in the grid.
  - We choose enough blocks so that `blocks_per_grid * threads_per_block >= N`.

Together, **grid × block = total number of threads** launched.

### 4.3 Launch and synchronize

```python
    # ----- 4. Launch the kernel -----
    vec_add_gpu[blocks_per_grid, threads_per_block](a_dev, b_dev, c_dev, N)

    # Wait for GPU work to finish
    cuda.synchronize()

    # ----- 5. Copy result back to host -----
    c_dev.copy_to_host(c_host)

    # ----- 6. Verify correctness -----
    expected = a_host + b_host
    max_err = np.max(np.abs(c_host - expected))
    print(f"Max error: {max_err:.6e}")

    return c_host
```

Concepts: [web:48][web:42][web:60]

- `kernel[grid_dim, block_dim](...)`  
Special Numba syntax that corresponds to CUDA’s `kernel<<<gridDim, blockDim>>>(...)`.  
Here:
  - `grid_dim = blocks_per_grid`
  - `block_dim = threads_per_block`
- `cuda.synchronize()`  
Waits until all work on the current device is finished.
- `device_array.copy_to_host(host_array)`  
Copies results back to CPU memory.

---

## 5. Understanding the indexing logic

Let’s unpack how `cuda.grid(1)` gives every thread a unique index.

Under the hood:

```python
idx = (
    cuda.blockIdx.x * cuda.blockDim.x +
    cuda.threadIdx.x
)
```

Where: [web:42][web:60]

- `cuda.blockIdx.x`  is the current block index in the grid (0, 1, 2, …).
- `cuda.blockDim.x`  is the number of threads per block (our `threads_per_block`).
- `cuda.threadIdx.x` is the thread index inside the block (0…blockDim.x-1).

Example:

- `threads_per_block = 256`
- `blocks_per_grid = 4` (for small N)
- Then:

```text
Block 0: threadIdx.x = 0..255 -> idx = 0..255
Block 1: threadIdx.x = 0..255 -> idx = 256..511
Block 2: threadIdx.x = 0..255 -> idx = 512..767
Block 3: threadIdx.x = 0..255 -> idx = 768..1023
```

The `if idx < N` guard simply ignores extra threads when `N` is not exactly a multiple of `threads_per_block * blocks_per_grid`.

---

## 6. Tying to the hardware model (briefly)

Even though you stay in Python, Numba uses the **same execution model as CUDA C++**: [web:48][web:42][web:60]

- Your kernel sees:
  - **Grid** of blocks (`blocks_per_grid`).
  - **Block** with `threads_per_block` threads.
  - `cuda.grid(1)` is your per-thread ID.
- On the GPU:
  - Blocks are scheduled across SMs.
  - Each block is internally broken into **warps** of 32 threads.
  - Warp schedulers on SM partitions execute warps on the cores.

You don’t see SMs/warps in Python, but your choice of **block size** and **grid size** directly affects how many threads and warps the GPU can run concurrently.

---

## 7. A complete minimal script

Here is a single file you can run end‑to‑end:

```python
import numpy as np
from numba import cuda

@cuda.jit
def vec_add_gpu(a, b, c, N):
    idx = cuda.grid(1)
    if idx < N:
        c[idx] = a[idx] + b[idx]

def main():
    N = 1_000_000

    # Host arrays
    a_host = np.random.rand(N).astype(np.float32)
    b_host = np.random.rand(N).astype(np.float32)
    c_host = np.empty_like(a_host)

    # Device arrays
    a_dev = cuda.to_device(a_host)
    b_dev = cuda.to_device(b_host)
    c_dev = cuda.device_array_like(a_host)

    # Configure the kernel
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Launch
    vec_add_gpu[blocks_per_grid, threads_per_block](a_dev, b_dev, c_dev, N)
    cuda.synchronize()

    # Copy back and verify
    c_dev.copy_to_host(c_host)
    expected = a_host + b_host
    max_err = np.max(np.abs(c_host - expected))
    print(f"Max error: {max_err:.6e}")

if __name__ == "__main__":
    main()
```

---

## 8. Where to go next

Once you are comfortable with this vector add example, good next steps are: [web:48][web:52][web:62]

- Try different `threads_per_block` values (128, 256, 512) and measure performance.
- Use `cuda.grid(2)` and `cuda.grid(3)` for 2D/3D problems.
- Explore shared memory (`cuda.shared.array`) for more complex kernels.
- Read the Numba CUDA “Writing Kernels” and examples sections for more patterns.

The core pattern, however, remains the same:

1. Write a Python function with `@cuda.jit`.
2. Use `cuda.grid(ndim)` to get a per-thread index.
3. Configure `[blocks, threads]`.
4. Manage host/device memory with `cuda.to_device` and `device_array_like`.

Once you have that pattern internalized, you can start building much more complex GPU kernels in Python.