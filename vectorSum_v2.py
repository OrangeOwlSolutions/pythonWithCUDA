# --- PyCuda initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

def iDivUp(a, b):
    return a // b + 1

N = 10000

BLOCKSIZE = 256

# --- Create random vectorson the CPU
h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

# --- Set CPU arrays as single precision
h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)
h_c = np.empty_like(h_a)

mod = SourceModule("""
__global__ void deviceAdd(float * __restrict__ d_c, const float * __restrict__ d_a, 
                                                    const float * __restrict__ d_b,
                                                    const int N)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= N) return;
  d_c[tid] = d_a[tid] + d_b[tid];
}
""")

# --- Define a reference to the __global__ function and call it
deviceAdd = mod.get_function("deviceAdd")
blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (iDivUp(N, BLOCKSIZE), 1, 1)
deviceAdd(cuda.Out(h_c), cuda.In(h_a), cuda.In(h_b), np.int32(N), block = blockDim, grid = gridDim)

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")
