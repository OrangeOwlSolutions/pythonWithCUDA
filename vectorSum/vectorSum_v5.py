import numpy as np

# --- PyCUDA initialization
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    return a // b + 1

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

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

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)
d_c = gpuarray.zeros_like(d_a)

# --- Define a reference to the __global__ function and call it
deviceAdd = mod.get_function("deviceAdd")
blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (iDivUp(N, BLOCKSIZE), 1, 1)
start.record()
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = d_c.get()

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

# --- Flush context printf buffer
cuda.Context.synchronize()
