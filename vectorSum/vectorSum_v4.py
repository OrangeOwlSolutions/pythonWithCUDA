import numpy as np

# --- PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 100000

# --- Create random vectorson the CPU
h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

# --- Set CPU arrays as single precision
h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)
d_c = gpuarray.empty_like(d_a)

from pycuda.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(
        "float *d_c, float *d_a, float *d_b, float a, float b",
        "d_c[i] = a * d_a[i] + b * d_b[i]",
        "linear_combination")

start.record()
lin_comb(d_c, d_a, d_b, 2, 3)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

# --- Copy results from device to host
h_c = d_c.get()

if np.array_equal(h_c, 2 * h_a + 3 * h_b):
  print("Test passed!")
else :
  print("Error!")

# --- Flush context printf buffer
cuda.Context.synchronize()
