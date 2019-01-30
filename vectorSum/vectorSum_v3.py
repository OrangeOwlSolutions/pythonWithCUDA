# --- PyCuda initialization
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

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

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)

start.record()
d_c = (d_a + d_b)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = d_c.get()

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")
