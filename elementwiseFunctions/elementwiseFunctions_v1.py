import numpy as np

# --- PyCUDA initialization
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.cumath as cumath

########
# MAIN #
########

start = cuda.Event()
end   = cuda.Event()

N = 10

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
d_c = (cumath.sqrt(cumath.fabs(d_a)) + cumath.exp(d_b))
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = %fs" % (secs))

h_c = d_c.get()

if np.all(abs(h_c - (np.sqrt(np.abs(h_a)) + np.exp(h_b))) < 1e-5):
  print("Test passed!")
else :
  print("Error!")
