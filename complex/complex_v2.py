import numpy as np

# --- PyCUDA initialization
import pycuda.gpuarray as gpuarray
import pycuda.cumath   as cumath
import pycuda.driver   as cuda
import pycuda.autoinit

########
# MAIN #
########

N = 10

# --- Create random vectorson the CPU
h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

# --- Set CPU arrays as single precision
h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)
h_c = np.empty_like(h_a, dtype = np.complex64)

d_a = gpuarray.to_gpu(h_a)
d_b = gpuarray.to_gpu(h_b)

d_c = d_a * cumath.exp(1j * d_b)

h_c = d_c.get()

if np.array_equal(h_c, h_a * np.exp(1j * h_b)):
  print("Test passed!")
else :
  print("Error!")

print(h_c)
print(h_a * np.exp(1j * h_b))

# --- Flush context printf buffer
cuda.Context.synchronize()
