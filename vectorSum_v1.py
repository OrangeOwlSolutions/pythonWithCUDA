# --- PyCuda initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

###################
# iDivUp FUNCTION #
###################
def iDivUp(a, b):
    return a // b + 1

###################
# iDivUp FUNCTION #
###################
def listGPUs():
  print(cuda.Device.count(), " device(s) found.")

  for gpuID in range(cuda.Device.count()):
    dev = cuda.Device(gpuID)
    print("Device #%d: %s" % (gpuID, dev.name()))
    print("  Compute Capability: %d.%d" % dev.compute_capability())
    print("  Total Memory: %s KB" % (dev.total_memory()//(1024)))
    atts = [(str(att), value)
            for att, value in dev.get_attributes().items()]
    atts.sort()
    
    for att, value in atts:
        print("  %s: %s" % (att, value))

########
# MAIN #
########

listGPUs()

start = cuda.Event()
end   = cuda.Event()

N = 10

BLOCKSIZE = 256

# --- Create random vectorson the CPU
h_a = np.random.randn(1, N)
h_b = np.random.randn(1, N)

# --- Set CPU arrays as single precision
h_a = h_a.astype(np.float32)
h_b = h_b.astype(np.float32)

# --- Allocate GPU device memory
d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
d_c = cuda.mem_alloc(h_a.nbytes)

# --- Memcopy from host to device
cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)

mod = SourceModule("""
  #include <stdio.h>
  __global__ void deviceAdd(float * __restrict__ d_c, const float * __restrict__ d_a, const float * __restrict__ d_b, const int N)
  {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    d_c[tid] = d_a[tid] + d_b[tid];
    //printf("tid %d; %f + %f = %f \\n", tid, d_a[tid], d_b[tid], d_c[tid]);
  } 
  """)

# --- Define a reference to the __global__ function and call it
deviceAdd = mod.get_function("deviceAdd")
blockDim  = (BLOCKSIZE, 1, 1)
gridDim   = (iDivUp(N, BLOCKSIZE), 1, 1)
start.record()
deviceAdd(d_c, d_a, d_b, np.int32(N), block = blockDim, grid = gridDim)
end.record() 
end.synchronize()
secs = start.time_till(end) * 1e-3
print("Processing time = ", secs, " seconds")


# --- Copy results from device to host
h_c = np.empty_like(h_a)
cuda.memcpy_dtoh(h_c, d_c)

if np.array_equal(h_c, h_a + h_b):
  print("Test passed!")
else :
  print("Error!")

# Flush context printf buffer
cuda.Context.synchronize()
