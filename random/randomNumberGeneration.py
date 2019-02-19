import time
import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

code = """
    #include <curand_kernel.h>

    const int nstates = %(NGENERATORS)s;
    __device__ curandState_t states[nstates];

    __global__ void initkernel(int seed)
    {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates) curand_init(seed, tidx, 0, states + tidx);
    }

    __global__ void randfillkernel(float *values, int N)
    {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;

        if (tidx < nstates) {
            for (int i = tidx; i < N; i += blockDim.x * gridDim.x) {
                values[i] = curand_uniform(states + tidx);
            }
        }
    }
"""

N = 1024
mod = SourceModule(code % { "NGENERATORS" : N }, no_extern_c = True, arch = "sm_52")
init_func = mod.get_function("_Z10initkerneli")
fill_func = mod.get_function("_Z14randfillkernelPfi")

seed = np.int32(time.time())
nvalues = 10 * N
init_func(seed, block=(N, 1, 1), grid=(1, 1, 1))
gdata = gpuarray.zeros(nvalues, dtype = np.float32)
fill_func(gdata, np.int32(nvalues), block = (N, 1, 1), grid = (1, 1, 1))
print(gdata)
