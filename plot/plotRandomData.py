import pycuda.autoinit
import pycuda.curandom as curandom
import matplotlib.pyplot as plt
import numpy as np

N = 1000
# --- Generating random data directly on the gpu
d_a = curandom.rand(N, dtype = np.float32, stream = None)

plt.plot(d_a.get())
plt.xlabel('Realization index')
plt.ylabel('Random numbers')

save_file = 0
if save_file:
    plt.savefig('test')
else:
    plt.show()
