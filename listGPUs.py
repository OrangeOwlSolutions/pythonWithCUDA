import pycuda.driver as cuda
import pycuda.autoinit

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
