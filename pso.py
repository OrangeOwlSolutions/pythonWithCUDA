import random
import numpy as np

if __name__ == "__PSO__":
    main()

###############################
# PARTICLE SWARM OPTIMIZATION #
###############################
def pso(fobj, bounds, Np = 20, iter = 1000, verbose = True, w = 0.5, c1 = 1, c2 = 2):
    D               = len(bounds)
    min_b, max_b    = np.asarray(bounds).T
    diff            = np.fabs(min_b - max_b)
    h_popNormalized = np.random.rand(Np, D)
    h_pop           = min_b + h_popNormalized * diff
    # --- Initialize old objective values
    h_fobjOld       = np.asarray([float("inf") for ind in h_pop])
    # --- Evaluate the fitness over population members
    h_fobj          = np.asarray([fobj(ind) for ind in h_pop])
    # --- Index of the best population member
    h_best_idx      = np.argmin(h_fobj)
    # --- Best population member
    h_best          = h_pop[h_best_idx]
    h_bestObj       = h_fobj[h_best_idx]
    # --- Best personal population members
    h_personalBest  = h_pop
    h_personalBestObj = h_fobj
    # --- Initialize the velocity
    velocity        = np.random.rand(Np, D)
    velocity        = min_b + velocity * diff
    # --- Algorithm loop over the iterations
    for i in range(iter):
        if verbose: print(f'iter: {i:>4d}, best solution: {h_bestObj:10.6f}')
        # --- Algorithm loop over the population members
        r1 = random.random()
        r2 = random.random()
        # --- Update velocity
        cognitiveVelocityComponent  = c1 * r1 * (h_personalBest - h_pop)
        socialVelocityComponent     = c2 * r2 * (h_best - h_pop)
        velocity                    = w * velocity + cognitiveVelocityComponent + socialVelocityComponent
        # --- Update position
        h_pop                       = h_pop + velocity
        # --- Evaluate the fitness over population members
        h_fobj                      = np.asarray([fobj(ind) for ind in h_pop])
        for j in range(Np):
            # --- Update local best
            if h_fobj[j] < h_personalBestObj[j]:
                h_personalBest[j]           = h_pop[j]
                h_personalBestObj[j]        = h_fobj[j]
            # --- Update global best
            if h_fobj[j] < h_bestObj:
                h_best_idx  = j
                h_best      = h_pop[j]
                h_bestObj   = h_fobj[j]

        yield h_best, h_fobj[h_best_idx]

##############################
# SPHERICAL TESTING FUNCTION #
##############################
def spherical(x):
  value = 0
  for i in range(len(x)):
      value += x[i]**2
  return value / len(x)

###############################
# ROSENBROCK TESTING FUNCTION #
###############################
# https://en.wikipedia.org/wiki/Rosenbrock_function
def rosenbrock(x):
    a = 1. - x[0]
    b = x[1] - x[0] * x[0]
    return a * a + b * b * 100.

########
# MAIN #
########
bounds  = [(-10, 10)] * 2
#Np      = 20

#pso(spherical, bounds)
result = list(pso(spherical, bounds))
#result = list(pso(rosenbrock, bounds))
print(result[-1])
