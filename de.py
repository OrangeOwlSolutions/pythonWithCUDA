# http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2013/Definitions%20of%20%20CEC%2013%20benchmark%20suite%200117.pdf
# https://www.mat.univie.ac.at/~neum/glopt/test.html
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.641.3566&rep=rep1&type=pdf

import numpy as np

#######################
# GENERATE POPULATION #
#######################

#######################################
# DIFFERENTIAL EVOLUTIONARY ALGORITHM #
#######################################
def de(fobj, bounds, F = 0.8, CR = 0.7, Np = 20, iter = 1000, verbose = True):
    # --- Np = Number of individuals in the population (Np >= 4 for mutation purposes)
    # --- F  = Mutation factor. A larger mutation factor increases the search radius but may slowdown 
    #     the convergence of the algorithm. Values for F are usually chosen from the interval [0.5, 2.0]. 
    #     The default value is set to a typical value of F = 0.8.
    # --- Dimensionality of each individual (number of unknowns)
    D               = len(bounds)
    # --- Compute bounds
    min_b, max_b    = np.asarray(bounds).T
    diff            = np.fabs(min_b - max_b)
    # --- Initialize popultion. The population is first uniformly generated between 0 and 1
    #     and then stretched to bounds.
    h_popNormalized = np.random.rand(Np, D)
    h_pop           = min_b + h_popNormalized * diff
    # --- Evaluate the fitness over population members
    h_fobj          = np.asarray([fobj(ind) for ind in h_pop])
    # --- Index of the best population member
    h_best_idx = np.argmin(h_fobj)
    # --- Best population member
    h_best = h_pop[h_best_idx]
    # --- Algorithm loop over the iterations
    for i in range(iter):
        if verbose: print(f'iter: {i:>4d}, best solution: {fobj(h_best):10.6f}')
        # --- Algorithm loop over the population members
        for j in range(Np):
            # --- Generate a list with the indexes of the vectors in the population, excluding the current one
            idxs = [idx for idx in range(Np) if idx != j]
            # --- Randomly choose 3 indexes without replacement
            a, b, c = h_popNormalized[np.random.choice(idxs, 3, replace = False)]
            # --- Create a mutant vector by combining a, b and c. Mutation is achieved by computing the 
            #     difference between population members b and c and adding those differences to population
            #     member a after multiplying them by the mutation factor F. 
            # --- Since the mutant vector does not necessarily belong to [0., 1.] ** D, it is clipped 
            #     to [0., 1.]
            mutant = np.clip(a + F * (b - c), 0, 1)
            # --- Binomial recombination: determine the crossover points for the current population member. 
            cross_points = np.random.rand(D) < CR
            # --- If the condition above is never satisfied, then randomly select the gene to be recombined
            if not np.any(cross_points):
                cross_points[np.random.randint(0, D)] = True
            # --- Where the mutation condition is satisfied, the trial vector equals the mutant vector
            #     otherwise it is equal to the population member
            trial = np.where(cross_points, mutant, h_popNormalized[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < h_fobj[j]:
                h_fobj[j] = f
                h_popNormalized[j] = trial
                if f < h_fobj[h_best_idx]:
                    h_best_idx = j
                    h_best = trial_denorm
        yield h_best, h_fobj[h_best_idx]

#fobj = lambda x: sum(x**2)/len(x)

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

#result = list(de(spherical, bounds=[(-100, 100)] * 2))
result = list(de(rosenbrock, bounds=[(-10, 10)] * 2))
print(result[-1])
#result = list(de(lambda x: x**2 / len(x), bounds=[(-100, 100)] * 32, iter=3000))
#print(result[-1])

#from yabox.problems import problem
#problem(lambda x: sum(x**2)/len(x), bounds=[(-5, 5)] * 2).plot3d()