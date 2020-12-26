import numpy as np
from Blackbox import problem_1, problem_2, problem_3
from time import time
from Grid_Search import GS
from Random_Search import RS
from Particle_Swarm_Optimization import PSO
from Genetic_Algorithm import GA


# all three problems have 2 input dimensions
# the problems can be queried in tensors, matrices, vectors and single queries.

# single values
x_one = np.array(1.0)
y_one = np.array(3.0)
z_one = problem_1(x_one, y_one)
print('Queried with two single values, returns a value of ', z_one, ' ,shape: ', z_one.shape)

# arrays, for populations maybe
x_arr = np.linspace(1,9, 20)
y_arr = np.linspace(1,9, 20)
z_arr = problem_2(x_arr, y_arr)
print('Queried with two arrays of shape', x_arr.shape, y_arr.shape, 'it returns an array of shape:', z_arr.shape)

# matrices ... like ... in a grid maybe
steps = 20
x = 0.01*np.arange(-steps/2, steps/2)
y = 0.01*np.arange(-steps/2, steps/2)
X, Y = np.meshgrid(x, y)
Z = problem_3(X, Y, X, Y)
print('Queried with 4 matrices of shape', X.shape, Y.shape, 'it returns a matrix of shape:', Z.shape)


### comparing the algorithms

for i in range(5):
    print('seed =', i, ':')

    print('Grid Search:')
    t0 = time()
    print('problem_1:')
    GS(problem_1, seed=i)

    print('problem_2:')
    GS(problem_2, seed=i)

    print('problem_3:')
    GS(problem_3, seed=i)
    t1 = time()
    print('total time:', 1000*(t1-t0), '(ms)\n')

    print('Random Search:')
    t0 = time()
    print('problem_1:')
    RS(problem_1, no_inputs=2, init_points=300, num_iter=1000, max_count=100, seed=i)

    print('problem_2:')
    RS(problem_2, no_inputs=2, init_points=300, num_iter=1000, max_count=100, seed=i)

    print('problem_3:')
    RS(problem_3, no_inputs=4, init_points=300, num_iter=1000, max_count=100, seed=i)
    t1 = time()
    print('total time:', 1000*(t1-t0), '(ms)\n')

    print('PSO:')
    t0 = time()
    print('problem_1:')
    PSO(problem_1, no_inputs=2, c1=0.1, c2=0.9, pop_size=300, num_iter=1000, max_count=100, runs=10, seed=i)

    print('problem_2:')
    PSO(problem_2, no_inputs=2, c1=0.1, c2=0.9, pop_size=300, num_iter=1000, max_count=100, runs=10, seed=i)

    print('problem_3:')
    PSO(problem_3, no_inputs=4, c1=0.1, c2=0.9, pop_size=300, num_iter=1000, max_count=100, runs=10, seed=i)
    t1 = time()
    print('total time:', 1000*(t1-t0), '(ms)\n')

    print('GA:')
    t0 = time()
    print('problem_1:')
    GA(problem_1, no_inputs=2, pop_size=300, selection_size=100, Pr=0.5, num_iter=1000, max_count=100, runs=10, seed=i)

    print('problem_2:')
    GA(problem_2, no_inputs=2, pop_size=300, selection_size=100, Pr=0.5, num_iter=1000, max_count=100, runs=10, seed=i)

    print('problem_3:')
    GA(problem_3, no_inputs=4, pop_size=300, selection_size=100, Pr=0.5, num_iter=1000, max_count=100, runs=10, seed=i)
    t1 = time()
    print('total time:', 1000*(t1-t0), '(ms)\n')
