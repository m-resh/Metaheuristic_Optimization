import numpy as np
from Blackbox import problem_1, problem_2, problem_3


def RS(func, no_inputs, init_points=300, num_iter=1000, max_count=100, seed=2):

    np.random.seed(seed)  # comment if you want

    def cost(a):
        if(a.shape[1] == 2):
            return func(a[:, 0], a[:, 1])
        elif(a.shape[1] == 4):
            return func(a[:, 0], a[:, 1], a[:, 2], a[:, 3])

    x_best = np.random.uniform(-10, 10, no_inputs).reshape((1, no_inputs))
    best_cost = cost(x_best)
    counter = 0
    x = np.random.uniform(-10, 10, no_inputs*init_points).reshape((init_points, no_inputs))
    x_cost = cost(x)
    for i in range(num_iter):
        counter += 1
        x_new = np.copy(x)
        x_step = np.random.uniform(-1, 1, no_inputs*init_points).reshape((init_points, no_inputs))
        x_new += x_step
        new_cost = cost(x_new)
        x[new_cost < x_cost] = np.copy(x_new[new_cost < x_cost])
        x_cost[new_cost < x_cost] = np.copy(new_cost[new_cost < x_cost])

        if(np.min(x_cost) < best_cost):
            x_best = x[np.argmin(x_cost)]
            best_cost = np.min(x_cost)
            counter = 0

        if(counter >= max_count):
            break

    print('min:', best_cost, ', for inputs =', x_best)


if __name__ == "__main__":

    print('problem_1:')
    RS(problem_1, 2)

    print('problem_2:')
    RS(problem_2, 2)

    print('problem_3:')
    RS(problem_3, 4)
