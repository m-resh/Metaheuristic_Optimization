import numpy as np
from Blackbox import problem_1, problem_2, problem_3


def GA(func, no_inputs, pop_size=300, selection_size=100, Pr=0.5, num_iter=1000, max_count=100, runs=10, seed=2):

    np.random.seed(seed)  # comment if you want

    def cost(a):
        if(a.shape[1] == 2):
            return func(a[:, 0], a[:, 1])
        elif(a.shape[1] == 4):
            return func(a[:, 0], a[:, 1], a[:, 2], a[:, 3])

    best_cost = 1e+7
    for run in range(runs):
        counter = 0
        x = np.random.uniform(-10, 10, no_inputs*pop_size).reshape((pop_size, no_inputs))
        x_cost = cost(x)
        sorted_indices = np.argsort(x_cost)
        x = x[sorted_indices]
        x_cost = x_cost[sorted_indices]

        for i in range(num_iter):
            fitness = (-x_cost + np.max(x_cost) + 1)  # it is always greater than zero
            fitness = fitness / np.sum(fitness)  # so that every value is between 0 and 1 and the sum of all is 1
            ### selection
            selected_indices = np.random.choice(pop_size, size=selection_size, replace=True, p=fitness)
            ### mutation
            parents = np.copy(x[selected_indices])
            randoms = np.random.choice(np.arange(-10, 10), size=parents.shape, replace=True, p=None)
            ## we want mutation to happen with a probability, not always. so, we will choose between the old value and
            ## the new one according to that probability.
            mask = np.random.choice([False, True], size=parents.shape, replace=True, p=[1-Pr, Pr])
            parents[mask] = randoms[mask]
            ### cross over
            np.random.shuffle(parents)
            children = np.zeros_like(parents)
            children[:int(parents.shape[0]/2)] = np.append(parents[:int(parents.shape[0]/2), :int(parents.shape[1]/2)],
                                                      parents[int(parents.shape[0]/2):, int(parents.shape[1]/2):], axis=1)
            children[int(parents.shape[0]/2):] = np.append(parents[int(parents.shape[0]/2):, :int(parents.shape[1]/2)],
                                                      parents[:int(parents.shape[0]/2), int(parents.shape[1]/2):], axis=1)
            x[-children.shape[0]:] = children
            x_cost = cost(x)
            sorted_indices = np.argsort(x_cost)
            x = x[sorted_indices]
            x_cost = x_cost[sorted_indices]

            counter += 1
            if(x_cost[0] < best_cost):
                best = x[0]
                best_cost = x_cost[0]
                counter = 0

            if(counter >= max_count):
                break

    print('min:', best_cost, ', for inputs =', best)


if __name__ == "__main__":

    print('problem_1:')
    GA(problem_1, 2)

    print('problem_2:')
    GA(problem_2, 2)

    print('problem_3:')
    GA(problem_3, 4)
