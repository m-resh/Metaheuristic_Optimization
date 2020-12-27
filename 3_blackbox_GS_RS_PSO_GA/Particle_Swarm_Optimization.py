import numpy as np
from Blackbox import problem_1, problem_2, problem_3


def PSO(func, no_inputs, c1=0.1, c2=0.9, pop_size=300, num_iter=1000, max_count=100, runs=10, seed=2):

    np.random.seed(seed)  # comment if you want

    def cost(a):
        if(a.shape[1] == 2):
            return func(a[:, 0], a[:, 1])
        elif(a.shape[1] == 4):
            return func(a[:, 0], a[:, 1], a[:, 2], a[:, 3])

    y_hat = np.random.uniform(-10, 10, no_inputs).reshape((1, no_inputs))
    y_hat_cost = cost(y_hat)
    for run in range(runs):
        counter = 0
        x = np.random.uniform(-10, 10, no_inputs*pop_size).reshape((pop_size, no_inputs))
        x_cost = cost(x)
        y = np.copy(x)
        y_cost = np.copy(x_cost)
        v = np.zeros_like(x)
        for i in range(num_iter):
            counter += 1
            x_cost = cost(x)
            y[x_cost < y_cost] = np.copy(x[x_cost < y_cost])
            y_cost[x_cost < y_cost] = x_cost[x_cost < y_cost]

            if(np.min(y_cost) < y_hat_cost):
                y_hat = y[np.argmin(y_cost)]
                y_hat_cost = np.min(y_cost)
                counter = 0

            if(counter >= max_count):
                break

            r1 = np.random.uniform(0, 1, no_inputs)
            r2 = np.random.uniform(0, 1, no_inputs)
            v += c1*np.multiply(r1, (y-x)) + c2*np.multiply(r2, (y_hat-x))
            x += v

    print('min:', y_hat_cost, ', for inputs =', y_hat)


if __name__ == "__main__":

    print('problem_1:')
    PSO(problem_1, 2)

    print('problem_2:')
    PSO(problem_2, 2)

    print('problem_3:')
    PSO(problem_3, 4)
