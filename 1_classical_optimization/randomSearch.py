import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from problems import *


def randomSearch(func, num_iter=1000, plot_curve=True):
    best_x, best_y = 15, 15
    best_cost = func(15, 15)
    xs, ys, costs = best_x, best_y, best_cost
    for iter in range(num_iter):
        x_step, y_step = np.random.uniform(-1, 1, 2)
        x = best_x + x_step
        y = best_y + y_step
        new_cost = func(x, y)
        if(new_cost < best_cost):
            best_x, best_y = x, y
            best_cost = new_cost
            xs = np.append(xs, best_x)
            ys = np.append(ys, best_y)
            costs = np.append(costs, best_cost)
    print('best cost: ', best_cost, 'for x & y = ', best_x, ',', best_y)
    if (plot_curve):
        fig = plt.figure(figsize=(15, 20))
        ax = fig.add_subplot(111, projection='3d')

        steps = 350
        x = 0.1*np.arange(-steps/2, steps/2)
        y = 0.1*np.arange(-steps/2, steps/2)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, func(x, y),
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False)
        ax.plot(xs, ys, costs, 'g')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Function Value (Cost)')
        plt.show()
    return best_x, best_y


if __name__ == "__main__":

    print('problem_0:')
    randomSearch(problem_0)
    print('problem_1:')
    randomSearch(problem_1)
    print('problem_2:')
    randomSearch(problem_2)
    print('problem_3:')
    randomSearch(problem_3)
