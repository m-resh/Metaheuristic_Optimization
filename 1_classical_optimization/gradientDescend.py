import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from problems import *


def gradient(func, x, y):
    h = 1e-4
    df_dx = np.subtract(func(x+h, y), func(x-h, y))/(2*h)
    df_dy = np.subtract(func(x, y+h), func(x, y-h))/(2*h)
    return np.array([df_dx, df_dy])


def gradientDescend(func, num_iter=1000, alpha=0.1, plot_curve=True):
    x, y = 15, 15
    cost = func(15, 15)
    xs, ys, costs = x, y, cost
    for iter in range(num_iter):
        g = gradient(func, x, y)
        step = alpha * g
        x -= step[0]
        y -= step[1]
        xs = np.append(xs, x)
        ys = np.append(ys, y)
        cost = func(x, y)
        costs = np.append(costs, cost)
    print('cost: ', func(x, y), 'for x & y = ', x, ',', y)
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
    return x, y


if __name__ == "__main__":

    print('problem_0:')
    gradientDescend(problem_0)
    print('problem_1:')
    gradientDescend(problem_1)
    print('problem_2:')
    gradientDescend(problem_2)
    print('problem_3:')
    gradientDescend(problem_3)
