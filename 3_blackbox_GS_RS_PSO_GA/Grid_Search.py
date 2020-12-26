import numpy as np
from Blackbox import problem_1, problem_2, problem_3


def GS(func, seed=2):

    np.random.seed(seed)  # comment if you want

    if(func == problem_1):
        steps = 2000
        x = 0.01*np.arange(-steps/2, steps/2)
        y = 0.01*np.arange(-steps/2, steps/2)
        X, Y = np.meshgrid(x, y)
        Z = problem_1(X, Y)
        ind = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
        print('problem_1 min:', Z[ind], 'for inputs =', x[ind[0]], ',', y[ind[1]])

    elif(func == problem_2):
        steps = 2000
        x = 0.01*np.arange(-steps/2, steps/2)
        y = 0.01*np.arange(-steps/2, steps/2)
        X, Y = np.meshgrid(x, y)
        Z = problem_2(X, Y)
        ind = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
        print('problem_2 min:', Z[ind], 'for inputs =', x[ind[0]], ',', y[ind[1]])

    elif(func == problem_3):
        steps = 100
        x = 0.1*np.arange(-steps/2, steps/2)
        y = 0.1*np.arange(-steps/2, steps/2)
        z = 0.1*np.arange(-steps/2, steps/2)
        w = 0.1*np.arange(-steps/2, steps/2)
        X, Y, Z, W = np.meshgrid(x, y, z, w)
        Z = problem_3(X, Y, Z, W)
        ind = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
        print('problem_3 min:', Z[ind], 'for inputs =', x[ind[0]], ',', y[ind[1]], ',', z[ind[2]], ',', w[ind[3]])


if __name__ == "__main__":

    GS(problem_1)
    GS(problem_2)
    GS(problem_3)
