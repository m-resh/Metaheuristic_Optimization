import numpy as np


def problem_0(x, y):
    w1 = 0.3
    w2 = 0.3
    # to prevent any integer issues
    x = np.array(x)
    y = np.array(y)
    # if only one value is provided (pointwise)
    if x.shape == () and y.shape == ():
        z = x**2 + y**2
        return z
    # else we are dealing with an array of values, as for example plotting or maybe a population
    else:
        len_x = x.shape[0]
        len_y = y.shape[0]
        z = np.zeros([len_x, len_y])
        for _x in range(len_x):
            for _y in range(len_y):
                z[_x, _y] = x[_x]**2 + y[_y]**2
        return z


def problem_1(x, y):
    w1 = 0.3
    w2 = 0.3
    # to prevent any integer issues
    x = np.array(x)
    y = np.array(y)
    # if only one value is provided
    if x.shape == () and y.shape == ():
        z = np.sqrt(x**2 + y**2)
        return z
    # else we are dealing with an array of values, so let's do that
    else:
        len_x = x.shape[0]
        len_y = y.shape[0]
        z = np.zeros([len_x, len_y])
        for _x in range(len_x):
            for _y in range(len_y):
                z[_x, _y] = np.sqrt(x[_x]**2 + y[_y]**2)
        return z


def problem_2(x, y):
    w1 = 0.45
    w2 = 0.5
    x = np.array(x)
    y = np.array(y)
    if x.shape == () and y.shape == ():
        case1 = np.sqrt(x**2 + y**2)-np.pi/(w1*w2)
        case2 = (np.sqrt(x**2 + y**2)-np.pi/(w1*w2))*np.cos(w1*w2*np.sqrt(x**2 + y**2))
        if case1 >= 0:
            z = case1
        else:
            z = case2
        return z
    else:
        len_x = x.shape[0]
        len_y = y.shape[0]
        z = np.zeros([len_x, len_y])
        for _x in range(len_x):
            for _y in range(len_y):
                case1 = np.sqrt(x[_x]**2 + y[_y]**2)-np.pi/(w1*w2)
                case2 =(np.sqrt(x[_x]**2 + y[_y]**2)-np.pi/(w1*w2))*np.cos(w1*w2*np.sqrt(x[_x]**2 + y[_y]**2))
                if case1 >= 0:
                    z[_x, _y] = case1
                else:
                    z[_x, _y] = case2
        return z


def problem_3(x, y):
    w1 = 0.45
    w2 = 0.5
    x = np.array(x)
    y = np.array(y)
    if x.shape == () and y.shape == ():
        z = np.sqrt(x**2 + y**2) - 1/(w1*w2)*np.cos(w1*x)*np.cos(w2*y)
        return z
    else:
        len_x = x.shape[0]
        len_y = y.shape[0]
        z = np.zeros([len_x, len_y])
        for _x in range(len_x):
            for _y in range(len_y):
                z[_x, _y] = np.sqrt(x[_x]**2 + y[_y]**2) - 1/(w1*w2)*np.cos(w1*x[_x])*np.cos(w2*y[_y])
    return z
