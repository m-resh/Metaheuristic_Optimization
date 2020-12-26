import numpy as np
seed = 2


def problem1(arg1, arg2, braodcast_warning=True):
    np.random.seed(seed)
    a = .08
    b = .09
    arg1 = np.array(arg1)+np.pi
    arg2 = np.array(arg2)+np.pi
    if arg1.shape != arg2.shape and braodcast_warning:
        print('Provided arrays or values with different shapes, No guarantee about broadcasting behavior!')
        print('To surpress this warning, provide the braodcast_warning=False argument')
    arg1 = np.square(arg1)
    arg2 = np.square(arg2)
    c = np.sqrt(np.add(arg1, arg2))
    d = np.cos(a*arg1)
    e = np.cos(b*arg2)
    f = -(a+b)/(a*b)*np.multiply(d, e)
    return np.add(np.pi*c, f)


def problem2(arg1, arg2, braodcast_warning=True):
    np.random.seed(seed)
    arg1 = 5
    arg2 = 0
    arg1 = np.array(arg1)
    arg2 = np.array(arg2)
    if arg1.shape != arg2.shape and braodcast_warning:
        print('Provided arrays or values with different shapes, No guarantee about broadcasting behavior!')
        print('To surpress this warning, provide the braodcast_warning=False argument')
    c1 = 3.0
    c2 = 0.2
    c3 = np.sqrt(np.add(np.square(arg1), np.square(arg2)))
    c4 = np.random.normal(loc=0, scale=.5, size=arg1.shape)
    c4 = np.where(np.greater(c4, c2*c1), c2*c1, c4)
    c4 = np.where(np.less(c4, -c2*c1), -c2*c1, c4)
    c5 = np.random.normal(loc=0, scale=.5, size=arg2.shape)
    c5 = np.where(np.greater(c5, c2*c1), c2*c1, c5)
    c5 = np.where(np.less(c5, -c2*c1), -c2*c1, c5)
    arg1 = np.add(c4, arg1)
    arg2 = np.add(c5, arg2)
    c6 = np.where(np.greater(c1, c3), -np.square(c1-np.sqrt(np.add(np.square(arg1), np.square(arg2)))), np.add(c4, c5))
    return c6/6.0


def problem3(arg1, arg2, x3, x4, braodcast_warning=True):
    return np.add(problem1(arg1, arg2, braodcast_warning)/2.0, problem2(x3, x4, braodcast_warning)/2.0)
# Created by pyminifier (https://github.com/liftoff/pyminifier)