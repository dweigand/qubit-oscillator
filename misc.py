# Defines the memoizition class. This Computes a function only if the result is
# not already stored

import numpy as np
from scipy import interpolate


class Memoize:
    """Save Argument and Result of an expensive function in dictionary"""
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        arglist = list(args)
        for i, arg in enumerate(arglist):
            if isinstance(arg, np.ndarray):
                arg.flags.writeable = False
                arglist[i] = hash(arg.data)
        key = tuple(arglist)
        if key not in self.memo:
            self.memo[key] = self.f(*args)
        return self.memo[key]

    def reset(self):
        self.memo = {}


def int2bin(Min, Max, m):
    return np.arange(Min, Max)[:,np.newaxis] >> np.arange(m)[::-1] & 1



def split_function(x, y):
    """
    Take a perdiodic array x and split both x and y in chunks along the period

    As example let
    x = [1, 2, 3, 1.1, 2.1, 3.1]
    y = [1, 2, 3, 4, 5, 6]
    Then, the function returns
    [[1, 2, 3], [1.1, 2.1, 3.1]], [[1, 2, 3], [4, 5, 6]]

    Args:
        x: periodic array
        y: array of same length
    Returns:
        list of arrays x cut by period
        list of arrays y cut by period of x
    """
    dist = np.diff(x)
    cuts = np.where(dist < 0)[0] + 1
    return np.split(x, cuts), np.split(y, cuts)


def wrap_function(x, y, x_inter):
    """
    Take a perdiodic array x, wrap y along the period, interpolate on x_inter

    As example let
    x = [1, 2, 3, 1, 2, 3]
    x_inter = [1, 1.5, 2, 2.5, 3]
    y = [1, 2, 1, 3, 4, 3]
    Then, the function returns
    [4, 5, 6, 5, 4]

    The function is set to 0 if x_inter is out of bounds. Thus, y_inter will
    be underestimated around the limits of x_inter
    Args:
        x: periodic array
        y: array of same length
        x_inter: linspace on one period of x
    Returns:
        array of length x_inter
    """
    x_list, y_list = split_function(x, y)
    counter = np.arange(len(x_list))
    f_list = [interpolate.interp1d(x_list[i], y_list[i],
                                   bounds_error=False,
                                   fill_value=0,
                                   assume_sorted=True) for i in counter]
    y_inter = np.asarray([f_list[i](x_inter) for i in counter])
    y_inter = np.sum(y_inter, axis=0)
    return y_inter
