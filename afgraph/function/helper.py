import numpy as np


def name_maker(old_name, new_name, prefix):
    if new_name is None and prefix is None:
        new_name = old_name + '_copy'
    elif prefix is not None:
        new_name = prefix + '_' + old_name
    return new_name


def get_vector_list():
    x = np.arange(-10, 10, .001)
    x1 = np.arange(-100, 100, 1)
    x2 = np.arange(-10000, 10000, 100)
    x3 = np.random.uniform(-10000, 10000, size=1000)
    return [x, x1, x2, x3]


def get_shape_list():
    shape_list = []
    for i in range(2, 10):
        shape = (2, ) * i
        size = 2**i
        array = np.random.rand(size).reshape(shape)
        shape_list.append(array)

    return shape_list
