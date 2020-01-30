from function_tree import *
from function_node import *

def ReLU():
    test_graph = FunctionTree('ReLU')
    max_node = Max('max')
    const_node = Constant('a', 0)
    test_graph.insert_node(max_node, 'Output', 'x')
    test_graph.insert_node(const_node, 'max')
    return test_graph


def c_ReLU():

    def f(x):
        return np.max([np.zeros_like(x), x], axis=0)
    return f


def Sigmoid():
    test_graph = FunctionTree('Sigmoid')
    recp_node = Reciprocal('recp')
    const_node_p1 = Constant('1', 1)
    const_node_m1 = Constant('-1', -1)
    sum_node = Sum('sum')
    exp_node = Exp('exp')
    prod_node = Product('prod')

    test_graph.insert_node(recp_node, 'Output', 'x')
    test_graph.insert_node(sum_node, 'recp', 'x')
    test_graph.insert_node(exp_node, 'sum', 'x')
    test_graph.insert_node(prod_node, 'exp', 'x')
    test_graph.insert_node(const_node_m1, 'prod')
    test_graph.insert_node(const_node_p1, 'sum')
    return test_graph


def c_Sigmoid():

    def f(x):
        return 1 / (1 + np.exp(-x))
    return f


def linear(m=1, b=2):
    test_graph = FunctionTree('Linear')
    prod_node = Product('prod')
    sum_node = Sum('sum')
    const_node_m = Constant('m', m)
    const_node_b = Constant('b', b)

    test_graph.insert_node(sum_node, "Output", 'x')
    test_graph.insert_node(const_node_b, 'sum')
    test_graph.insert_node(prod_node, 'sum', 'x')
    test_graph.insert_node(const_node_m, 'prod')
    return test_graph


def c_linear(m=1, b=2):

    def f(x, _m=m, _b=b):
        return _b + _m*x
    return f


def Scaled_Sigmoid(a=1, b=1):
    test_graph = linear(a, b)
    temp_test_graph = Sigmoid()
    test_graph.insert_tree(temp_test_graph, 'x')
    return test_graph


def c_Scaled_Sigmoid(a=1, b=1):

    def f(x, _a=a, _b=b, _c_linear=c_linear, _c_Sigmoid=c_Sigmoid):
        lin = _c_linear(m=_a,b=_b)
        sig = _c_Sigmoid()
        return lin(sig(x))
    return f


def trivial_1():
    return FunctionTree('Trivial_1')


def c_trivial_1():

    def f(x):
        return x
    return f


def trivial_2():
    test_graph = FunctionTree('Trivial_2')
    mean_node = Mean('mean')
    test_graph.insert_node(mean_node, 'Output', 'x')
    return test_graph


def c_trivial_2():

    def f(x):
        return np.mean([x], axis=0)
    return f


def test_1():
    """
    f(x) = max(.2, sin(x)^2)
    """
    test_graph = FunctionTree('Test_1')
    max_node = Max('max')
    const_node = Constant('0.2', .2)
    square_node = Square('square')
    sin_node = Sin('sin')
    test_graph.insert_node(max_node, 'Output', 'x')
    test_graph.insert_node(square_node, 'max', 'x')
    test_graph.insert_node(const_node, 'max')
    test_graph.insert_node(sin_node, 'square', 'x')
    return test_graph


def c_test_1():

    def f(x):
        return np.max([.2*np.ones_like(x), np.square(np.sin(x))], axis=0)
    return f


def test_2():
    """
    f(x) = max(.2, sigmoid(sin(x)^2))
    """
    test_graph = test_1()
    temp_test_graph = Sigmoid()
    test_graph.insert_tree(temp_test_graph, 'max', 'square')
    return test_graph


def c_test_2():

    def f(x, _sigmoid=c_Sigmoid):
        sig = _sigmoid()
        return np.max([0.2*np.ones_like(x), sig(np.square(np.sin(x)))], axis=0)
    return f


def test_3():
    """
    f(x) = sqrt(sin(square(cos(x))))
    """
    test_graph = FunctionTree('Test_3')
    cos_node = Cos('cos')
    sin_node = Sin('sin')
    square_node = Square('square')
    sqrt_node = Sqrt('sqrt')
    test_graph.insert_node(cos_node, 'Output','x')
    test_graph.insert_node(square_node, 'Output', 'cos')
    test_graph.insert_node(sqrt_node, 'Output', 'square')
    test_graph.insert_node(sin_node, 'Output', 'sqrt')
    return test_graph


def c_test_3():

    def f(x):
        return np.sin(np.sqrt(np.square(np.cos(x))))
    return f


def Quadratic(a=1, b=2, c=3):
    test_graph = FunctionTree('Quadratic')
    prod_node_1 = Product('prod_1')
    prod_node_2 = Product('prod_2')
    sum_node_1 = Sum('sum')
    const_node_a = Constant('a', a)
    const_node_b = Constant('b', b)
    const_node_c = Constant('c', c)
    square_node = Square('square')

    test_graph.insert_node(sum_node_1, "Output", 'x')
    test_graph.insert_node(const_node_c, 'sum')
    test_graph.insert_node(prod_node_1, 'sum', 'x', append=True)
    test_graph.insert_node(const_node_b, 'prod_1')
    test_graph.insert_node(prod_node_2, 'sum', 'x')
    test_graph.insert_node(const_node_a, 'prod_2')
    test_graph.insert_node(square_node, 'prod_2', 'x')
    return test_graph


def c_Quadratic(a=1, b=2, c=3):

    def f(x, _a=a, _b=b, _c=c):
        return _c + _b*x + _a*np.square(x)
    return f


def Quadratic_plus_Sigmoid(a=1, b=2, c=3):
    test_graph = Quadratic(a=a, b=b, c=c)
    temp_test_graph = Sigmoid()
    test_graph.insert_tree(temp_test_graph, 'sum')
    return test_graph


def c_Quadratic_plus_Sigmoid(a=1, b=2, c=3):

    def f(x, _a=a, _b=b, _c=c, _c_Quadratic=c_Quadratic, _c_Sigmoid=c_Sigmoid):
        q = _c_Quadratic(a=_a, b=_b, c=_c)
        s = _c_Sigmoid()
        return s(x) + q(x)
    return f

def Sigmoid_with_Quadratic_exp(a=1, b=2, c=3):
    test_graph = Sigmoid()
    temp_test_graph = Quadratic(a=a, b=b, c=c)
    test_graph.insert_tree(temp_test_graph, 'x')
    return test_graph


def c_Sigmoid_with_Quadratic_exp(a=1, b=2, c=3):

    def f(x, _a=a, _b=b, _c=c, _c_Quadratic=c_Quadratic, _c_Sigmoid=c_Sigmoid):
        sig = _c_Sigmoid()
        quad = _c_Quadratic(a=_a, b=_b, c=_c)
        return sig(quad(x))
    return f


def Quadratic_of_Sigmoid(a=1, b=2, c=3):
    test_graph = Quadratic(a=a, b=b, c=c)
    temp_test_graph = Sigmoid()
    test_graph.insert_tree(temp_test_graph, 'x')
    return test_graph


def c_Quadratic_of_Sigmoid(a=1, b=2, c=3):

    def f(x, _a=a, _b=b, _c=c, _c_Quadratic=c_Quadratic, _c_Sigmoid=c_Sigmoid):
        quad = _c_Quadratic(a=_a,b=_b,c=_c)
        sig = _c_Sigmoid()
        return quad(sig(x))
    return f


def Recp_Of_Quadratic_of_Scaled_Neg_Exp(a=1, b=2, c=3):
    test_graph = Sigmoid()
    temp_test_graph = Quadratic(a=a, b=b, c=c)
    test_graph.insert_tree(temp_test_graph, 'recp', 'sum')
    return test_graph


def c_Recp_Of_Quadratic_of_Scaled_Neg_Exp(a=1,b=2,c=3):

    def f(x, _a=a, _b=b, _c=c):
        y = 1.0 + np.exp(-x)
        return 1./(_a*y**2 + _b*y + _c)
    return f


