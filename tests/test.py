import pytest
from function_tree import FunctionTree
from function_node import *
from function_exceptions import *
from standard_graphs import *
from function_helper import get_vector_list, get_shape_list
from inspect import getmembers, isfunction

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_functions():
    import standard_graphs
    all_functions = {name: value for name, value in getmembers(standard_graphs, predicate=isfunction)}
    numpy_functions = [name for name in all_functions.keys() if name.startswith('c_')]
    graph_functions = [name[2:] for name in numpy_functions]
    return all_functions, zip(numpy_functions, graph_functions)

class TestTreeOperations:
    f_dict, functions = get_functions()
    shapes = get_vector_list()
    vectors = get_shape_list()

    @pytest.mark.parametrize("np_function, graph_function", functions)
    def test_vector(self, np_function, graph_function):
        n_f = TestTreeOperations.f_dict[np_function]()
        g_f = TestTreeOperations.f_dict[graph_function]()
        print(g_f)
        print()
        t_f = g_f.get_tensorflow()
        for v in TestTreeOperations.vectors:

            assert np.allclose(n_f(v), g_f(v))
            assert np.allclose(n_f(v), t_f(v))


    def test_fan_in(self):
        temp_graph = FunctionTree('test_fan_in')
        sin = Sin('sin')
        a = Constant('a', 1)
        temp_graph.insert_node(sin, 'Output', 'x')
        with pytest.raises(FanInError):
            temp_graph.insert_node(a, 'sin')

    def test_max_depth(self):
        temp_graph = FunctionTree('test_max_depth', max_depth=3)
        sin1 = Sin('sin1')
        sin2 = Sin('sin2')
        sin3 = Sin('sin3')
        temp_graph.insert_node(sin1, 'Output', 'x')
        temp_graph.insert_node(sin2, 'sin1', 'x')
        with pytest.raises(DepthError):
            temp_graph.insert_node(sin3, 'sin2', 'x')


    def test_copy_1(self):
        temp_graph = Sigmoid()
        temp_graph_copy = temp_graph.copy_tree('Sigmoid2')
        assert temp_graph == temp_graph_copy
        assert temp_graph is not temp_graph_copy

    def test_copy_2(self):
        temp_graph = Quadratic()
        temp_graph_copy = temp_graph.copy_tree('Quadratic2')
        assert temp_graph == temp_graph_copy
        assert temp_graph is not temp_graph_copy

    def test_insert_1(self):
        temp_quadratic = Quadratic()
        temp_sigmoid = Sigmoid()
        quadratic = temp_quadratic.copy_tree(new_name='Quadratic of Sigmoid')
        sigmoid = temp_sigmoid.copy_tree(new_name='Sigmoid Passed to Quadtraic')
        quadratic.insert_tree(temp_sigmoid, parent_node_name='x')
        sigmoid.insert_tree(temp_quadratic, parent_node_name='Output', child_node_name='recp_copy')
        assert quadratic == sigmoid
        assert quadratic is not sigmoid


    def test_loop_1(self):
        temp_graph = FunctionTree('test_loop_1')
        sin = Sin('sin')
        mean = Mean('mean')
        temp_graph.insert_node(sin, 'Output', 'x')
        temp_graph.insert_node(mean, 'sin', 'x')
        with pytest.raises(CloseLoopError):
            temp_graph.connect_nodes('mean', 'sin')

    def test_loop_2(self):
        temp_graph = FunctionTree('test_loop_2')
        sin = Sin('sin')
        mean = Mean('mean')
        cos = Cos('cos')
        square = Square('square')
        temp_graph.insert_node(sin, 'Output', 'x')
        temp_graph.insert_node(mean, 'Output', 'sin')
        temp_graph.insert_node(cos, 'Output', 'mean')
        temp_graph.insert_node(square, 'Output', 'cos')
        with pytest.raises(CloseLoopError):
            temp_graph.connect_nodes('mean', 'square')

    def test_depth_1(self):
        temp_graph = ReLU()
        assert temp_graph.get_depth() == 2
        assert temp_graph.get_node_depth('max') == 1
        assert temp_graph.get_node_depth('x') == 2
        assert temp_graph.get_node_depth('a') == 2

    def test_depth_2(self):
        temp_graph = Sigmoid()
        assert temp_graph.get_depth() == 5
        assert temp_graph.get_node_depth('x') == 5
        assert temp_graph.get_node_depth('-1') == 5
        assert temp_graph.get_node_depth('prod') == 4
        assert temp_graph.get_node_depth('exp') == 3
        assert temp_graph.get_node_depth('1') == 3
        assert temp_graph.get_node_depth('sum') == 2
        assert temp_graph.get_node_depth('recp') == 1

    def test_family_1(self):
        temp_graph = Sigmoid()
        assert temp_graph.get_parents('-1') == {'prod'}
        assert temp_graph.get_extended_parents('-1') == {'prod', 'exp', 'sum', 'recp', 'Output'}
        assert temp_graph.get_extended_children('1') == set([])
        assert temp_graph.get_extended_children('exp') == {'prod', 'x', '-1'}
        assert temp_graph.get_children('sum') == {'1', 'exp'}
        assert temp_graph.get_children('x') == set([])
        assert temp_graph.get_children('-1') == set([])

    def test_depth_3(self):
        temp_graph = Sigmoid_with_Quadratic_exp()
        assert temp_graph.get_depth() == 8
        assert temp_graph.get_node_depth("Quadratic_prod_2") == 6
        assert temp_graph.get_node_depth('1') == 3
        assert temp_graph.get_node_depth("Quadratic_b") == 7
        assert temp_graph.get_node_depth("Quadratic_a") == 7

    def test_family_3(self):
        temp_graph = Sigmoid_with_Quadratic_exp()
        assert temp_graph.get_extended_children("Quadratic_sum") == {"Quadratic_prod_2",
                                                                     "Quadratic_a",
                                                                     "Quadratic_square",
                                                                     "Quadratic_prod_1",
                                                                     "Quadratic_c",
                                                                     "Quadratic_b",
                                                                     "x"}
        assert temp_graph.get_children("Quadratic_sum") == {"Quadratic_prod_1", "Quadratic_prod_2", "Quadratic_c"}
        assert temp_graph.get_parents('x') == {"Quadratic_prod_1", 'Quadratic_square'}
        assert temp_graph.get_extended_parents("Quadratic_c") == {"Quadratic_sum",
                                                                  "prod",
                                                                  "exp",
                                                                  "sum",
                                                                  "recp",
                                                                  "Output"}











