import numpy as np
from copy import deepcopy
from function_helper import name_maker
import tensorflow as tf


class FunctionNode(object):

    def __init__(self, name):
        self.name = name
        self.cost = 0
        self.depth = 0
        self.fertile = True  # Can have a child {False, True}
        self.fan_in = 1  # Number of children a node can have.
        self.parental = True  # Can have a parent {False, True}
        self.domain = (-1*float('inf'), float('inf'))  # What values can be taken as input without error
        self.range = (-1*float('inf'), float('inf'))  # What values can be returned
        self.latex = name

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return '"{name}":{type}'.format(name=self.name, type=self.__class__.__name__)

    def __repr__(self):
        return self.__class__.__name__ + '("{name}")'.format(name=self.name)

    def graph_name(self):
        return self.latex + "\n" + '"' + self.name + '"'

    def copy_node(self, new_name=None, prefix=None):
        """
        returns a new instance of FunctionNode with new name
        """
        new_name = name_maker(self.name, new_name=new_name, prefix=prefix)
        copy_node = deepcopy(self)
        copy_node.name = new_name

        return copy_node

    @staticmethod
    def get_format(objct, f_format):
        LATEX_STR = 'get_latex_str_format'
        NUMPY_STR = 'get_numpy_str_format'
        TENSORFLOW = 'get_tensorflow_format'
        TENSORFLOW_STR = 'get_tensorflow_str_format'
        NUMPY = 'get_numpy_format'
        DEPENDENT = 'get_dependent_format'
        TENSORFLOW = 'get_tensorflow_format'
        DEPTH = 'get_depth_format'
        NODE_DEPTH = 'get_depth_node_format'
        COST = 'get_cost_format'
        CHILDREN = 'get_extended_children'
        PARENT = 'get_extended_parents'


        if f_format == 'latex_str':
            f = getattr(objct, LATEX_STR)
        elif f_format == 'tf':
            f = getattr(objct, TENSORFLOW)
        elif f_format == 'children_list':
            f = getattr(objct, CHILDREN)
        elif f_format == 'parent_list':
            f = getattr(objct, PARENT)
        elif f_format == 'node_depth':
            f = getattr(objct, NODE_DEPTH)
        elif f_format == 'numpy_str':
            f = getattr(objct, NUMPY_STR)
        elif f_format == 'dependent':
            f = getattr(objct, DEPENDENT)
        elif f_format == 'numpy':
            f = getattr(objct, NUMPY)
        elif f_format == 'tensorflow_str':
            f = getattr(objct, TENSORFLOW_STR)
        elif f_format == 'tensorflow':
            f = getattr(objct, TENSORFLOW)
        elif f_format == 'depth':
            f = getattr(objct, DEPTH)
        elif f_format == 'cost':
            f = getattr(objct, COST)
        else:
            raise NotImplementedError('{frmt} Format has not been implemented'.format(frmt=str(f_format)))
        return f

    def get_function(self, f_format, node_map, tree_map, pass_info=None):
        """
        Recusively returns function in f_format format.
        """
        f = FunctionNode.get_format(self, f_format)
        if self.name not in tree_map:
            return None
        elif not tree_map[self.name]:
            return f(pass_info=pass_info)
        else:
            children_list = []
            for child in tree_map[self.name]:
                children_list.append(node_map[child].get_function(f_format, node_map, tree_map, pass_info=pass_info))
                children_list = list(filter(None.__ne__, children_list))
            return f(children_list, pass_info=pass_info)

    def get_dependent_format(self, children_list=None, pass_info=None):
        return {'f': {self.name: children_list}}

    def get_extended_parents(self, children_list=None, pass_info=None):
        """

        :param children_list: list of list of nodes
        :param pass_info: none of which parents are being found
        :return: list of names of nodes that are influenced by node
        """
        if self.name == pass_info:
            return [self.name]
        else:
            return_list = []
            if children_list is not None:
                for c in children_list:
                    if pass_info in c:
                        return_list.extend(c)
            if len(return_list) > 0:
                return_list.append(self.name)
            return return_list

    def get_extended_children(self, children_list=None, pass_info=None):
        """
        :param children_list: list of lists of nodes
        :param pass_info: node of which children are being found
        :return: list of the names of nodes that influence node
        """
        if self.name == pass_info:  # Keep all nodes from children_list
            extended_child_list = []
            if children_list is not None:
                for c in children_list:
                    extended_child_list.extend(c)
            extended_child_list.append(self.name)
            return extended_child_list
        else:
            """
            Two Cases
                1) pass_info is in one of elements of children_list 
                    We return the union of any elements of children_list that have pass_info in them
                2) pass_info is not in one of the elements of children_list
                    Return union of all elements of children_list and self.name
            """
            pass_info_return_list = []
            return_list = []
            if children_list is not None:
                for c in children_list:
                    if pass_info in c:
                        pass_info_return_list.extend(c)
                    else:
                        return_list.extend(c)

            if len(pass_info_return_list) > 0:
                return pass_info_return_list
            else:
                return_list.append(self.name)
                return return_list

    def get_depth_node_format(self, children_list=None, pass_info=None):
        # Child_list is a list of dictionaries!
        return_dict = {}

        if self.name == pass_info:
            return_dict['node_depth'] = 0
        
        if children_list is not None:
            temp_depth_list = []
            for dct in children_list:
                if 'node_depth' in dct:
                    temp_depth_list.append(dct['node_depth'])
            if len(temp_depth_list) > 0:
                return_dict['node_depth'] = 1 + max(temp_depth_list)

        return return_dict

    def get_depth_format(self, children_list=None, pass_info=None):
        if children_list is None:
            return 0
        else:
            return 1 + max(children_list)

    def get_cost_format(self, children_list=None, pass_info=None):
        if children_list is None:
            return self.cost
        else:
            return sum(children_list) + self.cost*len(children_list)

    def get_latex_str_format(self, children_list=None, pass_info=None):
        raise NotImplementedError(
            'Latex String Format for {node} has not been implemented'.format(
                node=type(self).__name__))

    def get_numpy_format(self, children_list=None, pass_info=None):
        raise NotImplementedError(
            'Numpy Format for {node} has not been implemented'.format(
                node=type(self).__name__))

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        raise NotImplementedError(
            'Numpy String Format for {node} has not been implemented'.format(
                node=type(self).__name__))
    
    def _numpy_function(self, children_list=None, pass_info=None):
        raise NotImplementedError(
            'Numpy Function for {node} has not been implemented'.format(
                node=type(self).__name__))

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        raise NotImplementedError(
            'tensorflow Function for {node} has not been implemented'.format(
                node=type(self).__name__))


class Product(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = float('inf')
        self.latex = "*"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.product(['
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += '], axis = 0)'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt = '('
        test_1 = False
        for c_txt in sorted(children_list):
            try:
                test_1 = int(c_txt)
            except ValueError:
                pass
            if test_1 != 1:
                txt += c_txt
                txt += '*'
            else:
                test_1 = False
        txt = txt[:-1]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.product([sub_func(x=x) for sub_func in c_l], axis=0)
        return f

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _name=self.name):
            c_l_stack = tf.stack([sub_func(x=x) for sub_func in c_l], name=_name + '_product_stack')
            return tf.math.reduce_prod(c_l_stack, axis=0, name=_name)

        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _product(x):
            return np.product(a=x, axis=0)
        return _product


class Output(FunctionNode):
    
    def __init__(self, name):
        super().__init__(name)
        self.cost = 0
        self.parental = False
        self.latex = 'f(x)'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'def f(x):\n\treturn ' + children_list[0]

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return f'f(x) = ' + children_list[0]

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return [sub_func(x=x) for sub_func in c_l]
        return f

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        return children_list[0]

    def get_numpy_function(self, pass_info=None):

        def _output(x):
            return x
        return _output


class Operator(FunctionNode):

    def __init__(self, name, operator):
        super().__init__(name)
        self.cost = 1
        self.operator = operator
        self.latex = name

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return str(self.name) + '(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = self.name + '('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, operator=self.operator, c_l=children_list):
            return [operator(sub_func(x=x)) for sub_func in c_l][0]
        return f

    def get_numpy_function(self, pass_info=None):
        return self.operator


class Max(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = float('inf')
        self.latex = "max"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.max(['
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += '], axis = 0)'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = 'max('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.max([sub_func(x=x) for sub_func in c_l], axis=0)
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _max(x):
            return np.max(x, axis=0)
        return _max

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _name=self.name):
            c_l_stack = tf.stack([sub_func(x=x) for sub_func in c_l], name=_name + '_max_stack')
            return tf.math.reduce_max(c_l_stack, axis=0, name=_name)

        return f


class Min(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = float('inf')
        self.latex = "min"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.min(['
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += '], axis = 0)'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = 'min('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.min([sub_func(x=x) for sub_func in c_l], axis=0)
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _min(x):
            return np.min(x, axis=0)
        return _min

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _name=self.name):
            c_l_stack = tf.stack([sub_func(x=x) for sub_func in c_l], name=_name + '_min_stack')
            return tf.math.reduce_min(c_l_stack, axis=0, name=_name)

        return f


class Mean(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = float('inf')
        self.latex = "mean"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.mean(['
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += '], axis = 0)'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = 'mean('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.mean([sub_func(x=x) for sub_func in c_l], axis=0)
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _mean(x):
            return np.mean(x, axis=0)
        return _mean

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _name=self.name):
            c_l_stack = tf.stack([sub_func(x=x) for sub_func in c_l], name=_name + '_mean_stack')
            return tf.math.reduce_mean(c_l_stack, axis=0, name=_name)

        return f


class Median(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = float('inf')
        self.latex = "median"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.median(['
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += '], axis = 0)'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = 'median('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.median([sub_func(x=x) for sub_func in c_l], axis=0)
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _median(x):
            return np.median(x=x, axis=0)
        return _median

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _name=self.name):
            c_l_stack = tf.stack([sub_func(x=x) for sub_func in c_l], name=_name + '_median_stack')
            return tf.contrib.distributions.percentile(c_l_stack, q=50, name=_name)

        return f


class Reciprocal(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = 1
        self.latex = '1/.'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt = 'np.reciprocal(' + children_list[0] + ')'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt = '1/('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ', '
        txt = txt[:-2]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list):
            return np.reciprocal(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):
        return np.reciprocal

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.reciprocal(c_l[0](x=x), name=_name)

        return f


class Exp(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = 'exp'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.exp('
        for c_txt in sorted(children_list):
            txt += c_txt
        txt += ')'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = 'exp('
        for c_txt in sorted(children_list):
            txt += c_txt
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.exp(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        return np.exp

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.exp(c_l[0](x=x), name=_name)

        return f


class ArcTan(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = 'arctan'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.arctan('
        for c_txt in sorted(children_list):
            txt += c_txt
        txt += ')'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = 'arctan('
        for c_txt in sorted(children_list):
            txt += c_txt
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.arctan(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        return np.arctan

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.atan(c_l[0](x=x), name=_name)

        return f


class Sum(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.fan_in = float('inf')
        self.latex = '+'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        txt: str = 'np.sum(['
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ' ,'
        txt = txt[:-2]  # Removes last comma!
        txt += '], axis = 0)'
        return txt

    def get_latex_str_format(self, children_list=None, pass_info=None):
        txt: str = '('
        for c_txt in sorted(children_list):
            txt += c_txt
            txt += ' + '
        txt = txt[:-3]  # Removes last comma!
        txt += ')'
        return txt

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.sum([sub_func(x=x) for sub_func in c_l], axis=0)
        return f

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _name=self.name):
            c_l_stack = tf.stack([sub_func(x=x) for sub_func in c_l], name=_name + '_sum_stack')
            return tf.math.reduce_sum(c_l_stack, axis=0, name=_name)

        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _sum(x):
            return np.sum(x, axis=0)
        return _sum


class Constant(FunctionNode):
    
    def __init__(self, name, value):
        super().__init__(name)
        self.fertile = False
        self.value = value
        self.cost = 1
        self.latex = str(value)
 
    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return str(self.value)+'*np.ones_like(x)'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return str(self.value)

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, value=self.value):
            return np.ones_like(x)*value
        return f

    def get_tensorflow_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list, _value=self.value, __input=pass_info['input'],
              _name=self.name, _dtype=pass_info['dtype']):
            _ones = tf.ones(shape=tf.shape(__input), dtype=_dtype, name='ones_' + _name)
            return tf.math.scalar_mul(_value, _ones, name='const_'+_name)

        return f

    def get_dependent_format(self, children_list=None, pass_info=None):
        return self.name

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _constant(x):
            return self.value*np.ones_like(x)
        return _constant


class Square(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = '^2'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.square(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return '((' + children_list[0] + ')^2)'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.square(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _square(x):
            return np.square(x)
        return _square

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.square(c_l[0](x=x), name=_name)

        return f


class Sqrt(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = '^(1/2)'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.sqrt(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return '((' + children_list[0] + ')^0.5)'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.sqrt(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _sqrt(x):
            return np.sqrt(x)
        return _sqrt

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.sqrt(c_l[0](x=x), name=_name)

        return f


class Abs(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = 'abs'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.abs(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return '|' + children_list[0] + '|'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.abs(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):
        return np.abs

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.abs(c_l[0](x=x), name=_name)

        return f


class Sign(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = 'sign'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.sign(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return 'sign(' + children_list[0] + ')'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.sign(c_l[0](x=x))
        return f

    def get_numpy_function(self, children_list=None, pass_info=None):
        return np.sign

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.sign(c_l[0](x=x), name=_name)

        return f


class Sin(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = 'sin'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.sin(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return 'sin(' + children_list[0] + ')'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.sin(c_l[0](x=x))
        return f

    def get_numpy_function(self, pass_info=None):

        def _sin(x):
            return np.sin(x)
        return _sin

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.sin(c_l[0](x=x), name=_name)

        return f


class Cos(FunctionNode):
    def __init__(self, name):
        super().__init__(name)
        self.cost = 1
        self.latex = 'cos'

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.cos(' + children_list[0] + ')'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return 'cos(' + children_list[0] + ')'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x, c_l=children_list):
            return np.cos(c_l[0](x=x))
        return f

    def get_numpy_function(self, pass_info=None):
        return np.cos

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, _name=self.name):
            return tf.math.cos(c_l[0](x=x), name=_name)

        return f


class Uniform(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 0
        self.fertile = False
        self.latex = "U([0,1])"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.random.uniform(size=x.shape)'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return '~U([0,1])'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x):
            return np.random.uniform(size=np.size(x))
        return f

    def get_dependent_format(self, children_list=None, pass_info=None):
        return self.name

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _uniform(x):
            return np.random.uniform(size=np.shape(x))

        return _uniform

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, __input=pass_info['input'], _name=self.name, _dtype=pass_info['dtype']):
            return tf.random.uniform(tf.shape(__input), dtype=_dtype, name=_name)

        return f


class Normal(FunctionNode):

    def __init__(self, name):
        super().__init__(name)
        self.cost = 0
        self.fertile = False
        self.latex = "N(0,1)"

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'np.random.Normal(size=x.shape)'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return "~N(0,1)"

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x):
            return np.random.normal(size=np.size(x))
        return f

    def get_dependent_format(self, children_list=None, pass_info=None):
        return self.name

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _normal(x):
            return np.random.normal(size=np.shape(x))
        return _normal

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, c_l=children_list, __input=pass_info['input'], _name=self.name, _dtype=pass_info['dtype']):
            return tf.random.normal(tf.shape(__input), dtype=_dtype, name=_name)

        return f


class Input(FunctionNode):
    
    def __init__(self, name):
        super().__init__(name)
        self.cost = 0
        self.fertile = False
        self.latex = name

    def get_numpy_str_format(self, children_list=None, pass_info=None):
        return 'x'

    def get_latex_str_format(self, children_list=None, pass_info=None):
        return 'x'

    def get_numpy_format(self, children_list=None, pass_info=None):

        def f(x):
            return x
        return f

    def get_tensorflow_format(self, children_list=None, pass_info=None):
        def f(x, __input=pass_info['input']):
            return __input

        return f

    def get_dependent_format(self, children_list=None, pass_info=None):
        return self.name

    def get_numpy_function(self, children_list=None, pass_info=None):

        def _inpt(x):
            return x
        return _inpt
