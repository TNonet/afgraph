from .node import *
from .helper import name_maker
from copy import deepcopy, copy
from graphviz import Digraph
from matplotlib import pyplot as plt
from .exceptions import *


class FunctionTree:
    """
    A functional implementation of a graph

    Example FunctionTree # 1

    f(x) = g(h(x,y))
        Tree:
                Output <- Operator(g) <- Operator(h) <- Input(x)
                                                   \\__ Constant(y)
                depth = 2 
                node_map = {'Output': OutputOperator),
                            'g': UniaryOperator,
                            'h': BinaryOperator,
                            'y': ConstantOperator,
                            'x': InputOperator}

                tree_map = {'Output': ['g'],
                            'g': ['h'],
                            'h': ['x', 'y']}
                            'x': [],
                            'y' = []}
    """

    def __init__(self, name, max_depth=32, inpt='x'):
        self.name = name
        self.root = Output('Output')
        self.input = Input(inpt)
        self.max_depth = max_depth
        self.node_map = {
            self.root.name: self.root,
            self.input.name: self.input}  # Maps names to type
        self.tree_map = {self.root.name: [self.input.name],
                         self.Input.name: []}  # Maps names to location

    def __call__(self, x):
        f = self.get_numpy()
        return f(x=x)

    def __repr__(self):
        return "{name} (Nodes: {n}, Depth: {d})".format(name=self.name, n=len(self.node_map), d=self.get_depth())

    def __eq__(self, other):
        if not isinstance(other, FunctionTree):
            return False
        return self.get_numpy_str() == other.get_numpy_str()

    def __get_maps(self, node_map=None, tree_map=None):
        if tree_map is not None:
            temp_tree_map = deepcopy(tree_map)
        else:
            temp_tree_map = deepcopy(self.tree_map)

        if node_map is not None:
            temp_node_map = deepcopy(node_map)
        else:
            temp_node_map = deepcopy(self.node_map)
        return temp_node_map, temp_tree_map

    def copy_tree(self, new_name=None, prefix=None, input_name=None, output_name=None):
        """
        returns a new instance of FunctionTree with new name
        """

        old_tree_name = self.name
        copy_tree = copy(self)

        copy_tree.name = name_maker(old_tree_name, new_name=new_name, prefix=prefix)

        new_node_map = {}
        for node_name, node in self.node_map.items():
            if isinstance(node, Input):
                if input_name is not None:
                    new_node_map[input_name] = node.copy_node(new_name=input_name)
                else:
                    new_node_map[node_name] = node.copy_node(new_name=node_name)
            elif isinstance(node, Output):
                if output_name is not None:
                    new_node_map[output_name] = node.copy_node(new_name=output_name)
                else:
                    new_node_map[node_name] = node.copy_node(new_name=node_name)
            else:
                new_node_name = name_maker(node_name, new_name=None, prefix=prefix)
                new_node_map[new_node_name] = node.copy_node(new_name=None, prefix=prefix)
        new_tree_map = {}
        for p_node, c_node_list in self.tree_map.items():
            if isinstance(self.node_map[p_node], Output):
                if output_name is not None:
                    p_node_new_name = output_name
                else:
                    p_node_new_name = p_node
            elif isinstance(self.node_map[p_node], Input):
                if input_name is not None:
                    p_node_new_name = input_name
                else:
                    p_node_new_name = p_node
            else:
                p_node_new_name = name_maker(p_node, new_name=None, prefix=prefix)

            new_c_node_list = []
            for c_node in c_node_list:
                if isinstance(self.node_map[c_node], Input):
                    if input_name is not None:
                        c_node_new_name = input_name
                    else:
                        c_node_new_name = c_node
                else:
                    c_node_new_name = name_maker(c_node, new_name=None, prefix=prefix)

                new_c_node_list.append(c_node_new_name)
            new_tree_map[p_node_new_name] = new_c_node_list

        copy_tree.node_map = new_node_map
        copy_tree.tree_map = new_tree_map

        return copy_tree

    def get_parents(self, node_name=None, node_map=None, tree_map=None):
        """

        :param node_name:
        :param node_map:
        :param tree_map:
        :return: list of nodes have node_name as a child:
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if node_name not in temp_node_map:
            raise NameError('Node, {name} not in Tree'.format(name=node_name))
        p_list = set([])
        for p_node_name, c_node_list in temp_tree_map.items():
            if node_name in c_node_list:
                p_list.add(p_node_name)

        return p_list

    def get_extended_parents(self, node_name, node_map=None, tree_map=None):
        """

        :param node_name:
        :return: list of nodes that will be influenced by node_name
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if node_name not in temp_node_map:
            raise NameError('Node, {name} not in Tree'.format(name=node_name))
        node_list = temp_node_map[self.root.name].get_function(
            'parent_list',
            temp_node_map,
            temp_tree_map,
            pass_info=node_name)
        node_set = set(node_list)
        node_set.remove(node_name)
        return node_set

    def get_children(self, node_name, node_map=None, tree_map=None):
        """

        :param node_name:
        :return: list of nodes that will be influenced by node_name
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if node_name not in temp_node_map:
            raise NameError('Node, {name} not in Tree'.format(name=node_name))
        if node_name not in temp_tree_map:
            return set([])
        else:
            return set(temp_tree_map[node_name])

    def get_extended_children(self, node_name, node_map=None, tree_map=None):
        """

        :param node_name: Function Node
        :return: list of nodes that have influence on node_name
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if node_name not in temp_node_map:
            raise NameError('Node, {name} not in Tree'.format(name=node_name))
        node_list = temp_node_map[self.root.name].get_function(
            'children_list',
            temp_node_map,
            temp_tree_map,
            pass_info=node_name)
        node_set = set(node_list)

        if node_name not in node_set:
            return set([])
        else:
            node_set.remove(node_name)

        return node_set

    def get_tensorflow_str(self):
        """
        retursn FunctionTree as callable tensorflow function.
        """
        pass

    def get_numpy_str(self):
        """
        returns FunctionTree as callable numpy function
        """
        return self.node_map[self.root.name].get_function('numpy_str', self.node_map, self.tree_map)

    def get_latex_str(self):
        return self.node_map[self.root.name].get_function('latex_str', self.node_map, self.tree_map)

    def get_depth(self, node_map=None, tree_map=None):
        """

        :param node_map:
        :param tree_map:
        :return:
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)
        return temp_node_map[self.root.name].get_function('depth', temp_node_map, temp_tree_map)

    def get_node_depth(self, node_name, node_map=None, tree_map=None):
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if node_name not in temp_node_map:
            raise NameError('Node, {name}, not in Tree'.format(name=node_name))
        else:
            return_dict = temp_node_map[self.root.name].get_function(
                'node_depth',
                temp_node_map,
                temp_tree_map,
                pass_info=node_name)
        if 'node_depth' not in return_dict:
            # If nodes are being added before connected to root!
            # Max Graph Depth is calculated when graphs are connected!
            return -1
        return return_dict['node_depth']

    def get_cost(self):
        return self.node_map[self.root.name].get_function('cost', self.node_map, self.tree_map)

    def get_numpy(self):
        numpy_list = self.node_map[self.root.name].get_function('numpy', self.node_map, self.tree_map)

        def f(x):
            return np.squeeze(numpy_list(x=x))

        return f

    def get_tensorflow(self, tf_type=tf.float32):

        def f(x, __input_node_name=self.Input.name, __type=tf_type):
            if type(x) is not np.ndarray:
                x = np.array(x)
            rank = len(x.shape)
            none_rank = rank * (None,)
            __input = tf.compat.v1.placeholder(tf_type, shape=none_rank, name=__input_node_name)
            graph = self.node_map[self.root.name].get_function('tf',
                                                               self.node_map,
                                                               self.tree_map,
                                                               pass_info={'input': __input,
                                                                          'dtype': __type})
            with tf.compat.v1.Session() as sess:
                return sess.run(graph(x), feed_dict={__input: x})

        return f

    def get_graph(self):
        #  Build Graphs!
        g = Digraph(self.name, filename=str(self.name) + '.gv')
        for p_node, c_node_list in self.tree_map.items():
            p_node_name = self.node_map[p_node].graph_name()
            for c_node in c_node_list:
                c_node_name = self.node_map[c_node].graph_name()
                g.edge(c_node_name, p_node_name)
        g.attr(rankdir='LR')
        g.view()

    def get_plot(self, range_object=None, N=100):
        """

        :param range_object:
        :return:
        """
        n = 1.0/N
        if range_object is None:
            rng = np.arange(-2, 2, n)
        elif isinstance(range_object, (tuple, list)):
            rng = np.arange(range_object[0], range_object[1], n)
        elif isinstance(range_object, np.ndarray):
            rng = range_object
        elif isinstance(range_object, range):
            rng = range_object
        else:
            raise ValueError('Unknown Range Object')

        f = self.get_numpy()
        y = f(rng)

        plt.plot(rng, y)
        plt.show()

    def __add_node(self, node, node_map=None):
        """
        """
        temp_node_map, _ = self.__get_maps(node_map=node_map)

        if isinstance(node, Input) and self.input is not None:
            raise Exception('Cannot add Input node if already exists in graph')
        if isinstance(node, Output) and self.root is not None:
            raise Exception('Cannot add Output node if already exists in graph')
        if node.name in temp_node_map:
            raise NameError('Name of Node already exists in graph')

        temp_node_map[node.name] = node

        return temp_node_map

    def __add_edge(self, parent_node_name, child_node_name, node_map=None, tree_map=None):
        """
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if parent_node_name not in temp_node_map:
            raise NodeNameError(tree_name=self.name, p_node_name=parent_node_name)
        else:
            parent_node = temp_node_map[parent_node_name]
            if not parent_node.fertile:
                raise ChildError(self.name, parent_node_name, child_node_name)

        if child_node_name not in temp_node_map:
            raise NodeNameError(tree_name=self.name, c_node_name=child_node_name)
        else:
            child_node = temp_node_map[child_node_name]
            if not child_node.parental:
                raise ParentError(self.name, parent_node_name, child_node_name)

        if parent_node_name in temp_tree_map:
            temp_children = set(temp_tree_map[parent_node_name])
        else:
            temp_children = set([])

        temp_children.add(child_node_name)

        if len(temp_children) > parent_node.fan_in:
            raise FanInError(self.name, parent_node, temp_tree_map[parent_node_name], child_node_name)

        depth = self.get_node_depth(parent_node_name, node_map=temp_node_map, tree_map=temp_tree_map)
        if depth + 1 > self.max_depth:
            raise DepthError(tree_name=self.name,
                             max_depth=self.max_depth,
                             p_node_name=parent_node_name,
                             p_node_depth=depth,
                             c_node_name=child_node_name)

        if parent_node_name in self.get_extended_children(child_node_name, node_map=temp_node_map, tree_map=temp_tree_map):
            raise CloseLoopError(self.name, parent_node_name, child_node_name)

        if parent_node_name in temp_tree_map:
            temp_tree_map[parent_node_name].append(child_node_name)
        else:
            temp_tree_map[parent_node_name] = [child_node_name]

        if child_node_name not in temp_tree_map:
            temp_tree_map[child_node_name] = []

        return temp_tree_map

    def __delete_node(self, node_name, node_map=None, tree_map=None):
        """
        """
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)

        if node_name not in self.node_map:
            raise NameError('Tree: {t}. Node, {n}, does not exist in tree.')
        if len(self.get_parents(node_name, node_map=temp_node_map, tree_map=temp_tree_map)) > 0:
            raise Exception('Node is the child of another node.')
        if len(self.get_children(node_name, node_map=temp_node_map, tree_map=temp_tree_map)) > 0:
            raise Exception('Node is the parent of other nodes.')

        del temp_node_map[node_name]

        return temp_node_map

    def __delete_edge(self, parent_node_name, child_node_name, node_map=None, tree_map=None):
        temp_node_map, temp_tree_map = self.__get_maps(node_map=node_map, tree_map=tree_map)
        if parent_node_name not in temp_node_map:
            raise NodeNameError(tree_name=self.name, p_node_name=parent_node_name)
        if child_node_name not in temp_node_map:
            raise NodeNameError(tree_name=self.name, c_node_name=child_node_name)
        if parent_node_name not in temp_tree_map:
            raise Exception('Parent node, {p_node}, does not have any children'.format(p_node=parent_node_name))
        if child_node_name not in temp_tree_map[parent_node_name]:
            raise Exception('Edge, {c_node} -> {p_node}, does not exist'.format(
                c_node=child_node_name,
                p_node=parent_node_name))

        temp_tree_map[parent_node_name].remove(child_node_name)

        return temp_tree_map

    def rename_node(self, old_node_name, new_name=None, prefix=None):
        """
        """
        new_node_name = name_maker(old_name=old_node_name, new_name=new_name, prefix=prefix)
        if new_node_name in self.node_map:
            raise Exception('New node name already exists')

        new_node = self.node_map[old_node_name].copy_node(new_name=new_name, prefix=prefix)

        self.replace_node(old_node_name, new_node)

    def replace_node(self, old_node_name, new_node):
        """
        """
        if old_node_name not in self.node_map:
            raise Exception('Old Node is not in tree')
        else:
            old_node = self.node_map[old_node_name]

        if old_node_name not in self.tree_map:
            # Old_node_name is not connected
            raise Exception('Old nod is not connected (BIG ERROR)')
        elif self.tree_map[old_node_name] == []:
            # Old_node_name is a generator!
            if new_node.parental:
                raise Exception('Cannot replace a non-Parental Node with a Parental node')
        elif len(self.tree_map[old_node_name]) == 1:
            # Old_node_name is a channel or branch
            if not new_node.parental:
                raise Exception('Cannot replace a Parental node with a Non-Parental node.')
        else:
            if new_node.fan_in < len(self.tree_map[old_node_name]):
                raise Exception('Cannot replace a node with insufficnet fan_in')
            # Old_node_name is a branch

        parent_nodes = self.get_parents(old_node_name)
        child_nodes = self.get_children(old_node_name)

        temp_node_map, temp_tree_map = self.__get_maps()

        for p_node in parent_nodes:
            temp_tree_map = self.__delete_edge(p_node, old_node_name, temp_node_map, temp_tree_map)

        for c_node in child_nodes:
            temp_tree_map = self.__delete_edge(old_node_name, c_node, temp_node_map, temp_tree_map)

        temp_node_map = self.__delete_node(old_node_name, temp_node_map, temp_tree_map)
        temp_node_map = self.__add_node(new_node, temp_node_map)

        for p_node in parent_nodes:
            temp_tree_map = self.__add_edge(p_node, new_node.name, temp_node_map, temp_tree_map)

        for c_node in child_nodes:
            temp_tree_map = self.__add_edge(new_node.name, c_node, temp_node_map, temp_tree_map)

        self.node_map = temp_node_map
        self.tree_map = temp_tree_map

    def connect_nodes(self, parent_node_name, child_node_name):
        self.tree_map = self.__add_edge(parent_node_name=parent_node_name, child_node_name=child_node_name)

    def disconnect_nodes(self, parent_node_name, child_node_name):
        self.tree_map = self.__delete_edge(parent_node_name=parent_node_name, child_node_name=child_node_name)

    def insert_node(self, node, parent_node_name, child_node_name=None, append=False):
        """
        """
        temp_node_map, temp_tree_map = self.__get_maps()
        temp_node_map = self.__add_node(node, temp_node_map)

        if child_node_name is None:
            temp_tree_map = self.__add_edge(parent_node_name, node.name, temp_node_map, temp_tree_map)
        else:
            if append:
                temp_tree_map = self.__add_edge(parent_node_name, node.name, temp_node_map, temp_tree_map)
                temp_tree_map = self.__add_edge(node.name, child_node_name, temp_node_map, temp_tree_map)
            else:
                temp_tree_map = self.__delete_edge(parent_node_name, child_node_name, temp_node_map, temp_tree_map)
                temp_tree_map = self.__add_edge(parent_node_name, node.name, temp_node_map, temp_tree_map)
                temp_tree_map = self.__add_edge(node.name, child_node_name, temp_node_map, temp_tree_map)

        self.tree_map = temp_tree_map
        self.node_map = temp_node_map

    def insert_tree(self, tree, parent_node_name, child_node_name=None,
                    append=False, new_name=None, prefix=None):
        """

        :param tree:
        :param parent_node_name:
        :param child_node_name:
        :param append:
        :param new_name:
        :param prefix:
        :return:

        3 Cases:
            Add all nodes, other than Input or Output, from tree to graph.
            Add internal connections of Tree
            for p_node, c_node_list in tree.tree_map.items():
                if p_node is Output:
                    pass
                else:
                    for c_node in c_node_list:
                        if c_node is Input:
                            pass
                        else:
                            connect(c_node ---> c_node)

            1.  parent_node_name = A (not Input)
                child_node_name = B
                for c_node of tree.get_parents(tree.Input.name):
                    connect(A ---> c_node)
                for c_node of tree.get_children(tree.Output.name):
                    connect(c_node ---> B)
                if not append:
                    remove(A ---> B)

            2.  parent_node_name = A (not Input)
                child_node_name = None
                for c_node of tree.get_parents(tree.Input.name):
                    connect(self.Input.name -> c_node)
                for c_node of tree.get_children(tree.Output.name):
                    connect(c_node ---> A)

            3.  parent_node_name = Self.Input.name
                child_node_name = None
                for p_node of self.get_parents(self.Input.name):
                    connect(tree.get_children(tree.Output.name) ---> p_node)
                for c_node of tree.get_parents(tree.Input.name):
                    connect(self.Input.name ---> c_node)
        """
        temp_node_map, temp_tree_map = self.__get_maps()
        name_map = {}
        for node in tree.node_map.values():
            if not isinstance(node, (Input, Output)):
                if prefix is None:
                    new_node = node.copy_node(new_name=new_name, prefix=tree.name)
                else:
                    new_node = node.copy_node(new_name=new_name, prefix=prefix)
                name_map[node.name] = new_node.name
                temp_node_map = self.__add_node(new_node, temp_node_map)

        for p_node, c_node_list in tree.tree_map.items():
            if p_node != tree.Output.name:
                for c_node in c_node_list:
                    if c_node != tree.Input.name:
                        temp_tree_map = self.__add_edge(name_map[p_node],
                                                        name_map[c_node],
                                                        node_map=temp_node_map,
                                                        tree_map=temp_tree_map)

        if child_node_name is None:
            if parent_node_name == self.Input.name:
                before_output = tree.get_children(tree.Output.name).pop()
                for p_node in self.get_parents(self.Input.name):
                    if not append:
                        temp_tree_map = self.__delete_edge(p_node,
                                                           self.Input.name,
                                                           node_map=temp_node_map,
                                                           tree_map=temp_tree_map)
                    temp_tree_map = self.__add_edge(p_node,
                                                    name_map[before_output],
                                                    node_map=temp_node_map,
                                                    tree_map=temp_tree_map)
                for c_node in tree.get_parents(tree.Input.name):
                    temp_tree_map = self.__add_edge(name_map[c_node],
                                                    self.Input.name,
                                                    node_map=temp_node_map,
                                                    tree_map=temp_tree_map)
            else:
                for c_node in tree.get_parents(tree.Input.name):
                    temp_tree_map = self.__add_edge(name_map[c_node],
                                                    self.Input.name,
                                                    node_map=temp_node_map,
                                                    tree_map=temp_tree_map)
                for c_node in tree.get_children(tree.Output.name):
                    temp_tree_map = self.__add_edge(parent_node_name,
                                                    name_map[c_node],
                                                    node_map=temp_node_map,
                                                    tree_map=temp_tree_map)
        else:
            if not append:
                temp_tree_map = self.__delete_edge(parent_node_name,
                                                   child_node_name,
                                                   node_map=temp_node_map,
                                                   tree_map=temp_tree_map)
            for c_node in tree.get_parents(tree.Input.name):
                temp_tree_map = self.__add_edge(name_map[c_node],
                                                child_node_name,
                                                node_map=temp_node_map,
                                                tree_map=temp_tree_map)
            for c_node in tree.get_children(tree.Output.name):
                temp_tree_map = self.__add_edge(parent_node_name,
                                                name_map[c_node],
                                                node_map=temp_node_map,
                                                tree_map=temp_tree_map)

        self.tree_map = temp_tree_map
        self.node_map = temp_node_map

    @property
    def node_names(self):
        return self.node_map.keys()

    @property
    def nodes(self):
        return self.node_map.values()

    @property
    def Input(self):
        return self.input

    @property
    def Output(self):
        return self.root
