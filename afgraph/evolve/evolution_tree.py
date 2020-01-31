from ..function.node import *
from ..function.tree import *
import numpy as np


def generate_tree(name='test'):
    p_branch = .2
    p_infertile = .1
    p_channel = 1 - p_branch - p_infertile
    decay = .25

    branch_nodes = [Max, Sum, Mean, Min, Product, Median]
    infertile_nodes = [Constant, Input, Uniform, Normal]
    channel_node = [Square, Reciprocal, Sqrt, Exp, Sin, Cos, ArcTan, Abs, Sign]

    max_node_count = 10
    max_num_branches = 3
    max_size_branch = 3
    branch_disribution = np.random.randint

    tree = FunctionTree(name=name)

    stack = [tree.Output.name]
    c_node = tree.Input.name

    branch_count = 0
    nodes_count = 0
    while len(stack) > 0:
        p_node = stack.pop(0)
        if branch_count > max_num_branches:
            p_infertile += p_branch/2
            p_channel += p_branch/2
            p_branch = 0

            print(p_infertile, p_channel, p_branch)
        elif nodes_count > max_node_count - branch_count*(max_size_branch/2 -1):

            decay_amount = p_channel*decay
            p_channel -= decay_amount
            p_infertile += decay_amount

        elif nodes_count > 2* max_node_count:
            p_channel = 0
            p_infertile = 1

        new_node_type = np.random.choice(['branch', 'infertile', 'channel'], p=[p_branch, p_infertile, p_channel])
        if new_node_type == 'branch':
            new_node = np.random.choice(branch_nodes)
            num_nodes = branch_disribution(2, max_size_branch+1)
            branch_count += 1

        elif new_node_type == 'infertile':
            new_node = np.random.choice(infertile_nodes)
            num_nodes = 1
        else:
            new_node = np.random.choice(channel_node)
            num_nodes = 1

        new_node = new_node(name=str(nodes_count))

        print('Node to be added: ', new_node.latex, new_node.name)

        try:
            if new_node_type == 'infertile':
                tree.insert(new_node, parent_item=p_node)
            else:
                tree.insert(new_node, parent_item=p_node, child_item=c_node)
            print('Tree Map')
            print(tree.tree_map)
            nodes_count += num_nodes
            if new_node_type != 'infertile':
                for _ in range(num_nodes):
                    stack.append(new_node.name)
        except:
            stack.insert(0, p_node)

    return tree


class FunctionGenerator:
    """Generator of FunctionTrees

    # Arguments:
        base_cohert_name: A String, used as the base of naming new Trees
            Ex: A = FunctionGenerator(base_cohert_name = 'test')
                next(A) --> FunctionTree Object with name 'test_1'
                next(A) --> FunctionTree Object with name 'test_2'
                ...

        max_nodes: An Integer, maximum number of nodes in a graph

        max_branches: An Integer, maximum number of branch type function
            nodes in a graph

        node_type_probability: A dictionary, representing a discrete node distribution
            Ex: {'channel': 1/3, 'infertile': 1/3, 'branch': 1/3}

        node_probability: A numpy distribution,

        channel_nodes: A dictionary with FunctionNodes, used as the key,
            and "probability" of selecting that node when a channel node is needed

        infertile_nodes: A dictionary with FunctionNodes, used as the key,
            and "probability" of selecting that node when a infertile node is needed

        branch_nodes: A dictionary with FunctionNodes, used as the key,
            and "probability" of selecting that node when a branch node is needed



    # Returns (When called)
        FunctionTree,

    Logic:
        Breadth first creation!
        Determine_number of nodes
        Determine how many branch and infertile nodes

    """
    pass