class FanInError(Exception):
    """Error raised when node does not have fan_in to support child_node"""
    def __init__(self, tree_name, p_node, c_node_list, c_node_name):
        self.tree_name = tree_name
        self.p_node = p_node
        self.p_node_name = p_node.name
        self.c_node_list = c_node_list
        self.c_node_name = c_node_name

    def __str__(self):
        if self.p_node.fan_in > 0:
            return "Tree: {t} \n"\
                   "{p} cannot take child, {c_node}. \n" \
                   "fan_in: {fan_in} \n" \
                   "child_list: {c_list}".format(t=self.tree_name,
                                                 p=self.p_node,
                                                 fan_in=self.p_node.fan_in,
                                                 c_node=self.c_node_name,
                                                 c_list=self.c_node_list)
        else:
            return "{p} cannot take child, {c_node}".format(p=self.p_node_name, c_node=self.c_node_name)


class ChildError(Exception):
    """Error raised when attempting to add child to Node that cannot take a child"""
    def __init__(self, tree_name, p_node_name, c_node_name):
        self.tree_name = tree_name
        self.p_node_name = p_node_name
        self.c_node_name = c_node_name

    def __str__(self):
        return "Tree: {t} \n"\
               "{p} cannot take child, {c}. {p} is infertile.".format(t=self.tree_name,
                                                                      p=self.p_node_name,
                                                                      c=self.c_node_name)


class ParentError(Exception):
    """Error raised when attempting to add parent to Node that cannot take a parent"""
    def __init__(self, tree_name, p_node_name, c_node_name):
        self.tree_name = tree_name
        self.p_node_name = p_node_name
        self.c_node_name = c_node_name

    def __str__(self):
        return "Tree: {t} \n"\
               "{c} cannot take parent, {p}. {c} is in-parentable.".format(t=self.tree_name,
                                                                           p=self.p_node_name,
                                                                           c=self.c_node_name)


class DepthError(Exception):
    """Error raised when attempting to add an edge that will excede a graph's max depth"""
    def __init__(self, tree_name, max_depth, p_node_name, p_node_depth, c_node_name):
        self.tree_name = tree_name
        self.max_depth = max_depth
        self.p_node_name = p_node_name
        self.p_node_depth = p_node_depth
        self.c_node_name = c_node_name

    def __str__(self):
        return "Tree: {t} \n"\
               "{p} is at depth {d}. Adding {c} will exceed tree max_depth, {m}.".format(t=self.tree_name,
                                                                                         d=self.p_node_depth,
                                                                                         p=self.p_node_name,
                                                                                         c=self.c_node_name,
                                                                                         m=self.max_depth)


class CloseLoopError(Exception):
    """Error raised when attempting to add an edge that would create"""
    def __init__(self, tree_name, p_node_name, c_node_name):
        self.tree_name = tree_name
        self.p_node_name = p_node_name
        self.c_node_name = c_node_name

    def __str__(self):
        return "Tree: {t} \n" \
               "{p} cannot take child {c}. Will create closed loop".format(t=self.tree_name,
                                                                           p=self.p_node_name,
                                                                           c=self.c_node_name)


class NodeNameError(Exception):
    """Error raised when referenced node does not exist in graph"""
    def __init__(self, tree_name, p_node_name=None, c_node_name=None, exists=False):
        # exists == False -> Nodes Don't Exists
        # exists == True -> Nodes Already Exists
        self.tree_name = tree_name
        self.node_name = p_node_name or c_node_name
        if p_node_name is None:
            self.node_type = 'Parent'
        else:
            self.node_type = 'Child'

        if exists:
            self.qualifier = 'already'
        else:
            self.qualifier = 'does not'

    def __str__(self):
        return "Tree: {t} \n" \
               "{ty} node, {n}, {q} exist in tree".format(t=self.tree_name,
                                                          q=self.qualifier,
                                                          ty=self.node_type,
                                                          n=self.node_name)
