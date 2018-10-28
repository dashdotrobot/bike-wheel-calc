'RigidBody Class'

import numpy as np


class RigidBody:
    'Set of nodes with constained DOFs'

    def __init__(self, name, pos, nodes, node_id=None):

        self.name = name

        self.pos = np.array(pos)
        self.nodes = np.array(nodes)
        self.n_nodes = len(self.nodes)
        self.node_id = node_id
