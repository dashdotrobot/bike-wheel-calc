import numpy as np

class RigidBody:
    'Set of nodes with constained DOFs'

    def __init__(self, name, pos, nodes):

        self.name = name

        self.pos = np.array(pos)
        self.nodes = np.array(nodes)
        self.n_nodes = len(self.nodes)
