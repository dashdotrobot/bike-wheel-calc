import numpy as np

class SpokeSection:
    'Section definition for spoke elements'

    def __init__(self, d_spoke, young_mod):
        self.young_mod = young_mod
        self.area = np.pi / 4 * d_spoke**2


