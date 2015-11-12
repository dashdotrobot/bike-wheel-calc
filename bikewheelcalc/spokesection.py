import numpy as np


class SpokeSection:
    'Section definition for spoke elements'

    def __init__(self, d_spoke, young_mod):
        self.young_mod = young_mod           # Young's modulus
        self.area = np.pi / 4 * d_spoke**2   # cross-sectional area
        self.I = np.pi / 4 * (d_spoke/2)**4  # second moment of area

        self.density = 8000.0  # [kg/m^3] stainless steel 304
