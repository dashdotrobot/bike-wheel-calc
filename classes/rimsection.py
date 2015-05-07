class RimSection:
    'Section definition for rim elements'

    def __init__(self, area, I11, I22, I33, young_mod, shear_mod, K2=1.0, K3=1.0):
        self.area = area
        self.I11 = I11
        self.I22 = I22
        self.I33 = I33
        self.young_mod = young_mod
        self.shear_mod = shear_mod
        self.K2 = K2  # transverse shear constant
        self.K3 = K3  # transverse shear constant
