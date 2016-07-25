import numpy as np


class BicycleWheel:
    """Bicycle wheel definition.

    Defines a bicycle wheel including geometry, spoke properties, and rim
    properties. Instances of the BicycleWheel class can be used as an input
    for theoretical calculations and FEM models.
    """

    class Rim:
        'Rim definition.'

        def __init__(self, radius, area, I11, I22,
                     I33, Iw, young_mod, shear_mod,
                     sec_type=None, sec_params=None):
            self.radius = radius
            self.area = area
            self.I11 = I11
            self.I22 = I22
            self.I33 = I33
            self.Iw = Iw
            self.young_mod = young_mod
            self.shear_mod = shear_mod
            self.sec_type = sec_type
            self.sec_params = sec_params

        @classmethod
        def general(cls, radius, area, I11, I22,
                    I33, Iw, young_mod, shear_mod):
            'Define a rim with arbitrary section properties.'

            r = cls(radius=radius,
                    area=area, I11=I11, I22=I22, I33=I33, Iw=Iw,
                    young_mod=young_mod, shear_mod=shear_mod,
                    sec_type='general', sec_params={})

            return r

        @classmethod
        def box(cls, radius, young_mod, shear_mod, w, h, t):
            """Define a rim from a box cross-section.

            Args:
                w: width of the rim cross-section, from midline to midline.
                h: height of the rim cross-section (radial direction).
                t: wall thickness."""

            area = 2*(w+t/2)*t + 2*(h-t/2)*t

            # Torsion constant
            I11 = 2*t*(w*h)**2 / (w + h)

            # Moments of area
            I33 = 2*(t*(h-t)**3/12) + 2*(w*t**3/12 + w*t*(h/2-t/2)**2)
            I22 = 2*(t*(w-t)**3/12) + 2*(h*t**3/12 + w*t*(w/2-t/2)**2)

            # Warping constant
            Iw = 0.0  # closed thin-walled section

            r = cls(radius=radius,
                    area=area, I11=I11, I22=I22, I33=I33, Iw=Iw,
                    young_mod=young_mod, shear_mod=shear_mod,
                    sec_type='box', sec_params={'closed': True,
                                                'w': w, 'h': h, 't': t})

            return r

        @classmethod
        def C_channel(cls, radius, young_mod, shear_mod, w, h, t):
            'Construct a rim from a C channel cross-section.'

            area = w*t + 2*(h-t)*t

            # Torsion and warping constants
            # www.cisc-icca.ca/files/technical/techdocs/updates/torsionprop.pdf
            dp = w  # - t
            bp = h  # - t/2
            a = 1.0 / (2.0 + dp/(3*bp))

            # I11 = (2.0/3.0) * t**3 * (bp + dp)
            I11 = 1.0/3.0 * t**3 * (dp + 2*bp)
            Iw = dp**2 * bp**3 * t * ((1.0-3*a)/6 + a**2/2 * (1+dp/(6*bp)))

            # Moments of area -----------------------------
            # Centroid location
            y_c = (w*t*(t/2) + 2*(h-t)*t*((h-t)/2+t)) / area
            I33 = w*t**3/12 + w*t*(y_c - t/2)**2 +\
                2 * (t*(h-t)**3/12 + (h-t)*t*(y_c - ((h-t)/2+t))**2)
            I22 = (t*w**3)/12 + 2*(((h-t/2)*t**3)/12 + (h-t/2)*t*(w/2)**2)

            # Shear center --------------------------------
            y_s = -bp*a

            r = cls(radius=radius,
                    area=area, I11=I11, I22=I22, I33=I33, Iw=Iw,
                    young_mod=young_mod, shear_mod=shear_mod,
                    sec_type='C', sec_params={'closed': False,
                                              'w': w, 'h': h, 't': t,
                                              'y_s': y_s})

            return r

    class Hub:
        """Hub definition.

        Args:
            diam1: diameter of drive-side flange.
            diam2: diameter of left-side flange.
            width1: distance from rim plane to drive-side flange midplane.
            width2: distance from rim plane to left-side flange midplane.
        """

        def __init__(self, diam1=None, diam2=None, width1=None, width2=None):
            self.width1 = width1
            self.diam1 = diam1
            self.width2 = width2
            self.diam2 = diam2

            if diam2 is None:
                self.diam2 = diam1

            if width2 is None:
                self.width2 = width1

            self.radius1 = self.diam1 / 2
            self.radius2 = self.diam2 / 2

    class Spoke:
        """Spoke definition.

        Args:
            rim_pt: location of the spoke nipple as (R, theta, z)
            hub_pt: location of the hub eyelet as (R, theta, z)
        """

        def __init__(self, rim_pt, hub_pt, diameter, young_mod):
            self.EA = np.pi / 4 * diameter**2 * young_mod
            self.diameter = diameter

            self.rim_pt = rim_pt  # (R, theta, offset)
            self.hub_pt = hub_pt  # (R, theta, z)

            du = hub_pt[2] - rim_pt[2]
            dv = (rim_pt[0] - hub_pt[0]) +\
                hub_pt[0]*(1 - np.cos(hub_pt[1] - rim_pt[1]))
            dw = hub_pt[0]*np.sin(hub_pt[1] - rim_pt[1])

            self.length = np.sqrt(du**2 + dv**2 + dw**2)
            self.alpha = np.arcsin(du / self.length)
            self.phi = np.arctan(dw / dv)

            self.n = np.array([np.cos(self.phi) * np.sin(self.alpha),
                               np.cos(self.phi) * np.cos(self.alpha),
                               np.sin(self.phi)])

    def lace_radial(self, n_spokes, diameter, young_mod, offset=0.0):
        'Generate spokes in a radial spoke pattern.'

        self.lace_cross(n_spokes, 0, diameter, young_mod, offset)

    def lace_cross(self, n_spokes, n_cross, diameter, young_mod, offset=0.0):
        'Generate spokes in a "cross" pattern with n_cross crossings.'

        # Remove any existing spokes
        self.spokes = []

        for s in range(n_spokes):
            theta_rim = 2*np.pi/n_spokes * s
            side = 2*((s + 1) % 2) - 1
            s_dir = 2*((s % 4) < 2) - 1

            rim_pt = (self.rim.radius, theta_rim, side*offset)

            theta_hub = theta_rim + 2*n_cross*s_dir * (2*np.pi/n_spokes)
            if side == 1:
                hub_pt = (self.hub.radius1, theta_hub, self.hub.width1)
            else:
                hub_pt = (self.hub.radius2, theta_hub, -self.hub.width2)

            spoke = self.Spoke(rim_pt, hub_pt, diameter, young_mod)
            self.spokes.append(spoke)

    def calc_mass(self):
        'Calculate total mass of the wheel in kilograms.'

        # TODO
        pass

    def calc_rot_inertia(self):
        'Calculate rotational inertia about the hub axle.'

        # TODO
        pass

    def __init__(self):
        self.spokes = []
