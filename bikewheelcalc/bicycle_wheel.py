#!/usr/bin/env python

"""Definition of a bicycle wheel including geometry, spoke properties,
   and rim properties. Instances of the BicycleWheel class can be used
   as an input for theoretical calculations and FEM models."""

import numpy as np


class BicycleWheel:

    class Rim:
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
            'Construct a rim with arbitrary section properties.'

            r = cls(radius=radius,
                    area=area, I11=I11, I22=I22, I33=I33, Iw=Iw,
                    young_mod=young_mod, shear_mod=shear_mod,
                    sec_type='general', sec_params={})

            return r

        @classmethod
        def box(cls, radius, young_mod, shear_mod, w, h, t):
            'Construct a rim from a box cross-section.'

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

            # Torsion and warping constants ---------------
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

        # Remove any existing spokes
        self.spokes = []

        for s in range(n_spokes):
            theta = 2 * np.pi / n_spokes * s
            side = 2 * ((s + 1) % 2) - 1

            rim_pt = (self.rim.radius, theta, side * offset)

            if side == 1:
                hub_pt = (self.hub.radius1, theta, self.hub.width1)
            else:
                hub_pt = (self.hub.radius2, theta, -self.hub.width2)

            spoke = self.Spoke(rim_pt, hub_pt, diameter, young_mod)
            self.spokes.append(spoke)

    def __init__(self):
        self.spokes = []

# Testing code
if False:
    w = BicycleWheel()
    w.rim = w.Rim.C_channel(radius=0.3,
                            young_mod=69.0e9, shear_mod=26.0e9,
                            w=0.050, h=0.010, t=0.002)
    print 'J  = ', w.rim.I11
    print 'Iw = ', w.rim.Iw

    print w.rim.shear_mod*w.rim.I11
    print w.rim.young_mod*w.rim.Iw * 16 / w.rim.radius**2
