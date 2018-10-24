import numpy as np
import matplotlib.pyplot as plt
from .helpers import pol2rect


class Rim:
    'Rim definition.'

    def __init__(self, radius, area, I11, I22,
                 I33, Iw, young_mod, shear_mod,
                 sec_type='general', sec_params={}):
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
        I33 = 2*(t*(h+t)**3)/12 + 2*((w-t)*t**3/12 + (w-t)*t*(h/2)**2)
        I22 = 2*(t*(w+t)**3)/12 + 2*((h-t)*t**3/12 + (h-t)*t*(w/2)**2)

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

        area = (w+t)*t + 2*(h-t)*t

        # Torsion and warping constants
        # homepage.tudelft.nl/p3r3s/b16_chap7.pdf

        I11 = 1.0/3.0 * t**3 * (w + 2*(h-t))
        Iw = (t*h**3*w**2/12) * (3*h + 2*w)/(6*h + w)

        # Moments of area -----------------------------
        # Centroid location
        y_c = (h-t)*t*h / area
        I33 = (w+t)*t**3/12 + (w+t)*t*y_c**2 +\
            2 * (t*(h-t)**3/12 + (h-t)*t*(h/2 - y_c)**2)
        I22 = (t*w**3)/12 + 2*(((h-t)*t**3)/12 + (h-t)*t*(w/2)**2)

        # Shear center --------------------------------
        y_s = -3*h**2/(6*h + w)

        r = cls(radius=radius,
                area=area, I11=I11, I22=I22, I33=I33, Iw=Iw,
                young_mod=young_mod, shear_mod=shear_mod,
                sec_type='C', sec_params={'closed': False,
                                          'w': w, 'h': h, 't': t,
                                          'y_c': y_c, 'y_s': y_s})

        return r


class Hub:
    """Hub consisting of two parallel, circular flanges.

    Args:
        diameter_left: diameter of the left-side hub flange.
        diameter_right: diameter of the drive-side hub flange.
        width_left: distance from rim plane to left-side flange.
        width_right: distance from rim plane to drive-side flange.

    Usage:
        Symmetric:           Hub(diameter=0.05, width=0.05)
        Asymmetric, specify: Hub(diameter=0.05, width_left=0.03, width_right=0.02)
        Asymmetric, offset:  Hub(diameter_left=0.04, diameter_right=0.06, width=0.05, offset=0.01)
    """

    def __init__(self, diameter=None, diameter_left=None, diameter_right=None,
                 width=None, width_left=None, width_right=None, offset=None):

        # Set flange diameters
        self.diameter_left = diameter
        self.diameter_right = diameter

        if isinstance(diameter_left, float):
            self.diameter_left = diameter_left
        if isinstance(diameter_right, float):
            self.diameter_right = diameter_right

        # Set flange widths
        if isinstance(width, float):
            if offset is None:
                offset = 0.

            self.width_left = width/2 + offset
            self.width_right = width/2 - offset

            if (width_left is not None) or (width_right is not None):
                raise ValueError('Cannot specify width_left or width_right when using the offset parameter.')

        elif isinstance(width_left, float) and isinstance(width_right, float):
            self.width_left = width_left
            self.width_right = width_right
        else:
            raise ValueError('width_left and width_right must both be defined if not using the width parameter.')


class Spoke:
    """Spoke definition.

    Args:
        rim_pt: location of the spoke nipple as (R, theta, z)
        hub_pt: location of the hub eyelet as (R, theta, z)
    """

    def calc_k(self, tension=True):
        """Calculate matrix relating force and moment at rim due to the
        spoke under a rim displacement (u,v,w) and rotation phi"""

        n = self.n                      # spoke vector
        e3 = np.array([0., 0., 1.0])  # rim axial vector

        # Spoke nipple offset vector (relative to shear center)
        # TODO: Correctly calculate v-component of b_s based on rim radius.
        #       Set to zero for now.
        b = np.array([self.rim_pt[2], 0., 0.])

        K_e = self.EA / self.length

        if tension:
            K_t = self.tension / self.length
        else:
            K_t = 0.

        k_f = K_e*np.outer(n, n) + K_t*(np.eye(3) - np.outer(n, n))

        # Change in force applied by spoke due to rim rotation, phi
        dFdphi = k_f.dot(np.cross(e3, b).reshape((3, 1)))

        # Change in torque applied by spoke due to rim rotation
        dTdphi = np.cross(b, e3).dot(k_f).dot(np.cross(b, e3))

        k = np.zeros((4, 4))

        k[0:3, 0:3] = k_f
        k[0:3, 3] = dFdphi.reshape((3))
        k[3, 0:3] = dFdphi.reshape(3)
        k[3, 3] = dTdphi

        return k

    def calc_k_geom(self):
        'Calculate the coefficient of the tension-dependent spoke stiffness matrix.'

        n = self.n
        e3 = np.array([0., 0., 1.])

        # Spoke nipple offset vector (relative to shear center)
        # TODO: Correctly calculate v-component of b_s based on rim radius.
        #       Set to zero for now.
        b = np.array([self.rim_pt[2], 0., 0.])

        k_f = (1./self.length) * (np.eye(3) - np.outer(n, n))

        # Change in force applied by spoke due to rim rotation, phi
        dFdphi = k_f.dot(np.cross(e3, b).reshape((3, 1)))

        # Change in torque applied by spoke due to rim rotation
        dTdphi = np.cross(b, e3).dot(k_f).dot(np.cross(b, e3))

        k = np.zeros((4, 4))

        k[0:3, 0:3] = k_f
        k[0:3, 3] = dFdphi.reshape((3))
        k[3, 0:3] = dFdphi.reshape(3)
        k[3, 3] = dTdphi

        return k

    def __init__(self, rim_pt, hub_pt, diameter, young_mod):
        self.EA = np.pi / 4 * diameter**2 * young_mod
        self.diameter = diameter
        self.young_mod = young_mod
        self.tension = 0.

        self.rim_pt = rim_pt  # (R, theta, offset)
        self.hub_pt = hub_pt  # (R, theta, z)

        du = hub_pt[2] - rim_pt[2]
        dv = (rim_pt[0] - hub_pt[0]) +\
            hub_pt[0]*(1 - np.cos(hub_pt[1] - rim_pt[1]))
        dw = hub_pt[0]*np.sin(hub_pt[1] - rim_pt[1])

        self.length = np.sqrt(du**2 + dv**2 + dw**2)

        # Spoke axial unit vector
        self.n = np.array([du, dv, dw]) / self.length

        # Approximate projected angles
        self.alpha = np.arctan(du / dv)
        self.beta = np.arctan(dw / dv)


class BicycleWheel:
    """Bicycle wheel definition.

    Defines a bicycle wheel including geometry, spoke properties, and rim
    properties. Instances of the BicycleWheel class can be used as an input
    for theoretical calculations and FEM models.
    """

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
                hub_pt = (self.hub.diameter_right/2, theta_hub, self.hub.width_right)
            else:
                hub_pt = (self.hub.diameter_left/2, theta_hub, -self.hub.width_left)

            spoke = Spoke(rim_pt, hub_pt, diameter, young_mod)
            self.spokes.append(spoke)

    def apply_tension(self, T_avg):
        'Apply tension to spokes based on average radial tension.'

        # Assume that there are only two tensions in the wheel: left and right
        # and that spokes alternate left, right, left, right...
        s_0 = self.spokes[0]
        s_1 = self.spokes[1]
        T_0 = 2 * T_avg * np.abs(s_1.n[0]) /\
            (np.abs(s_0.n[0]*s_1.n[1]) + np.abs(s_1.n[0]*s_0.n[1]))
        T_1 = 2 * T_avg * np.abs(s_0.n[0]) /\
            (np.abs(s_0.n[0]*s_1.n[1]) + np.abs(s_1.n[0]*s_0.n[1]))

        for i in range(0, len(self.spokes), 2):
            self.spokes[i].tension = T_0

        for i in range(1, len(self.spokes), 2):
            self.spokes[i].tension = T_1

    def calc_kbar(self, tension=True):
        'Calculate smeared-spoke stiffness matrix'

        k_bar = np.zeros((4, 4))

        for s in self.spokes:
            k_bar = k_bar + s.calc_k(tension=tension)/(2*np.pi*self.rim.radius)

        return k_bar

    def calc_kbar_geom(self):
        'Calculate smeared-spoke stiffness matrix, geometric component'

        k_bar = np.zeros((4, 4))

        # Get scaling factor for tension on each side of the wheel
        s_0 = self.spokes[0]
        s_1 = self.spokes[1]
        T_d = np.abs(s_0.n[0]*s_1.n[1]) + np.abs(s_1.n[0]*s_0.n[1])

        for s in self.spokes:
            k_bar = k_bar + \
                np.abs(s.n[0])/T_d * s.calc_k_geom()/(np.pi*self.rim.radius)

        return k_bar

    def calc_mass(self):
        'Calculate total mass of the wheel in kilograms.'

        # TODO
        pass

    def calc_rot_inertia(self):
        'Calculate rotational inertia about the hub axle.'

        # TODO
        pass

    def draw(self, ax, opts={}):
        'Draw a graphical representation of the wheel'

        # Set default drawing options
        opts_d = {'axes_off': True,
                  'rim_color': 'black',
                  'rim_width': 2,
                  'hub_ds_color': 'black',
                  'hub_ls_color': 'gray',
                  'hub_ds_width': 1,
                  'hub_ls_width': 1,
                  'spk_ds_width': 1,
                  'spk_ls_width': 1,
                  'spk_ds_color': 'black',
                  'spk_ls_color': 'gray'}

        opts_d.update(opts)

        # Draw rim
        R = self.rim.radius
        ax.add_artist(plt.Circle((0, 0), 1.0, fill=False,
                                 color=opts_d['rim_color'],
                                 linewidth=opts_d['rim_width']))

        # Draw hub
        ax.add_artist(plt.Circle((0, 0), (self.hub.diameter_left/2)/R, fill=False,
                                 color=opts_d['hub_ls_color'],
                                 linewidth=opts_d['hub_ls_width'],
                                 zorder=-1))
        ax.add_artist(plt.Circle((0, 0), (self.hub.diameter_right/2)/R, fill=False,
                                 color=opts_d['hub_ds_color'],
                                 linewidth=opts_d['hub_ds_width'],
                                 zorder=1))

        # Draw spokes
        for i, s in enumerate(self.spokes):

            # Determine if its a drive-side or left-side spoke
            if s.hub_pt[2] < 0:
                ax.plot([pol2rect(s.hub_pt)[0]/R, pol2rect(s.rim_pt)[0]/R],
                        [pol2rect(s.hub_pt)[1]/R, pol2rect(s.rim_pt)[1]/R],
                        color=opts_d['spk_ds_color'],
                        linewidth=opts_d['spk_ds_width'],
                        zorder=i+2)
            else:
                ax.plot([pol2rect(s.hub_pt)[0]/R, pol2rect(s.rim_pt)[0]/R],
                        [pol2rect(s.hub_pt)[1]/R, pol2rect(s.rim_pt)[1]/R],
                        color=opts_d['spk_ls_color'],
                        linewidth=opts_d['spk_ls_width'],
                        zorder=-(i+2))

        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        if opts_d['axes_off']:
            ax.set_axis_off()

    def __init__(self):
        self.spokes = []
        self.rim = None
        self.hub = None
