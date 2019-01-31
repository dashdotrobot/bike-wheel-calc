'Core classes BicycleWheel, Hub, Rim, and Spoke'

import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from .helpers import pol2rect


class Rim:
    'Rim definition.'

    def __init__(self, radius, area,
                 I_rad, I_lat, J_tor, I_warp,
                 young_mod, shear_mod, density=None,
                 sec_type='general', sec_params={}):
        self.radius = radius
        self.area = area
        self.I_rad = I_rad
        self.I_lat = I_lat
        self.J_tor = J_tor
        self.I_warp = I_warp
        self.young_mod = young_mod
        self.shear_mod = shear_mod
        self.density = density
        self.sec_type = sec_type
        self.sec_params = sec_params

    @classmethod
    def general(cls, radius, area,
                I_rad, I_lat, J_tor, I_warp,
                young_mod, shear_mod, density=None):
        'Define a rim with arbitrary section properties.'

        r = cls(radius=radius, area=area,
                I_rad=I_rad, I_lat=I_lat, J_tor=J_tor, I_warp=I_warp,
                young_mod=young_mod, shear_mod=shear_mod, density=density,
                sec_type='general', sec_params={})

        return r

    @classmethod
    def box(cls, radius, w, h, t, young_mod, shear_mod, density=None):
        """Define a rim from a box cross-section.

        Args:
            w: width of the rim cross-section, from midline to midline.
            h: height of the rim cross-section (radial direction).
            t: wall thickness."""

        area = 2*(w+t/2)*t + 2*(h-t/2)*t

        # Torsion constant
        J_tor = 2*t*(w*h)**2 / (w + h)

        # Moments of area
        I_rad = 2*(t*(h+t)**3)/12 + 2*((w-t)*t**3/12 + (w-t)*t*(h/2)**2)
        I_lat = 2*(t*(w+t)**3)/12 + 2*((h-t)*t**3/12 + (h-t)*t*(w/2)**2)

        # Warping constant, closed thin-walled section
        I_warp = I_warp

        r = cls(radius=radius, area=area,
                I_rad=I_rad, I_lat=I_lat, J_tor=J_tor, I_warp=I_warp,
                young_mod=young_mod, shear_mod=shear_mod, density=density,
                sec_type='box', sec_params={'closed': True,
                                            'w': w, 'h': h, 't': t})

        return r

    @classmethod
    def C_channel(cls, radius, w, h, t, young_mod, shear_mod, density=None):
        'Construct a rim from a C channel cross-section.'

        area = (w+t)*t + 2*(h-t)*t

        # Torsion and warping constants
        # homepage.tudelft.nl/p3r3s/b16_chap7.pdf

        J_tor = 1.0/3.0 * t**3 * (w + 2*(h-t))
        I_warp = (t*h**3*w**2/12) * (3*h + 2*w)/(6*h + w)

        # Moments of area -----------------------------
        # Centroid location
        y_c = (h-t)*t*h / area
        I_rad = (w+t)*t**3/12 + (w+t)*t*y_c**2 +\
            2 * (t*(h-t)**3/12 + (h-t)*t*(h/2 - y_c)**2)
        I_lat = (t*w**3)/12 + 2*(((h-t)*t**3)/12 + (h-t)*t*(w/2)**2)

        # Shear center --------------------------------
        y_s = -3*h**2/(6*h + w)

        r = cls(radius=radius, area=area,
                I_rad=I_rad, I_lat=I_lat, J_tor=J_tor, I_warp=I_warp,
                young_mod=young_mod, shear_mod=shear_mod, density=density,
                sec_type='C', sec_params={'closed': False,
                                          'w': w, 'h': h, 't': t,
                                          'y_c': y_c, 'y_s': y_s, 'y_0': y_c - y_s})

        return r

    def calc_mass(self):
        'Return the rim mass'

        if self.density is not None:
            return self.density * 2*np.pi*self.radius * self.area
        else:
            return None

    def calc_rot_inertia(self):
        'Return the rotational inertia about the axle'

        if self.density is not None:
            return self.calc_mass() * self.radius**2
        else:
            return None


class Hub:
    """Hub consisting of two parallel, circular flanges.

    Args:
        diameter_nds: diameter of the left-side hub flange.
        diameter_ds: diameter of the drive-side hub flange.
        width_nds: distance from rim plane to left-side flange.
        width_ds: distance from rim plane to drive-side flange.

    Usage:
        Symmetric:           Hub(diameter=0.05, width=0.05)
        Asymmetric, specify: Hub(diameter=0.05, width_nds=0.03, width_ds=0.02)
        Asymmetric, offset:  Hub(diameter_nds=0.04, diameter_ds=0.06, width=0.05, offset=0.01)
    """

    def __init__(self, diameter=None, diameter_nds=None, diameter_ds=None,
                 width=None, width_nds=None, width_ds=None, offset=None):

        # Set flange diameters
        self.diameter_nds = diameter
        self.diameter_ds = diameter

        if isinstance(diameter_nds, float):
            self.diameter_nds = diameter_nds
        if isinstance(diameter_ds, float):
            self.diameter_ds = diameter_ds

        # Set flange widths
        if isinstance(width, float):
            if offset is None:
                offset = 0.

            self.width_nds = width/2 + offset
            self.width_ds = width/2 - offset

            if (width_nds is not None) or (width_ds is not None):
                raise ValueError('Cannot specify width_left or width_right when using the offset parameter.')

        elif isinstance(width_nds, float) and isinstance(width_ds, float):
            self.width_nds = width_nds
            self.width_ds = width_ds
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

        n = self.n                   # spoke vector
        e3 = np.array([0., 0., 1.])  # rim axial vector

        K_e = self.EA / self.length

        if tension:
            K_t = self.tension / self.length
        else:
            K_t = 0.

        k_f = K_e*np.outer(n, n) + K_t*(np.eye(3) - np.outer(n, n))

        # Change in force applied by spoke due to rim rotation, phi
        dFdphi = k_f.dot(np.cross(e3, self.b).reshape((3, 1)))

        # Change in torque applied by spoke due to rim rotation
        dTdphi = np.cross(self.b, e3).dot(k_f).dot(np.cross(self.b, e3))

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

        k_f = (1./self.length) * (np.eye(3) - np.outer(n, n))

        # Change in force applied by spoke due to rim rotation, phi
        dFdphi = k_f.dot(np.cross(e3, self.b).reshape((3, 1)))

        # Change in torque applied by spoke due to rim rotation
        dTdphi = np.cross(self.b, e3).dot(k_f).dot(np.cross(self.b, e3))

        k = np.zeros((4, 4))

        k[0:3, 0:3] = k_f
        k[0:3, 3] = dFdphi.reshape((3))
        k[3, 0:3] = dFdphi.reshape(3)
        k[3, 3] = dTdphi

        return k

    def calc_mass(self):
        'Return the spoke mass'

        if self.density is not None:
            return self.density * self.length * np.pi/4*self.diameter**2
        else:
            return None

    def calc_rot_inertia(self):
        'Return the spoke rotational inertia about its center-of-mass'

        if self.density is not None:
            return self.calc_mass()*(self.length*self.n[1])**2 / 12.
        else:
            return None

    def calc_tension_change(self, d, a=0.):
        'Calculate change in tension given d=(u,v,w,phi) and a tightening adjustment a'

        # Assume phi=0 if not given
        if len(d) < 4:
            d = np.append(d, 0.)

        # u_n = u_s + phi(e_3 x b)
        e3 = np.array([0., 0., 1.])
        un = np.array([d[0], d[1], d[2]]) + d[3]*np.cross(e3, self.b)

        return self.EA/self.length * (a - self.n.dot(un))

    def __init__(self, rim_pt, hub_pt, diameter, young_mod, density=None):
        self.EA = np.pi / 4 * diameter**2 * young_mod
        self.diameter = diameter
        self.young_mod = young_mod
        self.density = density
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

        # Spoke nipple offset vector (relative to shear center)
        # TODO: Correctly calculate v-component of b_s based on rim radius.
        #       Set to zero for now.
        self.b = np.array([rim_pt[2], 0., 0.])

        # Approximate projected angles
        self.alpha = np.arctan(du / dv)
        self.beta = np.arctan(dw / dv)


class BicycleWheel:
    """Bicycle wheel definition.

    Defines a bicycle wheel including geometry, spoke properties, and rim
    properties. Instances of the BicycleWheel class can be used as an input
    for theoretical calculations and FEM models.
    """


    def reorder_spokes(self):
        'Ensure that spokes are ordered according to theta_rim'

        a = np.argsort([s.rim_pt[1] for s in self.spokes])
        self.spokes = [self.spokes[i] for i in a]

    def lace_radial(self, n_spokes, diameter, young_mod, offset=0.0, density=None):
        'Add spokes in a radial spoke pattern.'

        return self.lace_cross(n_spokes, 0, diameter=diameter, young_mod=young_mod,
                               offset=offset, density=density)

    def lace_cross_nds(self, n_spokes, n_cross, diameter, young_mod, offset=0., density=None):
        'Add spokes on the non-drive-side with n_cross crossings'

        # Start with a leading spoke at theta=0, and alternate
        for s in range(n_spokes):
            theta_rim = 2*np.pi/n_spokes * s
            s_dir = 2*((s + 1) % 2) - 1  # [1, -1, 1, ...]

            rim_pt = (self.rim.radius, theta_rim, offset)
            hub_pt = (self.hub.diameter_nds/2,
                      theta_rim + 2*np.pi/n_spokes*n_cross*s_dir,
                      self.hub.width_nds)

            self.spokes.append(Spoke(rim_pt, hub_pt, diameter, young_mod, density=density))

        self.reorder_spokes()
        return True

    def lace_cross_ds(self, n_spokes, n_cross, diameter, young_mod, offset=0., density=None):
        'Add spokes on the drive-side with n_cross crossings'

        # Start with a leading spoke at theta=0, and alternate
        for s in range(n_spokes):
            theta_rim = 2*np.pi/n_spokes * (s + 0.5)
            s_dir = 2*((s + 1) % 2) - 1  # [1, -1, 1, ...]

            rim_pt = (self.rim.radius, theta_rim, -offset)
            hub_pt = (self.hub.diameter_ds/2,
                      theta_rim + 2*np.pi/n_spokes*n_cross*s_dir,
                      -self.hub.width_ds)

            self.spokes.append(Spoke(rim_pt, hub_pt, diameter, young_mod, density=density))

        self.reorder_spokes()
        return True

    def lace_cross(self, n_spokes, n_cross, diameter, young_mod, offset=0.0, density=None):
        'Generate spokes in a "cross" pattern with n_cross crossings.'

        # Remove any existing spokes
        self.spokes = []

        self.lace_cross_ds(n_spokes//2, n_cross, diameter, young_mod, offset, density=density)
        self.lace_cross_nds(n_spokes//2, n_cross, diameter, young_mod, offset, density=density)

        return True

    def apply_tension(self, T_avg=None, T_left=None, T_right=None):
        'Apply tension to spokes based on average radial tension.'

        # Assume that there are only two tensions in the wheel: left and right
        # and that spokes alternate left, right, left, right...
        s_l = self.spokes[0]
        s_r = self.spokes[1]

        if T_avg is not None:  # Specify average radial tension
            T_l = 2 * T_avg * np.abs(s_r.n[0]) /\
                (np.abs(s_l.n[0]*s_r.n[1]) + np.abs(s_r.n[0]*s_l.n[1]))
            T_r = 2 * T_avg * np.abs(s_l.n[0]) /\
                (np.abs(s_l.n[0]*s_r.n[1]) + np.abs(s_r.n[0]*s_l.n[1]))

            for i in range(0, len(self.spokes), 2):
                self.spokes[i].tension = T_l

            for i in range(1, len(self.spokes), 2):
                self.spokes[i].tension = T_r

        elif T_right is not None:  # Specify right-side tension
            T_r = T_right
            T_l = np.abs(s_r.n[0]/s_l.n[0]) * T_right

        elif T_left is not None:  # Specify left-side tension
            T_l = T_left
            T_r = np.abs(s_l.n[0]/s_r.n[0]) * T_left

        else:
            raise TypeError('Must specify one of the following arguments: T_avg, T_left, or T_right.')

        # Apply tensions
        for i in range(0, len(self.spokes), 2):
            self.spokes[i].tension = T_l

        for i in range(1, len(self.spokes), 2):
            self.spokes[i].tension = T_r

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

        m_rim = self.rim.calc_mass()
        if m_rim is None:
            m_rim = 0.
            warn('Rim density is not specified.')

        m_spokes = np.array([s.calc_mass() for s in self.spokes])
        if np.any(m_spokes == None):
            m_spokes = np.where(m_spokes == None, 0., m_spokes)
            warn('Some spoke densities are not specified.')

        return m_rim + np.sum(m_spokes)

    def calc_rot_inertia(self):
        'Calculate rotational inertia about the hub axle.'

        I_rim = self.rim.calc_rot_inertia()
        if I_rim is None:
            I_rim = 0.
            warn('Rim density is not specified.')

        I_spk = np.array([s.calc_rot_inertia() for s in self.spokes])
        if np.any(I_spk == None):
            I_spokes = 0.
            warn('Some spoke densities are not specified.')
        else:
            mr2_spk = np.array([s.calc_mass()*(0.5*(s.hub_pt[0] + s.rim_pt[0]))**2
                                for s in self.spokes])
            I_spokes = np.sum(I_spk) + np.sum(mr2_spk)

        return I_rim + I_spokes

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
