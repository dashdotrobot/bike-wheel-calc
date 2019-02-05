'Rayleigh-Ritz solution to fully-coupled uvw-phi equations.'

import numpy as np
from numpy import pi


class ModeMatrix:
    """Solve coupled lateral, radial, and torsional deflections."""

    def B_theta(self, theta=[0.], comps=[0, 1, 2, 3]):
        'Matrix to transform mode coefficients to vector components.'

        # Convert scalar values to arrays
        theta = np.atleast_1d(theta)
        comps = np.atleast_1d(comps)

        B = np.zeros((len(theta)*len(comps), 4 + 8*self.n_modes))

        # For each angle theta
        for it in range(len(theta)):

            # Zero mode
            for ic, c in enumerate(comps):
                B[len(comps)*it + ic, c] = 1.

            # Higher modes
            for n in range(1, self.n_modes + 1):
                cos_ni = np.cos(n*theta[it])
                sin_ni = np.sin(n*theta[it])

                for ic, c in enumerate(comps):
                    B[len(comps)*it + ic, 4 + 8*(n-1) + 2*c] = cos_ni
                    B[len(comps)*it + ic, 4 + 8*(n-1) + 2*c+1] = sin_ni

        return B

    def K_rim_matl(self, r0=True):
        'Elastic portion of K_rim.'

        w = self.wheel

        R = w.rim.radius                        # rim radius
        EA = w.rim.young_mod * w.rim.area       # axial stiffness
        EI_rad = w.rim.young_mod * w.rim.I_rad  # radial bending
        EI_lat = w.rim.young_mod * w.rim.I_lat  # lateral bending
        EIw = w.rim.young_mod * w.rim.I_warp    # warping constant
        GJ = w.rim.shear_mod * w.rim.J_tor      # torsion constant

        y0 = 0.  # shear-center offset
        if 'y_0' in w.rim.sec_params:
            y0 = w.rim.sec_params['y_0']

        r02 = 0.
        if r0:
            r02 = EI_rad/EA + EI_lat/EA + y0**2

        K_rim_matl = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        # zero mode
        K_rim_matl[1, 1] = 2*pi*EA/R
        K_rim_matl[3, 3] = 2*pi*EI_lat/R

        # higher modes
        for n in range(1, self.n_modes + 1):
            i0 = 4 + (n-1)*8

            # k_vv
            K_rim_matl[i0+2, i0+2] = EI_rad*pi/R**3*n**4 + EA*pi/R*(1 + y0/R*n**2)**2
            K_rim_matl[i0+3, i0+3] = K_rim_matl[i0+2, i0+2]

            # k_ww
            K_rim_matl[i0+4, i0+4] = EI_rad*pi/R**3*n**2 + EA*pi/R*n**2*(1 + y0/R)**2
            K_rim_matl[i0+5, i0+5] = K_rim_matl[i0+4, i0+4]

            # k_vw
            K_rim_matl[i0+2, i0+5] = -EI_rad*pi/R**3*n**3 -\
                EA*pi*n/R*(1 + y0/R*(1 + n**2) + y0**2/R**2*n**2)
            K_rim_matl[i0+5, i0+2] = K_rim_matl[i0+2, i0+5]
            K_rim_matl[i0+3, i0+4] = -K_rim_matl[i0+2, i0+5]
            K_rim_matl[i0+4, i0+3] = -K_rim_matl[i0+2, i0+5]

            # k_uu
            K_rim_matl[i0+0, i0+0] = (EI_lat*pi/R**3*n**4 + EIw*pi/R**5*n**4 +
                                      GJ*pi/R**3*n**2)
            K_rim_matl[i0+1, i0+1] = K_rim_matl[i0+0, i0+0]

            # k_up
            K_rim_matl[i0+0, i0+6] = -(EI_lat*pi/R**2*n**2 + EIw*pi/R**4*n**4 +
                                  GJ*pi/R**2*n**2)
            K_rim_matl[i0+6, i0+0] = K_rim_matl[i0+0, i0+6]
            K_rim_matl[i0+1, i0+7] = K_rim_matl[i0+0, i0+6]
            K_rim_matl[i0+7, i0+1] = K_rim_matl[i0+0, i0+6]

            # k_pp
            K_rim_matl[i0+6, i0+6] = (EI_lat*pi/R + EIw*pi/R**3*n**4 + GJ*pi/R*n**2)
            K_rim_matl[i0+7, i0+7] = K_rim_matl[i0+6, i0+6]

        return K_rim_matl

    def K_rim_geom(self, r0=True):
        'Tension-dependent portion of K_rim, such that K_rim = K_rim_matl - T_avg*K_rim_geom'

        w = self.wheel

        R = w.rim.radius

        y0 = 0.  # shear-center offset
        if 'y_0' in w.rim.sec_params:
            y0 = w.rim.sec_params['y_0']

        r02 = 0.
        if r0:
            r02 = w.rim.I_rad/w.rim.area + w.rim.I_lat/w.rim.area + y0**2

        K_rim_geom = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        # zero mode
        K_rim_geom[3, 3] = -2.*pi*R*y0

        # higher modes
        for n in range(1, self.n_modes + 1):
            i0 = 4 + (n-1)*8

            # k_uu
            K_rim_geom[i0+0, i0+0] = pi*n**2*(1. + r02/R**2)
            K_rim_geom[i0+1, i0+1] = K_rim_geom[i0+0, i0+0]

            # k_up
            K_rim_geom[i0+0, i0+6] = pi*n**2*(y0 - r02/R)
            K_rim_geom[i0+6, i0+0] = K_rim_geom[i0+0, i0+6]
            K_rim_geom[i0+1, i0+7] = K_rim_geom[i0+0, i0+6]
            K_rim_geom[i0+7, i0+1] = K_rim_geom[i0+0, i0+6]

            # k_pp
            K_rim_geom[i0+6, i0+6] = -pi*(R*y0 - r02*n**2)
            K_rim_geom[i0+7, i0+7] = K_rim_geom[i0+6, i0+6]

        return len(w.spokes)/(2*pi*R) * K_rim_geom

    def K_spk_geom(self, smeared_spokes=False):
        'Tension-dependent portion of K_spk, such that K_spk = K_spk_matl + T_avg*K_spk_geom'

        K_spk = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        if smeared_spokes:  # Smith-Pippard approximation

            k_avg = 2*pi*self.wheel.rim.radius * self.wheel.calc_kbar_geom()

            K_spk[0:4, 0:4] = k_avg  # n = 0
            
            for n in range(1, self.n_modes+1):  # n >= 1
                K_spk[(4 + 8*(n-1)):(4 + 8*n):2,
                      (4 + 8*(n-1)):(4 + 8*n):2] = k_avg/2
                K_spk[(5 + 8*(n-1)):(5 + 8*n):2,
                      (5 + 8*(n-1)):(5 + 8*n):2] = k_avg/2

        else:  # Fully-discrete spokes

            # Get scaling factor for tension on each side of the wheel
            s_0 = self.wheel.spokes[0]
            s_1 = self.wheel.spokes[1]
            T_d = np.abs(s_0.n[0]*s_1.n[1]) + np.abs(s_1.n[0]*s_0.n[1])

            for s in self.wheel.spokes:
                B = self.B_theta(s.rim_pt[1])
                K_spk = K_spk + 2*np.abs(s.n[0])/T_d * B.T.dot(s.calc_k_geom().dot(B))

        return K_spk

    def K_rim(self, tension=True, r0=True):
        'Calculate rim strain energy stiffness matrix.'

        K_rim = self.K_rim_matl(r0=r0)
        if tension:
            T_avg = np.sum([s.tension*s.n[1]/len(self.wheel.spokes)
                            for s in self.wheel.spokes])
            K_rim = K_rim - T_avg*self.K_rim_geom(r0=r0)

        return K_rim

    def K_spk(self, smeared_spokes=False, tension=True):
        'Calculate spoke mode stiffness matrix.'

        K_spk = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        if smeared_spokes:  # Smith-Pippard approximation

            k_avg = 2*pi*self.wheel.rim.radius * self.wheel.calc_kbar(tension=tension)

            K_spk[0:4, 0:4] = k_avg  # n = 0
            
            for n in range(1, self.n_modes+1):  # n >= 1
                K_spk[(4 + 8*(n-1)):(4 + 8*n):2,
                      (4 + 8*(n-1)):(4 + 8*n):2] = k_avg/2
                K_spk[(5 + 8*(n-1)):(5 + 8*n):2,
                      (5 + 8*(n-1)):(5 + 8*n):2] = k_avg/2

        else:  # Fully-discrete spokes

            for s in self.wheel.spokes:
                B = self.B_theta(s.rim_pt[1])
                K_spk = K_spk + B.T.dot(s.calc_k(tension=tension).dot(B))

        return K_spk

    def F_ext(self, theta, f):
        'Calculate external force vector.'

        Bi = self.B_theta(theta).T
        F_ext = Bi.dot(np.array(f).reshape((4, 1)))

        return F_ext.flatten()

    def A_adj(self):
        'Calculate spoke adjustment matrix.'

        e3 = np.array([0., 0., 1.])  # rim axial vector

        A = np.zeros((4 + self.n_modes*8, len(self.wheel.spokes)))

        for i, s in enumerate(self.wheel.spokes):
            b = np.array([s.rim_pt[2], 0., 0.])
            A[:, i] = s.EA/s.length * self.B_theta(s.rim_pt[1]).T\
                .dot(np.append(s.n, e3.dot(np.cross(s.b, s.n))))

        return A


    def get_ix_uncoupled(self, dim='lateral'):
        'Get indices for either lateral/torsional or radial/tangential modes.'

        if dim == 'lateral':
            ix = np.sort([0, 3] +
                         list(range(4, 4 + 8*self.n_modes, 8)) +   # u_c
                         list(range(5, 4 + 8*self.n_modes, 8)) +   # u_s
                         list(range(10, 4 + 8*self.n_modes, 8)) +  # phi_c
                         list(range(11, 4 + 8*self.n_modes, 8)))   # phi_s
        else:
            ix = np.sort([1, 2] +
                         list(range(6, 4 + 8*self.n_modes, 8)) +   # v_c
                         list(range(7, 4 + 8*self.n_modes, 8)) +   # v_s
                         list(range(8, 4 + 8*self.n_modes, 8)) +   # w_c
                         list(range(9, 4 + 8*self.n_modes, 8)))    # w_s

        return ix

    def get_K_uncoupled(self, K, dim='lateral'):
        'Calculate stiffness matrix with radial/lateral coupling removed.'

        ix = self.get_ix_uncoupled(dim=dim)

        return K[np.ix_(ix, ix)]

    def rim_def_lat(self, theta, dm):
        'Calculate lateral rim deflection at the location(s) specified by theta'
        return self.B_theta(theta, 0).dot(dm)

    def rim_def_rad(self, theta, dm):
        'Calculate radial rim deflection at the location(s) specified by theta.'
        return self.B_theta(theta, 1).dot(dm)

    def rim_def_tan(self, theta, dm):
        'Calculate tangential rim deflection at the location(s) specified by theta.'
        return self.B_theta(theta, 2).dot(dm)

    def rim_def_rot(self, theta, dm):
        'Calculate rim cross-section rotation at the location(s) specified by theta.'
        return self.B_theta(theta, 3).dot(dm)

    def spoke_tension_change(self, dm, a=None):
        'Return a vector of tension changes for each spoke.'

        if a is None:
            a = np.zeros(self.n_spokes)

        dT = [s.calc_tension_change(self.B_theta(s.rim_pt[1]).dot(dm), adj)
              for s, adj in zip(self.wheel.spokes, a)]

        return np.array(dT)

    def __init__(self, wheel, N=10):

        self.wheel = wheel
        self.n_spokes = len(wheel.spokes)
        self.n_modes = N
