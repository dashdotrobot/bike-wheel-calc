'Rayleigh-Ritz solution to fully-coupled uvw-phi equations.'

import numpy as np
from numpy import pi


def xp(v):
    'Calculate cross-product matrix such that xp*u = v x u'

    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])

class ModeMatrix:
    """Solve coupled lateral, radial, and torsional deflections."""

    def B_theta(self, theta=[0.], comps=[0, 1, 2, 3], deriv=0):
        'Modal shape function and its derivatives'

        theta = np.atleast_1d(theta)
        comps = np.atleast_1d(comps)

        fxns = [np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x), np.sin]

        X = np.zeros((len(theta)*len(comps), 4 + 8*self.n_modes))

        # For each angle theta
        for it in range(len(theta)):

            # Zero mode
            for ic, c in enumerate(comps):
                X[len(comps)*it + ic, c] = 1*(deriv == 0)

            # Higher modes
            for n in range(1, self.n_modes + 1):
                f1_ni = n**deriv * fxns[deriv%4](n*theta[it])
                f2_ni = n**deriv * fxns[(deriv-1)%4](n*theta[it])

                for ic, c in enumerate(comps):
                    X[len(comps)*it + ic, 4 + 8*(n-1) + 2*c] = f1_ni
                    X[len(comps)*it + ic, 4 + 8*(n-1) + 2*c+1] = f2_ni

        return X

    def K_rim_matl(self, r0=True):
        'Elastic portion of K_rim.'

        w = self.wheel

        y0 = 0.  # shear-center offset
        if 'y_0' in w.rim.sec_params:
            y0 = w.rim.sec_params['y_0']

        R = w.rim.radius  # rim radius at shear center
        Rc = R + y0       # rim radius at centroid
        f = R/Rc          # correction factor for section properties

        EA = f*w.rim.young_mod * w.rim.area       # axial stiffness
        EI_rad = f*w.rim.young_mod * w.rim.I_rad  # radial bending
        EI_lat = f*w.rim.young_mod * w.rim.I_lat  # lateral bending
        EIw = f*w.rim.young_mod * w.rim.I_warp    # warping constant
        GJ = (1/f)*w.rim.shear_mod * w.rim.J_tor  # torsion constant

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
                B = self.B_theta(s.theta)
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
                B = self.B_theta(s.theta)
                K_spk = K_spk + B.T.dot(s.calc_k(tension=tension).dot(B))

        return K_spk

    def K_spk_new(self, tension=True, grad=True, wrot=True):
        'Correct mode stiffness formulation, with gradients of u'

        K_spk = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        X_w = np.array([[0., 0., -wrot*1./self.wheel.rim.radius, 0.],
                        [0., 0., 0., 0.], [0., 0., 0., 1.]])
        Xpw = grad*np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.]])

        for s in self.wheel.spokes:
            Bu = self.B_theta(s.theta, comps=[0, 1, 2])
            Bw = X_w.dot(self.B_theta(s.theta)) + Xpw.dot(self.B_theta(s.theta, deriv=1))

            kf = s.calc_kf_n(tension=tension)
            nx = xp(s.n)
            bx = xp(s.b)
            T0 = s.tension if tension else 0.

            K_spk = (K_spk
                     + Bu.T.dot(kf).dot(Bu)                  # Force-stiffness
                     + Bu.T.dot(kf).dot(bx).dot(Bw)          #  - bs correction u-phi
                     + Bw.T.dot(bx.T).dot(kf).dot(Bu)        #  - bs correction u-phi
                     + Bw.T.dot(bx).dot(kf).dot(bx).dot(Bw)  #  - bs correction phi-phi
                     + T0/2*Bw.T.dot(nx).dot(Bu)             # Moment stiffness
                     + T0/2*Bu.T.dot(nx.T).dot(Bw)           #  - Transpose of above
                     + T0/2*Bw.T.dot(nx).dot(bx).dot(Bw)     #  - Spoke offset correction
                     + T0/2*Bw.T.dot(bx.T).dot(nx.T).dot(Bw))  #  - Transpose of above

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
            A[:, i] = s.EA/s.length * self.B_theta(s.theta).T\
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

        dT = [s.calc_tension_change(self.B_theta(s.theta).dot(dm), adj)
              for s, adj in zip(self.wheel.spokes, a)]

        return np.array(dT)

    def moment_rad(self, theta, dm):
        'Calculate radial bending moment at the location(s) specified by theta.'

        R = self.wheel.rim.radius
        f = (R/(R + self.wheel.rim.sec_params['y_0'])
             if 'y_0' in self.wheel.rim.sec_params else 1.)
        EI = f*self.wheel.rim.young_mod*self.wheel.rim.I_rad

        return EI/R**2*(self.B_theta(theta, comps=1, deriv=2) +
                        self.B_theta(theta, comps=2, deriv=1)).dot(dm)

    def moment_lat(self, theta, dm):
        'Calculate lateral bending moment at the location(s) specified by theta.'

        R = self.wheel.rim.radius
        f = (R/(R + self.wheel.rim.sec_params['y_0'])
             if 'y_0' in self.wheel.rim.sec_params else 1.)
        EI = f*self.wheel.rim.young_mod*self.wheel.rim.I_lat

        return EI/R**2*(self.B_theta(theta, comps=0, deriv=2) +
                        R*self.B_theta(theta, comps=3)).dot(dm)

    def moment_tor(self, theta, dm):
        'Calculate twisting moment at the location(s) specified by theta.'

        R = self.wheel.rim.radius
        f = (R/(R + self.wheel.rim.sec_params['y_0'])
             if 'y_0' in self.wheel.rim.sec_params else 1.)
        GJ = (1/f)*self.wheel.rim.shear_mod*self.wheel.rim.J_tor
        EIw = f*self.wheel.rim.young_mod*self.wheel.rim.I_warp

        # Saint-Venant torsion (active)
        M_GJ = GJ/R**2*(R*self.B_theta(theta, comps=3, deriv=1) -
                        self.B_theta(theta, comps=0, deriv=1)).dot(dm)

        # Warping torsion (reactive)
        M_W = EIw/R**4*(R*self.B_theta(theta, comps=3, deriv=3) -
                        self.B_theta(theta, comps=0, deriv=3)).dot(dm)

        return M_GJ + M_W

    def normal_force(self, theta, dm):
        'Calculate axial force in the rim at the location(s) specified by theta.'

        R = self.wheel.rim.radius
        y0 = (self.wheel.rim.sec_params['y_0']
              if 'y_0' in self.wheel.rim.sec_params else 0.)
        EA = R/(R+y0)*self.wheel.rim.young_mod*self.wheel.rim.area

        return EA/R*(self.B_theta(theta, comps=2, deriv=1) -
                     self.B_theta(theta, comps=1) +
                     y0*self.B_theta(theta, comps=1, deriv=2)/R +
                     y0*self.B_theta(theta, comps=2, deriv=1)/R).dot(dm)

    def shear_force_rad(self, theta, dm):
        'Calculate in-plane shear in the rim at the location(s) specified by theta.'

        R = self.wheel.rim.radius
        f = (R/(R + self.wheel.rim.sec_params['y_0'])
             if 'y_0' in self.wheel.rim.sec_params else 1.)
        EI = f*self.wheel.rim.young_mod*self.wheel.rim.I_rad

        return -EI/R**3*(self.B_theta(theta, comps=1, deriv=3) +
                         self.B_theta(theta, comps=2, deriv=2)).dot(dm)

    def shear_force_lat(self, theta, dm):
        'Calculate out-of-plane shear in the rim at the location(s) specified.'
        
        R = self.wheel.rim.radius
        f = (R/(R + self.wheel.rim.sec_params['y_0'])
             if 'y_0' in self.wheel.rim.sec_params else 1.)
        EI = f*self.wheel.rim.young_mod*self.wheel.rim.I_lat

        return -(EI/R**3*(self.B_theta(theta, comps=0, deriv=3) +
                          R*self.B_theta(theta, comps=3, deriv=1)).dot(dm)
                 + self.moment_tor(theta, dm)/R)

    def __init__(self, wheel, N=10):

        self.wheel = wheel
        self.n_spokes = len(wheel.spokes)
        self.n_modes = N
