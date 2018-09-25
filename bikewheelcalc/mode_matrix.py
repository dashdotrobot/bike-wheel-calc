#!/usr/bin/env python

"""Rayleigh-Ritz solution to fully-coupled uvw-phi equations."""

import numpy as np
from scipy.optimize import minimize
from .continuum_analysis import calc_buckling_tension, calc_continuum_stiff


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

    def K_rim(self, buckling=False):
        'Calculate rim strain energy stiffness matrix.'

        pi = np.pi

        w = self.wheel
        R = w.rim.radius                   # rim radius
        y0 = 0.                            # shear-center offset
        EA = w.rim.young_mod * w.rim.area  # axial stiffness
        EI1 = w.rim.young_mod * w.rim.I33  # radial bending
        EI2 = w.rim.young_mod * w.rim.I22  # lateral bending
        EIw = w.rim.young_mod * w.rim.Iw   # wapring constant
        GJ = w.rim.shear_mod * w.rim.I11   # torsion constant

        ry = np.sqrt(EI1/EA)
        rx = np.sqrt(EI2/EA)
        r2 = rx**2 + ry**2 + y0**2

        # Average compressive force in rim
        # TODO Accurately calculate for arbitrary geometry
        # C = -self.n_spokes*self.tension / (2*pi)
        if buckling:
            Tbar = np.sum([s.tension*s.n[1] for s in w.spokes])/(2*pi*R)
        else:
            Tbar = 0.

        K_rim = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        # zero mode
        K_rim[1, 1] = 2*pi*EA/R
        K_rim[3, 3] = 2*pi*EI2/R + 2*pi*R*y0*Tbar

        # higher modes
        for n in range(1, self.n_modes + 1):
            i0 = 4 + (n-1)*8

            # k_vv
            K_rim[i0+2, i0+2] = (EI1*pi/R**3*n**4 + EA*pi/R +
                                 2*EA*pi/R**2*y0*n**2 + EA*pi/R**3*y0**2*n**4)
            K_rim[i0+3, i0+3] = K_rim[i0+2, i0+2]

            # k_ww
            K_rim[i0+4, i0+4] = EI1*pi/R**3*n**2 +\
                EA*pi*(n**2/R + 2*y0*n**2/R**2 + y0**2*n**2/R**3)
            K_rim[i0+5, i0+5] = K_rim[i0+4, i0+4]

            # k_vw
            K_rim[i0+2, i0+5] = -EI1*pi/R**3*n**3 -\
                EA*pi*(n/R + n**3*y0/R**2 + n*y0/R**2 + n**3*y0**2/R**3)
            K_rim[i0+5, i0+2] = K_rim[i0+2, i0+5]
            K_rim[i0+3, i0+4] = -K_rim[i0+2, i0+5]
            K_rim[i0+4, i0+3] = -K_rim[i0+2, i0+5]

            # k_uu
            K_rim[i0+0, i0+0] = (EI2*pi/R**3*n**4 + EIw*pi/R**5*n**4 +
                                 GJ*pi/R**3*n**2 -
                                 Tbar*pi*n**2*(1. + ry**2/R**2))
            K_rim[i0+1, i0+1] = K_rim[i0+0, i0+0]

            # k_ub
            K_rim[i0+0, i0+6] = -(EI2*pi/R**2*n**2 + EIw*pi/R**4*n**4 +
                                  GJ*pi/R**2*n**2 +
                                  Tbar*pi*n**2*(y0 + ry**2/R))
            K_rim[i0+6, i0+0] = K_rim[i0+0, i0+6]
            K_rim[i0+1, i0+7] = K_rim[i0+0, i0+6]
            K_rim[i0+7, i0+1] = K_rim[i0+0, i0+6]

            # k_bb
            K_rim[i0+6, i0+6] = EI2*pi/R + EIw*pi/R**3*n**4 + GJ*pi/R*n**2 -\
                Tbar*pi*(n**2*r2 - R*y0)
            K_rim[i0+7, i0+7] = K_rim[i0+6, i0+6]

        return K_rim

    def K_spk(self, smeared_spokes=True):
        'Calculate spoke mode stiffness matrix.'

        K_spk = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        if smeared_spokes:  # Smith-Pippard approximation

            k_avg = np.zeros(4)
            for s in self.wheel.spokes:
                k_avg = k_avg + s.calc_k()

            # n = 0
            K_spk[0:4, 0:4] = k_avg

            # n >= 1
            for n in range(1, self.n_modes+1):
                K_spk[(4 + 8*(n-1)):(4 + 8*n):2,
                      (4 + 8*(n-1)):(4 + 8*n):2] = k_avg/2
                K_spk[(5 + 8*(n-1)):(5 + 8*n):2,
                      (5 + 8*(n-1)):(5 + 8*n):2] = k_avg/2

            return K_spk

        else:  # Fully-discrete spokes

            for s in self.wheel.spokes:
                B = self.B_theta(s.rim_pt[1])
                K_spk = K_spk + B.T.dot(s.calc_k().dot(B))

            return K_spk

    def F_ext(self, f_theta, f):
        'Calculate external force vector.'

        F_ext = np.zeros(4 + self.n_modes*8).reshape(4 + self.n_modes*8, 1)

        for i in range(len(f_theta)):
            Bi = self.B_theta(f_theta[i]).T
            F_ext = F_ext + Bi.dot(f[i, :].reshape((4, 1)))

        return F_ext.flatten()

    def lat_mode_stiff(self, n, smeared_spokes=True, buckling=True):
        'Calculate lateral mode stiffness'

        k_s = calc_continuum_stiff(self.wheel)
        k_uu = k_s[0, 0]
        k_ub = k_s[0, 3]
        k_bb = k_s[3, 3]

        # shortcuts
        pi = np.pi
        ns = len(self.wheel.spokes)
        R = self.wheel.rim.radius
        EI = self.wheel.rim.young_mod * self.wheel.rim.I22
        EIw = self.wheel.rim.young_mod * self.wheel.rim.Iw
        GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11

        rx = np.sqrt(self.wheel.rim.I22 / self.wheel.rim.area)
        ry = np.sqrt(self.wheel.rim.I33 / self.wheel.rim.area)

        CT = GJ + EIw*n**2/R**2

        # Shear center coordinate
        if 'y_s' in self.wheel.rim.sec_params:
            y0 = self.wheel.rim.sec_params['y_c'] -\
                self.wheel.rim.sec_params['y_s']
        else:
            y0 = 0.0

        Nr = np.sum([s.tension*s.n[1] for s in self.wheel.spokes]) / (2*pi)

        if n == 0:
            U_uu = 2*pi*R*k_uu
            U_ub = 2*pi*R*k_ub
            U_bb = 2*pi*EI/R + 2*pi*R*k_bb + 2*pi*Nr*y0
        else:  # n > 0
            U_uu = pi*EI*n**4/R**3 + pi*CT*n**2/R**3 + pi*R*k_uu \
                - pi*Nr*n**2/R - pi*Nr*n**2*ry**2/R**3

            U_ub = -pi*EI*n**2/R**2 - pi*CT*n**2/R**2 + pi*R*k_ub \
                - pi*Nr*n**2*ry**2/R**2 - pi*Nr*n**2*y0/R

            U_bb = pi*EI/R + pi*CT*n**2/R + pi*R*k_bb\
                + pi*Nr*y0 - pi*Nr*n**2*(rx**2 + ry**2 + y0**2)/R

        # Solve linear system
        K = np.zeros((2, 2))
        K[0, 0] = U_uu
        K[0, 1] = U_ub
        K[1, 0] = U_ub
        K[1, 1] = U_bb

        x = np.linalg.solve(K, np.array([1, 0]))

        # Displacement stiffness
        Kn_u = 1.0 / x[0]

        # Rotation stiffness
        if x[1] == 0.0:
            Kn_p = float('inf')
        else:
            Kn_p = 1.0 / x[1]

        return Kn_u

    def calc_lat_stiff(self, smeared_spokes=True, buckling=False, coupling=False):
        'Calculate lateral stiffness.'

        F_ext = self.F_ext([0.], np.array([[1., 0., 0., 0.]]))
        d = np.zeros(F_ext.shape)

        if coupling:
            K = self.K_rim(buckling=buckling) +\
                self.K_spk(smeared_spokes=smeared_spokes)
            d = np.linalg.solve(K, F_ext)
        else:
            ix_uc = self.get_ix_uncoupled(dim='lateral')
            F_ext = F_ext[ix_uc]
            K = self.get_K_uncoupled(dim='lateral',
                                      smeared_spokes=smeared_spokes,
                                      buckling=buckling)
            d_uc = np.linalg.solve(K, F_ext)
            d[ix_uc] = d_uc

        return 1.0 / self.B_theta(0.).dot(d)[0]

    def calc_rad_stiff(self, smeared_spokes=True, buckling=False, coupling=False):
        'Calculate radial stiffness.'

        F_ext = self.F_ext([0.], np.array([[0., 1., 0., 0.]]))
        d = np.zeros(F_ext.shape)

        if coupling:
            K = self.K_rim(buckling=buckling) +\
                self.K_spk(smeared_spokes=smeared_spokes)
            d = np.linalg.solve(K, F_ext)
        else:
            ix_uc = self.get_ix_uncoupled(dim='radial')
            F_ext = F_ext[ix_uc]
            K = self.get_K_uncoupled(dim='radial',
                                     smeared_spokes=smeared_spokes,
                                     buckling=buckling)
            d_uc = np.linalg.solve(K, F_ext)
            d[ix_uc] = d_uc

        return 1.0 / self.B_theta(0.).dot(d)[1]

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

    def get_K_uncoupled(self, K=None, dim='lateral',
                         smeared_spokes=True, buckling=False):
        'Calculate stiffness matrix with radial/lateral coupling removed.'

        # Calculate stiffness matrix, if not already supplied
        if K is None:
            K = (self.K_spk(smeared_spokes=smeared_spokes) +
                 self.K_rim(buckling=buckling))

        ix = self.get_ix_uncoupled(dim=dim)

        return K[np.ix_(ix, ix)]

    def buckling_tension(self, smeared_spokes=True, coupling=True):
        'Estimate buckling tension from condition number of stiffness matrix.'

        def neg_cond(T):
            self.wheel.apply_tension(T)

            if coupling:
                K = (self.K_rim(buckling=True) +
                     self.K_spk(smeared_spokes=smeared_spokes))

            else:
                K = self.get_K_uncoupled(buckling=True,
                                          smeared_spokes=smeared_spokes)

            return -np.linalg.cond(K)

        # Find approximate buckling tension from analytical solution
        Tc_approx = calc_buckling_tension(self.wheel)[0]

        # Maximize the condition number as a function of tension
        res = minimize(fun=neg_cond, x0=[Tc_approx], method='Nelder-Mead',
                       options={'maxiter': 50})

        return res.x[0]



    def __init__(self, wheel, N=10, tension=0.0):

        self.wheel = wheel
        self.n_spokes = len(wheel.spokes)
        self.n_modes = N

        self.tension = tension
