#!/usr/bin/env python

"""Rayleigh-Ritz solution to fully-coupled uvw-phi equations."""

import numpy as np
from scipy.optimize import minimize


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
        EIw = w.rim.young_mod * w.rim.Iw   # warping constant
        GJ = w.rim.shear_mod * w.rim.I11   # torsion constant

        ry = np.sqrt(EI1/EA)
        rx = np.sqrt(EI2/EA)
        r2 = rx**2 + ry**2 + y0**2

        # Average net radial tension per unit length
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

    def K_spk(self, smeared_spokes=True, tension=True):
        'Calculate spoke mode stiffness matrix.'

        K_spk = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        if smeared_spokes:  # Smith-Pippard approximation

            k_avg = np.zeros(4)
            for s in self.wheel.spokes:
                k_avg = k_avg + s.calc_k(tension=tension)

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
                K_spk = K_spk + B.T.dot(s.calc_k(tension=tension).dot(B))

            return K_spk

    def F_ext(self, f_theta, f):
        'Calculate external force vector.'

        F_ext = np.zeros(4 + self.n_modes*8).reshape(4 + self.n_modes*8, 1)

        for i in range(len(f_theta)):
            Bi = self.B_theta(f_theta[i]).T
            F_ext = F_ext + Bi.dot(f[i, :].reshape((4, 1)))

        return F_ext.flatten()

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
                         smeared_spokes=True, buckling=True):
        'Calculate stiffness matrix with radial/lateral coupling removed.'

        # Calculate stiffness matrix, if not already supplied
        if K is None:
            K = (self.K_spk(smeared_spokes=smeared_spokes) +
                 self.K_rim(buckling=buckling))

        ix = self.get_ix_uncoupled(dim=dim)

        return K[np.ix_(ix, ix)]

    def __init__(self, wheel, N=10):

        self.wheel = wheel
        self.n_spokes = len(wheel.spokes)
        self.n_modes = N
