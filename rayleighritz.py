#!/usr/bin/env python

"""Theoretical calculations for bicycle wheels."""

import numpy as np
from bikewheelcalc import *


class RayleighRitzDiscrete:
    """Solve coupled lateral, radial, and torsional deflections with fully
    discrete spokes using the Rayleigh-Ritz method."""

    def B_theta(self, theta):
        'Matrix to transform mode coefficients to vector components.'

        B = np.zeros((3, 4 + 8*self.n_modes))

        B[0, 0] = 1
        B[1, 1] = 1
        B[2, 2] = 1

        for n in range(1, self.n_modes + 1):
            c = np.cos(n*theta)
            s = np.sin(n*theta)

            B[0, 3 + (n-1)*8 + 1] = c
            B[1, 3 + (n-1)*8 + 3] = c
            B[2, 3 + (n-1)*8 + 5] = c
            B[0, 3 + (n-1)*8 + 2] = s
            B[1, 3 + (n-1)*8 + 4] = s
            B[2, 3 + (n-1)*8 + 6] = s

        return B

    def calc_K_rim(self):
        'Calculate rim bending matrix.'

        pi = np.pi

        w = self.wheel

        EAr = w.rim.young_mod * w.rim.area  # axial stiffness
        EIr = w.rim.young_mod * w.rim.I33   # radial bending
        EIz = w.rim.young_mod * w.rim.I22   # lateral bending
        GJ = w.rim.shear_mod * w.rim.I11    # torsion

        # rim radius and cross-section radius of gyration
        R = w.rim.radius
        r = np.sqrt((w.rim.I22 + w.rim.I33) / w.rim.area)

        # Average compressive force in rim
        C = -self.n_spokes*self.tension / (2*pi)

        K_rim = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        # zero mode
        K_rim[1, 1] = 2*pi*EAr/R
        K_rim[3, 3] = 2*pi*EIz/R

        # higher modes
        for n in range(1, self.n_modes + 1):
            i0 = 3 + (n-1)*8

            # k_vv
            K_rim[i0+3, i0+3] = EIr*pi/R**3*n**4 + EAr*pi/R
            K_rim[i0+4, i0+4] = K_rim[i0+3, i0+3]

            # k_ww
            K_rim[i0+5, i0+5] = EIr*pi/R**3*n**2 + EAr*pi/R*n**2
            K_rim[i0+6, i0+6] = K_rim[i0+5, i0+5]

            # k_vw
            K_rim[i0+3, i0+6] = EIr*pi/R**3*n**3 + EAr*pi/R*n
            K_rim[i0+6, i0+3] = K_rim[i0+3, i0+6]
            K_rim[i0+4, i0+5] = -K_rim[i0+3, i0+6]
            K_rim[i0+5, i0+4] = -K_rim[i0+3, i0+6]

            # k_uu
            K_rim[i0+1, i0+1] = (EIz*pi/R**3*n**4 + GJ*pi/R**3*n**2 +
                                 pi*C/R*n**2)
            K_rim[i0+2, i0+2] = K_rim[i0+1, i0+1]

            # k_ub
            K_rim[i0+1, i0+7] = -(EIz*pi/R**2*n**2 + GJ*pi/R**2*n**2 -
                                  pi*C*(r/R)**2*n**2)
            K_rim[i0+2, i0+8] = K_rim[i0+1, i0+7]
            K_rim[i0+7, i0+1] = K_rim[i0+1, i0+7]
            K_rim[i0+8, i0+2] = K_rim[i0+1, i0+7]

            # k_bb
            K_rim[i0+7, i0+7] = EIz*pi/R + GJ*pi/R*n**2 + pi*C*r**2/R*n**2
            K_rim[i0+8, i0+8] = K_rim[i0+7, i0+7]

        return K_rim

    def calc_K_spk(self):
        'Calculate spoke elasticity matrix.'

        K_sp_el = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))
        K_sp_t = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        for s in self.wheel.spokes:
            Bi = self.B_theta(s.rim_pt[1])
            Ni = np.dot(s.n.reshape(1, 3), Bi)

            K_sp_el = K_sp_el + (s.EA / s.length) * (np.dot(Ni.T, Ni))
            K_sp_t = K_sp_t + (self.tension / s.length) * (np.dot(Bi.T, Bi))

        return K_sp_el, K_sp_t

    def calc_F_ext(self, f_theta, f):
        'Calculate external force vector.'

        F_ext = np.zeros(4 + self.n_modes*8).reshape(4 + self.n_modes*8, 1)

        for i in range(len(f_theta)):
            Bi = self.B_theta(f_theta[i]).T
            F_ext = F_ext + Bi.dot(f[i, :].reshape((3, 1)))

        return F_ext.flatten()

    def calc_F_adj(self, Omega):
        'Calculate spoke adjustment vector.'

        F_adj = np.zeros(4 + self.n_modes*8).reshape(4 + self.n_modes*8, 1)

        for s in self.wheel.spokes:
            BiT = self.B_theta(s.rim_pt[1]).T

            F_adj = F_adj + \
                s.EA/s.length * self.k_n * Omega[s] * np.dot(BiT, s.n)

        return F_adj.flatten()

    def calc_A_adj(self):
        'Calculate spoke adjustment matrix: F_adj = A_adj*Omega.'

        A_adj = np.zeros((4 + self.n_modes*8, self.n_spokes))

        for si in range(self.n_spokes):
            s = self.wheel.spokes[si]
            theta_s = s.rim_pt[1]
            BiT = self.B_theta(theta_s).T

            A_col = (s.EA/s.length) * self.k_n * np.dot(BiT, s.n)
            A_adj[:, si] = A_col.flatten()

        return A_adj

    def calc_lat_stiff(self):
        'Lateral stiffness under point load applied to rim.'

        K_rim = self.calc_K_rim()
        K_sp_el, K_sp_t = self.calc_K_spk()

        f_theta = [0.0]
        f_ext = np.array([[1.0, 0.0, 0.0]])
        F_ext = self.calc_F_ext(f_theta, f_ext)

        # Solve matrix equation (K_rim + K_sp_el)u_m = F_ext
        u_m = np.linalg.solve((K_rim + K_sp_el + K_sp_t), F_ext)

        u0 = u_m[0] + np.sum(u_m[4::8])

        return 1.0 / u0

    def calc_influence_fxn(self, Omega, comp=1, theta=None):
        'Calculate lateral deflection given spoke tightening, Omega.'

        if theta is None:
            # Positions of spoke nipples
            theta = self.geom.a_rim_nodes

        # Stiffness matrices
        K_rim = self.calc_K_rim()
        K_sp_el, K_sp_t = self.calc_K_spk()

        # Spoke adjustment vector
        F_adj = self.calc_F_adj(Omega)

        # Solve matrix equation (K_rim + K_sp_el)u = F_ext
        u_m = np.linalg.solve((K_rim + K_sp_el + K_sp_t), -F_adj)

        return self.get_displacement(u_m, comp, theta)

    def calc_tension_influence_fxn(self):
        'Calculate spoke tension influence matrix.'

        # Calculate stiffness matrix and adjustment matrix
        K_rim = self.calc_K_rim()
        K_sp_el, K_sp_t = self.calc_K_spk()
        K = K_rim + K_sp_el + K_sp_t
        A_adj = self.calc_A_adj()

        # Calculate spoke tension influence matrices
        Theta_t = np.zeros((self.n_spokes, self.n_spokes))
        Theta_d = np.zeros((self.n_spokes, 4 + self.n_modes*8))
        for si in range(self.n_spokes):
            s = self.wheel.spokes[si]
            theta_s = s.rim_pt[1]
            Bi = self.B_theta(theta_s)

            Theta_t[s, s] = self.k_n * s.EA/s.length
            Theta_d[s, :] = -s.EA/s.length * np.dot(s.n.T, Bi).flatten()

        return Theta_d.dot(np.linalg.inv(K).dot(A_adj)) + Theta_t

    def get_displacement(self, u_m, comp, theta):
        """Get the [comp] component of the rim displacement.
        (comp: 1=u, 2=v, 4=beta)"""

        i_0 = {1: 0, 2: 1, 3: 2, 4: 3}  # Lookup table for component indices
        i_m = {1: 0, 2: 2, 3: 6, 4: 6}
        d = u_m[i_0[comp]] * np.ones(len(theta))

        for m in range(1, self.n_modes + 1):
            d_c = u_m[3 + (m-1)*8 + i_m[comp] + 1]  # cos coefficient
            d_s = u_m[3 + (m-1)*8 + i_m[comp] + 2]  # sin coefficient

            d = d + d_c*np.cos(m*theta) + d_s*np.sin(m*theta)

        return d

    def calc_cond_K(self):
        'Calculate condition number of stiffness matrix'

        K_rim = self.calc_K_rim()
        K_sp_el, K_sp_t = self.calc_K_spk()

        return np.linalg.cond(K_rim + K_sp_el + K_sp_t)

    def est_T_crit(self, T_rng=None, N=100):
        'Estimate buckling tension by maximizing the condition number of K'

        if T_rng is None:
            # Get an initial estimate of Tcrit
            rar = RayleighRitz(self.geom, self.r_sec, self.s_sec)
            Tc_RRC, n = rar.min_buckling_mode(tens_stiff=True)

            T_test = Tc_RRC * np.logspace(-0.1, 0.1, N)
        else:
            T_test = np.linspace(T_rng[0], T_rng[1], N)

        cond_k = np.zeros(T_test.shape)

        for i in range(len(T_test)):
            self.tension = T_test[i]
            cond_k[i] = self.calc_cond_K()

        i_max = np.argmax(cond_k)

        return T_test[i_max]

    def __init__(self, wheel, N=10, tension=0.0):

        self.wheel = wheel
        self.n_spokes = len(wheel.spokes)
        self.n_modes = N
        self.tension = tension

        self.k_n = 1.0  # 0.0254 / 55.0  # distance per turn - spoke nipple


class RayleighRitzContinuum:
    'Solve lateral deflections using the Rayleigh-Ritz method (continuum).'

    def solve_euler(self):
        k_s = self.wheel.calc_continuum_stiff()
        k_uu = k_s[0, 0]
        k_ub = k_s[0, 3]
        k_bb = k_s[3, 3]

        # Axial force in the rim
        n_spokes = len(self.geom.lace_hub_n)
        N = -n_spokes*self.tension / (2*np.pi)

        cV = np.zeros(self.n, dtype=np.float)
        cB = np.zeros(self.n, dtype=np.float)
        cG = np.zeros(self.n, dtype=np.float)  # Not part of euler formulation

        EI = self.wheel.rim.young_mod * self.wheel.rim.r_sec.I22
        GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11

        # Rim radius and radius of gyration
        R = self.wheel.rim.radius
        r = np.sqrt((self.r_sec.I22 + self.r_sec.I33) / self.r_sec.area)

        pi = np.pi

        if self.tens_stiff:
            l = np.sqrt(((self.geom.d_rim - self.geom.d1_hub)/2)**2 +
                        self.geom.w1_hub**2)
            k_vv = k_vv + n_spokes/(2*pi*R) * (self.tension / l)

        # n=1 (zero-th mode)
        A0 = np.zeros((2, 2))

        A0[0, 0] = 2*pi*R*k_uu
        A0[0, 1] = 2*pi*R*k_ub
        A0[1, 0] = k_ub
        A0[1, 1] = k_bb + EI/R**2

        b0 = np.array([1, 0])

        x0 = np.linalg.solve(A0, b0)
        cV[0] = x0[0]
        cB[0] = x0[1]

        # n=1 -> higher
        for i in range(1, self.n):
            A = np.zeros((2, 2))

            A[0, 0] = ((EI*pi/R**3)*i**4 + (GJ*pi/R**3)*i**2 + pi*R*k_vv +
                       N*pi/R*i**2)
            A[0, 1] = -((EI*pi/R**2)*i**2 + (GJ*pi/R**2)*i**2 -
                        pi*R*k_vb - (N*pi*r**2/R)*i**2)

            A[1, 0] = -A[0, 1]
            A[1, 1] = -((EI*pi/R) + (GJ*pi/R)*i**2 + pi*R*k_bb +
                        N*pi*(r/R)**2*i**2)

            b = np.array([1, 0])
            x = np.linalg.solve(A, b)

            cV[i] = x[0]
            cB[i] = x[1]

        self.cV = cV
        self.cB = cB
        self.cG = cG  # trivially zero

    def calc_mode_stiff(self, n, tens_stiff=True):

        if tens_stiff:
            k_s = self.wheel.calc_continuum_stiff(tension=self.tension)
        else:
            k_s = self.wheel.calc_continuum_stiff(tension=0.0)

        k_uu = k_s[0, 0]
        k_ub = k_s[0, 3]
        k_bb = k_s[3, 3]

        # Axial force in the rim
        n_spokes = len(self.wheel.spokes)
        N = -n_spokes*self.tension / (2*np.pi)

        EI = self.wheel.rim.young_mod * self.wheel.rim.I22
        GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11

        # Rim radius and radius of gyration
        R = self.wheel.rim.radius
        r = np.sqrt((self.wheel.rim.I22 + self.wheel.rim.I33) /
                    self.wheel.rim.area)

        pi = np.pi

        A = np.zeros((2, 2))
        if n == 0:
            # zero-th mode
            A[0, 0] = 2*pi*R*k_uu
            A[0, 1] = 2*pi*R*k_ub
            A[1, 0] = k_ub
            A[1, 1] = k_bb + EI/R**2

            b0 = np.array([1, 0])
            x0 = np.linalg.solve(A, b0)

            cV = x0[0]
            # cB = x0[1]

        else:
            A = np.zeros((2, 2))

            A[0, 0] = ((EI*pi/R**3)*n**4 + (GJ*pi/R**3)*n**2 + pi*R*k_uu +
                       N*pi/R*n**2)
            A[0, 1] = -((EI*pi/R**2)*n**2 + (GJ*pi/R**2)*n**2 -
                        pi*R*k_ub - N*pi*(r/R)**2*n**2)

            A[1, 0] = -A[0, 1]
            A[1, 1] = -((EI*pi/R) + (GJ*pi/R)*n**2 + pi*R*k_bb +
                        N*pi*r**2/R*n**2)

            b = np.array([1, 0])
            x = np.linalg.solve(A, b)

            cV = x[0]
            # cB = x[1]

        return 1.0/cV

    def calc_buckling_tension(self, n, tens_stiff=True, stiff_factor=1.0):

        k_s = self.wheel.calc_continuum_stiff(tension=0.0)
        k_uu = k_s[0, 0] * stiff_factor
        k_ub = k_s[0, 3] * stiff_factor
        k_bb = k_s[3, 3] * stiff_factor

        # shortcuts
        n_s = len(self.wheel.spokes)
        R = self.wheel.rim.radius
        l = self.wheel.spokes[0].length
        EI = self.wheel.rim.young_mod * self.wheel.rim.I22
        EIw = self.wheel.rim.young_mod * self.wheel.rim.Iw
        GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11
        mu = (GJ + EIw*n**2/R**2) / EI

        A = 1 + mu*n**2 + k_bb*R**2/EI
        B = n**4 + mu*n**2
        C = 2*n**2*(1 + mu) - k_ub*R**3/EI
        D = mu*n**2*(n**2 - 1)**2

        if tens_stiff:
            f_T = n**2 / (n**2 - R/l)
        else:
            f_T = 1.0

        T_c = 2*np.pi*EI/(n_s*R**2*n**2*A) * f_T *\
            (A*k_uu*R**4/EI + B*k_bb*R**2/EI + C*k_ub*R**3/EI + D)

        return T_c

    def calc_buckling_tension_quad(self, n):
        'Calculate buckling tension from full quadratic form.'

        k_s = self.wheel.calc_continuum_stiff(tension=0.0)
        k_uu = k_s[0, 0]
        k_ub = k_s[0, 3]
        k_bb = k_s[3, 3]

        # shortcuts
        pi = np.pi
        ns = len(self.wheel.spokes)
        R = self.wheel.rim.radius
        l = self.wheel.spokes[0].length
        EI = self.wheel.rim.young_mod * self.wheel.rim.I22
        EIw = self.wheel.rim.young_mod * self.wheel.rim.Iw
        GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11
        CT = GJ + EIw*n**2/R**2

        rx = np.sqrt(self.wheel.rim.I22 / self.wheel.rim.area)
        ry = np.sqrt(self.wheel.rim.I33 / self.wheel.rim.area)

        if 'y_s' in self.wheel.rim.sec_params:
            y0 = self.wheel.rim.sec_params['y_c'] -\
                self.wheel.rim.sec_params['y_s']
        else:
            y0 = 0.0

        kT = float(ns) / (2*np.pi*R*l)

        A = -2*kT*n**2*ns*pi*rx**2 + (n**4*ns**2*rx**2)/R**2 -\
            2*kT*n**2*ns*pi*ry**2 + (n**4*ns**2*ry**2)/R**2 +\
            (n**4*ns**2*rx**2*ry**2)/R**4 -\
            (n**2*ns**2*y0)/R + 2*kT*ns*pi*R*y0 - (n**2*ns**2*ry**2*y0)/R**3 +\
            (2*n**4*ns**2*ry**2*y0)/R**3 - 2*kT*n**2*ns*pi*y0**2 +\
            (n**4*ns**2*ry**2*y0**2)/R**4

        B = -2*k_bb*n**2*ns*pi + 4*EI*kT*pi**2 + 4*CT*kT*n**2*pi**2 -\
            (2*EI*n**2*ns*pi)/R**2 - (2*CT*n**4*ns*pi)/R**2 +\
            4*kT*k_bb*pi**2*R**2 - 2*k_uu*n**2*ns*pi*rx**2 -\
            (2*CT*n**4*ns*pi*rx**2)/R**4 - (2*EI*n**6*ns*pi*rx**2)/R**4 -\
            2*k_uu*n**2*ns*pi*ry**2 - (2*EI*n**2*ns*pi*ry**2)/R**4 +\
            (4*EI*n**4*ns*pi*ry**2)/R**4 - (2*EI*n**6*ns*pi*ry**2)/R**4 -\
            (2*k_bb*n**2*ns*pi*ry**2)/R**2 - (4*k_ub*n**2*ns*pi*ry**2)/R +\
            4*k_ub*n**2*ns*pi*y0 + (2*CT*n**2*ns*pi*y0)/R**3 -\
            (2*EI*n**4*ns*pi*y0)/R**3 - (4*CT*n**4*ns*pi*y0)/R**3 +\
            2*k_uu*ns*pi*R*y0 - 2*k_uu*n**2*ns*pi*y0**2 -\
            (2*CT*n**4*ns*pi*y0**2)/R**4 - (2*EI*n**6*ns*pi*y0**2)/R**4

        C = -(-((2*EI*n**2*pi)/R**2) - (2*CT*n**2*pi)/R**2 + 2*k_ub*pi*R)**2 +\
            ((2*CT*n**2*pi)/R**3 + (2*EI*n**4*pi)/R**3 + 2*k_uu*pi*R) *\
            ((2*EI*pi)/R + (2*CT*n**2*pi)/R + 2*k_bb*pi*R)

        # Solve for the smaller root
        T_c = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)

        return T_c

    def Tc_small_mu(self):
        'Approximate formula for torsion-dominated buckling'

        k_s = self.wheel.calc_continuum_stiff(tension=0.0)
        k_uu = k_s[0, 0]

        GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11
        R = self.wheel.rim.radius
        n_s = len(self.wheel.spokes)

        Tc = 11.875 * GJ/(n_s*R**2) * np.power(k_uu*R**4/GJ, 2.0/3.0)

        return Tc

    def Tc_euler(self):
        'Buckling tension coefficient.'

        return 2*np.pi*self.wheel.rim.young_mod*self.wheel.rim.I22 /\
            (len(self.wheel.spokes) * self.wheel.rim.radius**2)

    def min_buckling_mode(self, tens_stiff=True, stiff_factor=1.0, quad=False):
        # Check first 20 modes, starting with n=2

        if quad:
            T_n = [self.calc_buckling_tension_quad(n)
                   for n in range(2, 21)]
        else:
            T_n = (self.calc_buckling_tension(np.arange(2, 21),
                                              tens_stiff=tens_stiff,
                                              stiff_factor=stiff_factor)).tolist()

        T_min = min(T_n)
        n_min = T_n.index(min(T_n)) + 2

        return T_min, n_min

    def calc_stiff(self, N=10):

        f_n = [1.0/self.calc_mode_stiff(i) for i in range(N)]
        return 1.0/sum(f_n)

    def deflection(self, theta):
        if type(theta) == np.ndarray:
            n = len(theta)
        else:
            n = 1

        v = np.zeros(n)
        b = np.zeros(n)
        g = np.zeros(n)

        for i in range(len(self.cV)):
            v = v + self.cV[i]*np.cos(i*theta)
            b = b + self.cB[i]*np.cos(i*theta)
            g = g + self.cG[i]*np.sin(i*theta)

        return v, b, g

    def get_len_scale(self):

        v0, b0, g0 = self.deflection(0)

        def y(x):
            v, b, g = self.deflection(x)

            return v[0]/v0 - 0.5

        x_min = 0.0
        x_max = np.pi/2

        while np.abs(x_max - x_min) > 0.0001:
            xt = 0.5 * (x_max + x_min)

            if y(xt)*y(x_min) > 0:
                x_min = xt
            else:
                x_max = xt

        return xt

    def __init__(self, wheel, N=10, tension=0.0):

        self.wheel = wheel
        self.tension = tension
        self.N = N  # number of terms in approximation
