#!/usr/bin/env python

"""Theoretical calculations for bicycle wheels."""

import numpy as np


def calc_spoke_stiff(g, r_sec, s_sec, T_avg=0.0):
    'Calculate spoke system stiffness'

    offset = g.lace_offset[0]  # offset for one spoke (assume all the same)

    # bracing angle
    a_1 = np.arctan(2*(g.w1_hub - offset) /
                    (g.d_rim - g.d1_hub))
    a_2 = np.arctan(2*(g.w2_hub - offset) /
                    (g.d_rim - g.d2_hub))

    # spoke length
    l_1 = np.sqrt(((g.d_rim - g.d1_hub)/2)**2 +
                  (g.w1_hub - offset)**2)
    l_2 = np.sqrt(((g.d_rim - g.d2_hub)/2)**2 +
                  (g.w1_hub - offset)**2)

    # Assume same number of left and right spokes
    n_1 = len(g.lace_hub_n)/2
    n_2 = len(g.lace_hub_n)/2

    k_uu = s_sec.young_mod*s_sec.area/(np.pi*g.d_rim) * \
        (n_1*np.sin(a_1)**2/l_1 + n_2*np.sin(a_2)**2/l_2)

    k_bb = (offset)**2 * s_sec.young_mod*s_sec.area/(np.pi*g.d_rim) *\
        (n_1*np.cos(a_1)**2/l_1 + n_2*np.cos(a_2)**2/l_2)

    k_ub = offset * s_sec.young_mod*s_sec.area/(np.pi*g.d_rim) *\
        (n_1*np.cos(a_1)*np.sin(a_1)/l_1 + n_2*np.cos(a_2)*np.sin(a_2)/l_2)

    k_rr = s_sec.young_mod*s_sec.area/(np.pi*g.d_rim) * \
        (n_1*np.cos(a_1)**2/l_1 + n_2*np.cos(a_2)**2/l_2)

    return k_uu, k_bb, k_ub, k_rr


def calc_spoke_len(g, offset=0.0):
    l_1 = np.sqrt(((g.d_rim - g.d1_hub)/2)**2 +
                  (g.w1_hub - offset)**2)
    l_2 = np.sqrt(((g.d_rim - g.d2_hub)/2)**2 +
                  (g.w1_hub - offset)**2)
    return (l_1 + l_2)/2


def calc_Pn_lat(geom, r_sec, s_sec):
    'Lateral Pippard number'
    k_vv, k_bb, k_vb, k_rr = calc_spoke_stiff(geom, r_sec, s_sec)

    n_spokes = len(geom.lace_hub_n)

    ei = r_sec.young_mod * r_sec.I22
    gj = r_sec.shear_mod * r_sec.I11
    cc = (gj/ei)/(gj/ei + 1)

    lp = n_spokes/(np.pi*geom.d_rim) * np.power(4*ei / k_vv, 0.25) * cc

    return lp


def calc_Pn_rad(geom, r_sec, s_sec):
    'Radial Pippard number'
    k_vv, k_bb, k_vb, k_rr = calc_spoke_stiff(geom, r_sec, s_sec)

    n_spokes = len(geom.lace_hub_n)

    ei = r_sec.young_mod * r_sec.I33

    lp = n_spokes/(np.pi*geom.d_rim) * np.power(4*ei / k_rr, 0.25)

    return lp


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

    def calc_spoke_n(self, s):
        'Calculate normal vector along spoke.'

        a_hub = self.geom.a_hub_nodes[s] - self.geom.a_rim_nodes[s]

        side = self.geom.s_hub_nodes[s]
        if side == 1:
            d_hub = self.geom.d1_hub
            w_hub = self.geom.w1_hub
        else:
            d_hub = self.geom.d2_hub
            w_hub = -self.geom.w2_hub

        u = w_hub
        v = self.geom.d_rim/2 - (d_hub/2)*np.cos(a_hub)
        w = (d_hub/2)*np.sin(a_hub)

        ls = np.sqrt(u**2 + v**2 + w**2)

        return np.array([u/ls, v/ls, w/ls]).reshape((3, 1)), ls

    def calc_K_rim(self):
        'Calculate rim bending matrix.'

        Pi = np.pi

        EAr = self.r_sec.young_mod * self.r_sec.area  # axial stiffness
        EIr = self.r_sec.young_mod * self.r_sec.I33   # radial bending
        EIz = self.r_sec.young_mod * self.r_sec.I22   # lateral bending
        GJ = self.r_sec.shear_mod * self.r_sec.I11    # torsion

        # rim radius and cross-section radius of gyration
        R = self.geom.d_rim / 2
        r = np.sqrt((self.r_sec.I22 + self.r_sec.I33)/self.r_sec.area)

        # Average compressive force in rim
        C = -self.n_spokes*self.tension / (2*Pi)

        K_rim = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        # zero mode
        K_rim[1, 1] = 2*Pi*EAr/R
        K_rim[3, 3] = 2*Pi*EIz/R

        # higher modes
        for n in range(1, self.n_modes + 1):
            i0 = 3 + (n-1)*8

            # k_vv
            K_rim[i0+3, i0+3] = EIr*Pi/R**3*n**4 + EAr*Pi/R
            K_rim[i0+4, i0+4] = K_rim[i0+3, i0+3]

            # k_ww
            K_rim[i0+5, i0+5] = EIr*Pi/R**3*n**2 + EAr*Pi/R*n**2
            K_rim[i0+6, i0+6] = K_rim[i0+5, i0+5]

            # k_vw
            K_rim[i0+3, i0+6] = EIr*Pi/R**3*n**3 + EAr*Pi/R*n
            K_rim[i0+6, i0+3] = K_rim[i0+3, i0+6]
            K_rim[i0+4, i0+5] = -K_rim[i0+3, i0+6]
            K_rim[i0+5, i0+4] = -K_rim[i0+3, i0+6]

            # k_uu
            K_rim[i0+1, i0+1] = (EIz*Pi/R**3*n**4 + GJ*Pi/R**3*n**2 +
                                 Pi*C/R*n**2)
            K_rim[i0+2, i0+2] = K_rim[i0+1, i0+1]

            # k_ub
            K_rim[i0+1, i0+7] = -(EIz*Pi/R**2*n**2 + GJ*Pi/R**2*n**2 -
                                  Pi*C*(r/R)**2*n**2)
            K_rim[i0+2, i0+8] = K_rim[i0+1, i0+7]
            K_rim[i0+7, i0+1] = K_rim[i0+1, i0+7]
            K_rim[i0+8, i0+2] = K_rim[i0+1, i0+7]

            # k_bb
            K_rim[i0+7, i0+7] = EIz*Pi/R + GJ*Pi/R*n**2 + Pi*C*r**2/R*n**2
            K_rim[i0+8, i0+8] = K_rim[i0+7, i0+7]

        return K_rim

    def calc_K_spk(self):
        'Calculate spoke elasticity matrix.'

        K_sp_el = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))
        K_sp_t = np.zeros((4 + self.n_modes*8, 4 + self.n_modes*8))

        for s in range(self.n_spokes):

            # Spoke direction and length
            nhat, ls = self.calc_spoke_n(s)

            Bi = self.B_theta(self.geom.a_rim_nodes[s])
            Ni = np.dot(nhat.T, Bi)

            k_s_el = self.s_sec.young_mod*self.s_sec.area / ls
            k_s_T = self.tension / ls

            K_sp_el = K_sp_el + k_s_el*(np.dot(Ni.T, Ni))
            K_sp_t = K_sp_t + k_s_T*(np.dot(Bi.T, Bi))

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

        for s in range(self.n_spokes):
            theta_s = self.geom.a_rim_nodes[s]
            BiT = self.B_theta(theta_s).T

            # Spoke direction and length
            nhat, ls = self.calc_spoke_n(s)
            EA = self.s_sec.young_mod * self.s_sec.area

            F_adj = F_adj + EA/ls * self.k_n * Omega[s] * np.dot(BiT, nhat)

        return F_adj.flatten()

    def calc_A_adj(self):
        'Calculate spoke adjustment matrix: F_adj = A_adj*Omega.'

        A_adj = np.zeros((4 + self.n_modes*8, self.n_spokes))

        for s in range(self.n_spokes):
            theta_s = self.geom.a_rim_nodes[s]
            BiT = self.B_theta(theta_s).T

            # Spoke direction and length
            nhat, ls = self.calc_spoke_n(s)
            EA = self.s_sec.young_mod * self.s_sec.area

            A_col = (EA/ls) * self.k_n * np.dot(BiT, nhat)

            A_adj[:, s] = A_col.flatten()

        return A_adj

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
        for s in range(self.n_spokes):
            theta_s = self.geom.a_rim_nodes[s]
            Bi = self.B_theta(theta_s)
            nhat, ls = self.calc_spoke_n(s)
            EA = self.s_sec.young_mod * self.s_sec.area

            Theta_t[s, s] = self.k_n * EA/ls
            Theta_d[s, :] = -EA/ls * np.dot(nhat.T, Bi).flatten()

        return Theta_d.dot(np.linalg.inv(K).dot(A_adj)) + Theta_t

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

    def __init__(self, geom, r_sec, s_sec, N=10, tension=0.0):

        self.geom = geom
        self.r_sec = r_sec
        self.s_sec = s_sec

        self.n_spokes = len(geom.lace_hub_n)

        self.k_n = 1.0  #0.0254 / 55.0  # distance per turn - spoke nipple

        self.n_modes = N
        self.tension = tension


class RayleighRitz:
    'Solve lateral deflections using the Rayleigh-Ritz method (continuum).'

    def solve_euler(self):
        k_vv, k_bb, k_vb, k_rr = calc_spoke_stiff(self.geom,
                                                  self.r_sec, self.s_sec)

        # Axial force in the rim
        n_spokes = len(self.geom.lace_hub_n)
        N = -n_spokes*self.tension / (2*np.pi)

        cV = np.zeros(self.n, dtype=np.float)
        cB = np.zeros(self.n, dtype=np.float)
        cG = np.zeros(self.n, dtype=np.float)  # Not part of euler formulation

        EI = self.r_sec.young_mod * self.r_sec.I22
        GJ = self.r_sec.shear_mod * self.r_sec.I11

        # Rim radius and radius of gyration
        R = self.geom.d_rim/2
        r = np.sqrt((self.r_sec.I22 + self.r_sec.I33) / self.r_sec.area)

        Pi = np.pi

        if self.tens_stiff:
            l = np.sqrt(((self.geom.d_rim - self.geom.d1_hub)/2)**2 +
                        self.geom.w1_hub**2)
            k_vv = k_vv + n_spokes/(2*Pi*R) * (self.tension / l)

        # n=1 (zero-th mode)
        A0 = np.zeros((2, 2))

        A0[0, 0] = 2*Pi*R*k_vv
        A0[0, 1] = 2*Pi*R*k_vb
        A0[1, 0] = k_vb
        A0[1, 1] = k_bb + EI/R**2

        b0 = np.array([1, 0])

        x0 = np.linalg.solve(A0, b0)
        cV[0] = x0[0]
        cB[0] = x0[1]

        # n=1 -> higher
        for i in range(1, self.n):
            A = np.zeros((2, 2))

            A[0, 0] = ((EI*Pi/R**3)*i**4 + (GJ*Pi/R**3)*i**2 + Pi*R*k_vv +
                       N*Pi/R*i**2)
            A[0, 1] = -((EI*Pi/R**2)*i**2 + (GJ*Pi/R**2)*i**2 -
                        Pi*R*k_vb - (N*Pi*r**2/R)*i**2)

            A[1, 0] = -A[0, 1]
            A[1, 1] = -((EI*Pi/R) + (GJ*Pi/R)*i**2 + Pi*R*k_bb +
                        N*Pi*(r/R)**2*i**2)

            b = np.array([1, 0])

            x = np.linalg.solve(A, b)

            cV[i] = x[0]
            cB[i] = x[1]

        self.cV = cV
        self.cB = cB
        self.cG = cG  # trivially zero

    def calc_mode_stiff(self, n):
        k_uu, k_bb, k_ub, k_rr = calc_spoke_stiff(self.geom,
                                                  self.r_sec, self.s_sec)

        # Axial force in the rim
        n_spokes = len(self.geom.lace_hub_n)
        N = -n_spokes*self.tension / (2*np.pi)

        EI = self.r_sec.young_mod * self.r_sec.I22
        GJ = self.r_sec.shear_mod * self.r_sec.I11

        # Rim radius and radius of gyration
        R = self.geom.d_rim/2
        r = np.sqrt((self.r_sec.I22 + self.r_sec.I33) / self.r_sec.area)

        Pi = np.pi

        if self.tens_stiff:
            l = np.sqrt(((self.geom.d_rim - self.geom.d1_hub)/2)**2 +
                        self.geom.w1_hub**2)
            k_uu = k_uu + n_spokes/(2*Pi*R) * (self.tension / l)

        A = np.zeros((2, 2))
        if n == 0:
            # zero-th mode
            A[0, 0] = 2*Pi*R*k_uu
            A[0, 1] = 2*Pi*R*k_ub
            A[1, 0] = k_ub
            A[1, 1] = k_bb + EI/R**2

            b0 = np.array([1, 0])
            x0 = np.linalg.solve(A, b0)

            cV = x0[0]
            # cB = x0[1]

        else:
            A = np.zeros((2, 2))

            A[0, 0] = ((EI*Pi/R**3)*n**4 + (GJ*Pi/R**3)*n**2 + Pi*R*k_uu +
                       N*Pi/R*n**2)
            A[0, 1] = -((EI*Pi/R**2)*n**2 + (GJ*Pi/R**2)*n**2 -
                        Pi*R*k_ub - N*Pi*(r/R)**2*n**2)

            A[1, 0] = -A[0, 1]
            A[1, 1] = -((EI*Pi/R) + (GJ*Pi/R)*n**2 + Pi*R*k_bb +
                        N*Pi*r**2/R*n**2)

            b = np.array([1, 0])
            x = np.linalg.solve(A, b)

            cV = x[0]
            # cB = x[1]

        return 1.0/cV

    def calc_buckling_tension(self, n, tens_stiff=False):

        k_vv, k_bb, k_vb, k_rr = calc_spoke_stiff(self.geom, self.r_sec,
                                                  self.s_sec)

        a_1 = np.arctan(2*(self.geom.w1_hub) /
                        (self.geom.d_rim - self.geom.d1_hub))
        a_2 = np.arctan(2*(self.geom.w2_hub) /
                        (self.geom.d_rim - self.geom.d2_hub))

        # shortcuts
        R = self.geom.d_rim/2
        EI = self.r_sec.young_mod * self.r_sec.I22
        GJ = self.r_sec.shear_mod * self.r_sec.I11
        mu = GJ / EI
        Pi = np.pi

        n_spokes = len(self.geom.lace_hub_n)
        if tens_stiff:
            l = calc_spoke_len(self.geom)
            T_cr = 2*Pi*EI/(n_spokes*R**2) *\
                (mu*(1-n**2)**2/(1+mu*n**2) + k_vv*R**4/(EI*n**2)) /\
                (1 - R/(l*n**2))
        else:
            T_cr = 2*Pi*EI/(n_spokes*R**2) *\
                (mu*(1-n**2)**2/(1+mu*n**2) + k_vv*R**4/(EI*n**2))

        return T_cr

    def min_buckling_mode(self, tens_stiff=True):
        # Check first 20 modes

        T_n = (self.calc_buckling_tension(np.arange(2, 21),
                                          tens_stiff=tens_stiff)).tolist()

        T_min = min(T_n)
        n_min = T_n.index(min(T_n)) + 2

        return T_min, n_min

    def calc_stiff(self):
        v0 = np.sum(self.cV)
        return 1.0/v0

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

    def __init__(self, geom, r_sec, s_sec, n=10, tension=0.0,
                 form='euler', tens_stiff=False):

        self.geom = geom
        self.r_sec = r_sec
        self.s_sec = s_sec

        # Include non-linear "string-tension" stiffness component
        self.tens_stiff = tens_stiff
        self.tension = tension

        self.n = n  # number of terms in approximation

        # Solve
        self.solve_euler()


class Pippard:

    def deflection(self, theta):

        psi = np.pi - theta

        R = self.geom.d_rim/2
        EI = self.r_sec.young_mod * self.r_sec.I22
        GJ = self.r_sec.shear_mod * self.r_sec.I11

        n = EI/GJ
        a = self.a
        b = self.b
        g = self.g

        G = self.G
        H = self.H
        U = self.U

        v = R/(n+1)*((n-g**2)/g**2*G*np.cosh(g*psi) -
                     np.cos(a*psi)*np.cosh(b*psi)/(a**2+b**2)**2*(H*((a**2+b**2)**2 +
                                                    n*(a**2-b**2)) + 2*a*b*n*U) -
                     np.sin(a*psi)*np.sinh(b*psi)/(a**2+b**2)**2*(U*((a**2+b**2)**2 +
                                                    n*(a**2-b**2)) - 2*a*b*n*H))

        return v

    def max_t_radial(self):

        k_vv, k_bb, k_vb, k_rr = calc_spoke_stiff(self.geom,
                                                  self.r_sec, self.s_sec)

        R = self.geom.d_rim/2
        EIr = self.r_sec.young_mod * self.r_sec.I33

        K = R**4*k_rr/EIr
        N = len(self.geom.lace_hub_n)

        ar = np.sqrt((np.sqrt(K+1)+1)/2)
        br = np.sqrt((np.sqrt(K+1)-1)/2)
        th = np.pi/N

        TpP = K/(K+1)*1/N + np.exp(br*(1-N/ar)*th)/2 *\
            (K/(K+1)*(1-np.exp(-2*br*th))*np.cos(ar*th) +
             np.sqrt(K)/(K+1)*(1+np.exp(-2*br*th))*np.sin(ar*th))

        TpP = K/(K+1)*1/N + np.exp(br*(1-N/ar)*th)/2 *\
            (K/(K+1)*(1-np.exp(-2*br*th))*np.cos(ar*th) +
             K/(K+1)*(1+np.exp(-2*br*th))*np.sin(ar*th))

        return TpP

    def calc_stiff(self):
        delta_z = self.deflection(0)

        return 1.0 / delta_z

    def __init__(self, geom, r_sec, s_sec):
        self.geom = geom
        self.r_sec = r_sec
        self.s_sec = s_sec

        k_vv, k_bb, k_vb, k_rad = calc_spoke_stiff(geom, r_sec, s_sec)

        R = geom.d_rim/2
        EI = r_sec.young_mod * r_sec.I22
        GJ = r_sec.shear_mod * r_sec.I11

        K2 = R**4 * k_vv/EI
        n = EI/GJ

        M = 1/27 + K2*(3*n + 2)/6
        Q = np.sqrt(K2/27*(K2**2 + 2*K2 + 1 + n*((27*n*K2)/4 + 9*K2 + 1)))

        X = np.power(M + Q, 1.0/3.0) - np.power(np.abs(M - Q), 1.0/3.0)
        Y = np.power(M + Q, 1.0/3.0) + np.power(np.abs(M - Q), 1.0/3.0)

        g = np.sqrt(X - 2.0/3.0)
        a = np.sqrt(0.5*(np.sqrt(X**2/4 + (2*X)/3 + 4.0/9.0 + (3*Y**2)/4) +
                                (X/2 + 2.0/3.0)))
        b = np.sqrt(0.5*(np.sqrt(X**2/4 + (2*X)/3 + 4.0/9.0 + (3*Y**2)/4) -
                                (X/2 + 2.0/3.0)))

        Z1 = (R**2*(n + 1)) / (6*np.sqrt(3)*EI*Q)

        self.G = Z1*((2*a*b*g)/np.sinh(g*np.pi))

        self.H = -Z1*((a*(a**2 + b**2 + g**2)*np.cos(a*np.pi)*np.sinh(b*np.pi) +
                  b*(a**2 + b**2 - g**2)*np.sin(a*np.pi)*np.cosh(b*np.pi)) /
                (np.cosh(b*np.pi)**2 - np.cos(a*np.pi)**2))

        self.U = -Z1*((a*(a**2 + b**2 + g**2)*np.sin(a*np.pi)*np.cosh(b*np.pi) -
                  b*(a**2 + b**2 - g**2)*np.cos(a*np.pi)*np.sinh(b*np.pi)) /
                (np.cosh(b*np.pi)**2 - np.cos(a*np.pi)**2))

        self.a = a
        self.b = b
        self.g = g


class Hetenyi:

    def stiff(self):

        e = self.eta
        a = self.alpha
        b = self.beta
        Pi = np.pi

        y0 = self.R**3/(4*a*b*self.EI)

        t1 = 2*a*b/(np.pi*e**2)

        t2 = (b*np.sinh(a*Pi)*np.cosh(a*Pi) + a*np.sin(b*Pi)*np.cos(b*Pi)) /\
            (e*(np.sinh(a*Pi)**2 + np.sin(b*Pi)**2))

        return -1.0 / (y0*(t1 - t2))

    def __init__(self, geom, r_sec, s_sec):
        self.geom = geom
        self.r_sec = r_sec
        self.s_sec = s_sec

        self.R = geom.d_rim/2
        self.EI = r_sec.young_mod*r_sec.I33

        k_uu, k_bb, k_ub, k_rr = calc_spoke_stiff(geom, r_sec, s_sec)

        self.eta = np.sqrt(self.R**4*k_rr/self.EI + 1)
        self.alpha = np.sqrt((self.eta-1)/2)
        self.beta = np.sqrt((self.eta+1)/2)
