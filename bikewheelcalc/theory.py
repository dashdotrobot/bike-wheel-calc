'Tools for theoretical calculations on bicycle wheels.'

import numpy as np
from numpy import pi
from scipy.optimize import minimize
from bikewheelcalc import BicycleWheel, ModeMatrix


def calc_buckling_tension(wheel, approx='linear', N=20):
    'Find minimum critical tension within first N modes.'

    def calc_Tc_mode_quad(n):
        'Calculate critical tension for nth mode using full quadratic form.'

        CT = GJ + EIw*n**2/R**2

        y0 = 0.
        if 'y_s' in wheel.rim.sec_params:
            y0 = wheel.rim.sec_params['y_s']

        A = (R**2*kT - n**2*R)*y0

        B = (EI/R*(R*kT - n**2 - n**4*y0/R)
             +CT*n**2/R*(R*kT - n**2 - y0/R*(2*n**2 - 1))
             -R*kpp*(n**2 - R*kT) + 2*R*y0*kup*n**2 + R**2*kuu*y0)

        C = (EI*CT*n**2/R**4*(n**2-1)**2
             +EI*(kuu + 2*kup/R*n**2 + kpp/R**2*n**4)
             +CT*n**2*(kuu + 2*kup/R + kpp/R**2))

        # Solve for the smaller root
        return (2*pi*R/ns) * (-B - np.sqrt(B**2 - 4*A*C))/(2*A)

    def calc_Tc_mode_lin(n):
        'Calculate critical tension for nth mode using linear approximation.'

        mu = (GJ + EIw*n**2/R**2)/EI
        luu, lup, lpp = (R**4/EI) * np.array([kuu, kup, kpp])

        t_c = (luu +
               (mu*n**2*(n**2-1)**2 + (n**4+mu*n**2)*lpp
                + 2*n**2*(mu+1)*lup - lup**2)/(1+mu*n**2+lpp))

        return 2*pi*EI/(ns*R**2) * t_c/(n**2 - R*kT)

    # shortcuts
    ns = len(wheel.spokes)
    R = wheel.rim.radius
    EI = wheel.rim.young_mod * wheel.rim.I22
    EIw = wheel.rim.young_mod * wheel.rim.Iw
    GJ = wheel.rim.shear_mod * wheel.rim.I11

    kbar = wheel.calc_kbar(tension=False)
    kuu, kup, kpp = (kbar[0, 0], kbar[0, 3], kbar[3, 3])
    kT = (2*pi*R/ns)*wheel.calc_kbar_geom()[0, 0]

    if approx == 'linear':
        T_cn = [calc_Tc_mode_lin(n) for n in range(2, N+1)]
        T_c = min(T_cn)
        n_c = T_cn.index(T_c) + 2

    elif approx == 'quadratic':
        T_cn = [calc_Tc_mode_quad(n) for n in range(2, N+1)]
        T_c = min(T_cn)
        n_c = T_cn.index(T_c) + 2

    elif approx == 'modematrix':
        T_c = calc_Tc_modematrix(smeared_spokes=True, coupling=False)
        n_c = None

    elif approx == 'small_mu':
        T_c = 11.875 * GJ/(ns*R**2) * np.power(k_uu*R**4/GJ, 2.0/3.0)
        n_c = np.power(k_uu*R**4/(2*GJ), 1.0/6.0)

    else:
        raise ValueError('Unknown approximation: {:s}'.format(approx))

    return T_c, n_c


def calc_buckling_tension_modematrix(smeared_spokes=True, coupling=True, r0=True):
    'Estimate buckling tension from condition number of stiffness matrix.'

    mm = ModeMatrix(wheel, N=N)

    def neg_cond(T):
        wheel.apply_tension(T)

        K = (mm.K_rim(buckling=True, r0=r0) +
             mm.K_spk(smeared_spokes=smeared_spokes))

        if not coupling:
            K = mm.get_K_uncoupled(buckling=True,
                                   smeared_spokes=smeared_spokes)

        return -np.linalg.cond(K)

    # Find approximate buckling tension from linear analytical solution
    Tc_approx = calc_buckling_tension(wheel, approx='linear')

    # Maximize the condition number as a function of tension
    res = minimize(fun=neg_cond, x0=[Tc_approx], method='Nelder-Mead',
                   options={'maxiter': 50})

    return res.x[0]


def lat_mode_stiff(wheel, n, smeared_spokes=True, buckling=True, tension=True):
    'Calculate lateral mode stiffness'

    kbar = wheel.calc_kbar(tension=tension)
    kuu = kbar[0, 0]
    kup = kbar[0, 3]
    kpp = kbar[3, 3]

    # shortcuts
    ns = len(self.wheel.spokes)
    R = self.wheel.rim.radius
    EI = self.wheel.rim.young_mod * self.wheel.rim.I22
    EIw = self.wheel.rim.young_mod * self.wheel.rim.Iw
    GJ = self.wheel.rim.shear_mod * self.wheel.rim.I11
    CT = GJ + EIw*n**2/R**2

    Tb = 0.
    if buckling:
        Tb = np.sum([s.tension*s.n[1] for s in self.wheel.spokes]) / (2*pi*R)

    if n == 0:
        return 2*pi*R*(kuu - R**2*kup**2/(EI + R**2*kpp))
    elif n == 1:
        return pi*R*kuu + pi*(((EI/R**3 + CT/R**3)*(kpp/R + 2*R*kup) - kup**2)/
                              (EI/R**3 + CT/R**3 + kpp/R))
    elif n > 0 and isinstance(n, int):
        pass
    else:
        raise ValueError('Invalid value for integer mode n: {:s}'
                         .format(str(n)))


def calc_lat_stiff(wheel, theta=0., N=20, smeared_spokes=True, tension=True, buckling=True, coupling=False, r0=False):
    'Calculate lateral stiffness.'

    mm = ModeMatrix(wheel, N=N)

    F_ext = mm.F_ext(0., [1., 0., 0., 0.])
    d = np.zeros(F_ext.shape)

    K = (mm.K_rim(tension=buckling, r0=r0) +
         mm.K_spk(tension=tension, smeared_spokes=smeared_spokes))

    if coupling:
        d = np.linalg.solve(K, F_ext)
    else:
        ix_uc = mm.get_ix_uncoupled(dim='lateral')
        K = mm.get_K_uncoupled(K=K, dim='lateral')
        d[ix_uc] = np.linalg.solve(K, F_ext[ix_uc])

    return 1. / mm.B_theta(0.).dot(d)[0]


def calc_rad_stiff(wheel, theta=0., N=20, smeared_spokes=True, tension=True, buckling=True, coupling=True, r0=False):
    'Calculate radial stiffness.'

    mm = ModeMatrix(wheel, N=N)

    F_ext = mm.F_ext(theta, [0., 1., 0., 0.])
    d = np.zeros(F_ext.shape)

    K = (mm.K_rim(tension=buckling, r0=r0) +
         mm.K_spk(tension=tension, smeared_spokes=smeared_spokes))

    if coupling:
        d = np.linalg.solve(K, F_ext)
    else:
        ix_uc = mm.get_ix_uncoupled(dim='radial')
        K = mm.get_K_uncoupled(K=K, dim='radial')
        d[ix_uc] = np.linalg.solve(K, F_ext[ix_uc])

    return 1. / mm.B_theta(theta).dot(d)[1]

def calc_tor_stiff(wheel, theta=0., N=20, smeared_spokes=True, tension=True, buckling=True, coupling=True, r0=False):
    'Calculate torsional (wind-up) stiffness in [N/rad].'

    mm = ModeMatrix(wheel, N=N)

    F_ext = mm.F_ext(theta, [0., 0., 1., 0.])
    d = np.zeros(F_ext.shape)

    K = (mm.K_rim(tension=buckling, r0=r0) +
         mm.K_spk(tension=tension, smeared_spokes=smeared_spokes))

    if coupling:
        d = np.linalg.solve(K, F_ext)
    else:
        ix_uc = mm.get_ix_uncoupled(dim='radial')
        K = mm.get_K_uncoupled(K=K, dim='radial')
        d[ix_uc] = np.linalg.solve(K, F_ext[ix_uc])

    return wheel.rim.radius / mm.B_theta(theta).dot(d)[2]


def calc_Pn_lat(wheel):
    'Lateral Pippard number (ratio of length scale to spoke spacing).'

    k_sp = calc_continuum_stiff(wheel)
    k_uu = k_sp[0, 0]

    n_spokes = len(wheel.spokes)
    EI = wheel.rim.young_mod * wheel.rim.I22
    GJ = wheel.rim.shear_mod * wheel.rim.I11
    cc = (GJ/EI)/(GJ/EI + 1)

    Pn_lat = n_spokes/(8*np.pi*wheel.rim.radius) *\
        np.power(4*EI * cc / k_uu, 0.25)

    return Pn_lat


def calc_Pn_rad(wheel):
    'Radial Pippard number (ratio of length scale to spoke spacing).'

    k_sp = calc_continuum_stiff(wheel)
    k_vv = k_sp[1, 1]

    n_spokes = len(wheel.spokes)
    EI = wheel.rim.young_mod * wheel.rim.I33

    Pn_rad = n_spokes/(2*np.pi*wheel.rim.radius) *\
        np.power(4*EI / k_vv, 0.25)

    return Pn_rad


def calc_lambda_lat(wheel):
    'Calculate lambda = k_uu*R^4/EI_lat'

    k_sp = calc_continuum_stiff(wheel)
    k_uu = k_sp[0, 0]

    return k_uu*wheel.rim.radius**4 / (wheel.rim.young_mod * wheel.rim.I22)


def calc_lambda_rad(wheel):
    'Calculate lambda = k_vv*R^4/EI_rad'

    k_sp = calc_continuum_stiff(wheel)
    k_vv = k_sp[1, 1]

    return k_vv*wheel.rim.radius**4 / (wheel.rim.young_mod * wheel.rim.I33)


def print_continuum_stats(wheel):
    'Print summary information about the wheel.'

    print('lambda (lat) :', calc_lambda_lat(wheel))
    print('lambda (rad) :', calc_lambda_rad(wheel))
    print('R/le (lat)   :', np.power(calc_lambda_lat(wheel), 0.25))
    print('R/le (rad)   :', np.power(calc_lambda_rad(wheel), 0.25))
    print('Pn_lat       :', calc_Pn_lat(wheel))
    print('Pn_rad       :', calc_Pn_rad(wheel))


def mode_stiff(wheel, n, tension=0.0):
    'Calculate stiffness for the nth mode.'

    k_s = calc_continuum_stiff(wheel, tension)
    k_uu = k_s[0, 0]
    k_up = k_s[0, 3]
    k_pp = k_s[3, 3]

    # shortcuts
    pi = np.pi
    ns = len(wheel.spokes)
    R = wheel.rim.radius
    # l = wheel.spokes[0].length
    EI = wheel.rim.young_mod * wheel.rim.I22
    EIw = wheel.rim.young_mod * wheel.rim.Iw
    GJ = wheel.rim.shear_mod * wheel.rim.I11

    rx = np.sqrt(wheel.rim.I22 / wheel.rim.area)
    ry = np.sqrt(wheel.rim.I33 / wheel.rim.area)

    CT = GJ + EIw*n**2/R**2

    # Shear center coordinate
    if 'y_s' in wheel.rim.sec_params:
        y0 = wheel.rim.sec_params['y_c'] -\
            wheel.rim.sec_params['y_s']
    else:
        y0 = 0.0

    # Nr = ns*tension / (2*pi)
    Nr = np.sum([s.tension*s.n[1] for s in wheel.spokes]) / (2*pi)

    if n == 0:
        U_uu = 2*pi*R*k_uu
        U_ub = 2*pi*R*k_up
        U_bb = 2*pi*EI/R + 2*pi*R*k_pp + 2*pi*Nr*y0
    else:  # n > 0
        U_uu = pi*EI*n**4/R**3 + pi*CT*n**2/R**3 + pi*R*k_uu \
            - pi*Nr*n**2/R - pi*Nr*n**2*ry**2/R**3

        U_ub = -pi*EI*n**2/R**2 - pi*CT*n**2/R**2 + pi*R*k_up \
            - pi*Nr*n**2*ry**2/R**2 - pi*Nr*n**2*y0/R

        U_bb = pi*EI/R + pi*CT*n**2/R + pi*R*k_pp\
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

    return Kn_u, Kn_p


def lateral_stiffness(wheel, N=20, tension=0.0):
    'Calculate lateral stiffness for a point load in N/m.'

    Fn = np.zeros(N)  # flexibility of nth mode

    for n in range(len(Fn)):
        Fn[n] = 1.0 / mode_stiff(wheel, n, tension)[0]

    K_lateral = 1.0 / sum(Fn)

    return K_lateral


def lateral_stiffness_phi(wheel, N=20, tension=0.0):
    'Calculate the lateral/rotation stiffness P/phi for a point load.'

    Fn = np.zeros(N)  # flexibility of nth mode

    for n in range(len(Fn)):
        Fn[n] = 1.0 / mode_stiff(wheel, n, tension)[1]

    K_lateral_phi = 1.0 / sum(Fn)

    return K_lateral_phi
