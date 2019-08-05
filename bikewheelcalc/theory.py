'Tools for theoretical calculations on bicycle wheels.'

import numpy as np
from numpy import pi
from bikewheelcalc import BicycleWheel, ModeMatrix


def calc_buckling_tension(wheel, approx='linear', N=20):
    'Find minimum critical tension within first N modes.'

    def calc_Tc_mode_quad(n):
        'Calculate critical tension for nth mode using full quadratic form.'

        CT = GJ + EIw*n**2/R**2

        y0 = 0.
        if 'y_0' in wheel.rim.sec_params:
            y0 = wheel.rim.sec_params['y_0']

        A = -y0*R*(n**2 - R*kT)

        B = (-EI/R*(n**2 + n**4*y0/R - R*kT)
             -CT*n**2/R*(n**2 - R*kT + y0/R*(2*n**2 - 1))
             -R*kpp*(n**2 - R*kT) + 2*R*y0*kup*n**2 + R**2*y0*kuu)

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
    EI = wheel.rim.young_mod * wheel.rim.I_lat
    EIw = wheel.rim.young_mod * wheel.rim.I_warp
    GJ = wheel.rim.shear_mod * wheel.rim.J_tor

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

    elif approx == 'small_mu':
        T_c = 11.875 * GJ/(ns*R**2) * np.power(k_uu*R**4/GJ, 2.0/3.0)
        n_c = np.power(k_uu*R**4/(2*GJ), 1.0/6.0)

    else:
        raise ValueError('Unknown approximation: {:s}'.format(approx))

    return T_c, n_c


def calc_buckling_tension_modematrix(wheel, smeared_spokes=False, coupling=True, r0=True, N=24):
    'Estimate buckling tension from condition number of stiffness matrix.'

    mm = ModeMatrix(wheel, N=N)

    K_matl = (mm.K_rim_matl(r0=r0) +
              mm.K_spk(tension=False, smeared_spokes=smeared_spokes))

    K_geom = (mm.K_spk_geom(smeared_spokes=smeared_spokes) -
              mm.K_rim_geom(r0=r0))

    # Need to solve generalized eigienvalue problem:
    #   (K_matl + T*K_geom)*x = 0   =>   K_matl*x = T*(-K_geom)*x
    if coupling:
        A, B = (K_matl, -K_geom)
    else:
        A, B = (mm.get_K_uncoupled(K_matl),
                -mm.get_K_uncoupled(K_geom))

    # Find a scalar t such that A - t*B is invertible.
    #  This is equivalent to finding t such that K_matl + t*K_geom is
    #  invertible. A proper choice is a small multiple of estimated Tc
    Tc_lin_est = calc_buckling_tension(wheel, approx='linear', N=N)

    t = 0.1*Tc_lin_est[0]

    # Solve the related eigenvalue problem (A - t*B)^-1 * B
    w_c, v_c = np.linalg.eig(np.linalg.inv(A - t*B).dot(B))

    # Select non-zero eigenvalues and transform back to (A, B) eigenvalues
    nz_eig_ix = np.nonzero(~np.isclose(w_c, 0))[0]
    w_ab = t + 1./w_c[nz_eig_ix]

    return np.min(np.real(w_ab)[np.real(w_ab) > 0])


def lat_mode_stiff(wheel, n, smeared_spokes=True, buckling=True, tension=True):
    'Calculate lateral mode stiffness'

    kbar = wheel.calc_kbar(tension=tension)
    kuu = kbar[0, 0]
    kup = kbar[0, 3]
    kpp = kbar[3, 3]

    # shortcuts
    ns = len(self.wheel.spokes)
    R = self.wheel.rim.radius
    EI = self.wheel.rim.young_mod * self.wheel.rim.I_lat
    EIw = self.wheel.rim.young_mod * self.wheel.rim.I_warp
    GJ = self.wheel.rim.shear_mod * self.wheel.rim.J_tor
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

    k_sp = wheel.calc_kbar(wheel)
    k_uu = k_sp[0, 0]

    n_spokes = len(wheel.spokes)
    EI = wheel.rim.young_mod * wheel.rim.I_lat
    GJ = wheel.rim.shear_mod * wheel.rim.J_tor
    cc = (GJ/EI)/(GJ/EI + 1)

    Pn_lat = n_spokes/(8*np.pi*wheel.rim.radius) *\
        np.power(4*EI * cc / k_uu, 0.25)

    return Pn_lat


def calc_Pn_rad(wheel):
    'Radial Pippard number (ratio of length scale to spoke spacing).'

    k_sp = wheel.calc_kbar(wheel)
    k_vv = k_sp[1, 1]

    n_spokes = len(wheel.spokes)
    EI = wheel.rim.young_mod * wheel.rim.I_rad

    Pn_rad = n_spokes/(2*np.pi*wheel.rim.radius) *\
        np.power(4*EI / k_vv, 0.25)

    return Pn_rad


def calc_lambda_lat(wheel):
    'Calculate lambda = k_uu*R^4/EI_lat'

    k_sp = wheel.calc_kbar(wheel)
    k_uu = k_sp[0, 0]

    return k_uu*wheel.rim.radius**4 / (wheel.rim.young_mod * wheel.rim.I_lat)


def calc_lambda_rad(wheel):
    'Calculate lambda = k_vv*R^4/EI_rad'

    k_sp = wheel.calc_kbar(wheel)
    k_vv = k_sp[1, 1]

    return k_vv*wheel.rim.radius**4 / (wheel.rim.young_mod * wheel.rim.I_rad)
