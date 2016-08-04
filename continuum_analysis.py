#!/usr/bin/env python

'Tools for continuum analysis (a la Pippard) of bicycle wheels.'

import numpy as np




def calc_buckling_tension(wheel, approx=None, N=20):
    'Find minimum critical tension within first N modes.'

    def calc_Tc_mode_quad(n):
        'Calculate critical tension for nth mode using full quadratic form.'

        CT = GJ + EIw*n**2/R**2

        if 'y_s' in wheel.rim.sec_params:
            y0 = wheel.rim.sec_params['y_c'] -\
                wheel.rim.sec_params['y_s']
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

    def calc_Tc_mode_lin(n):
        'Calculate critical tension for nth mode using linear approximation.'

        CT = GJ + EIw*n**2/R**2
        mu = CT/EI

        A = 1 + mu*n**2 + k_bb*R**2/EI
        B = n**4 + mu*n**2
        C = 2*n**2*(1 + mu) - k_ub*R**3/EI
        D = mu*n**2*(n**2 - 1)**2

        f_T = n**2 / (n**2 - R/l)

        T_c = 2*pi*EI/(ns*R**2*n**2*A) * f_T *\
            (A*k_uu*R**4/EI + B*k_bb*R**2/EI + C*k_ub*R**3/EI + D)

        return T_c

    k_s = calc_continuum_stiff(wheel, tension=0.0)
    k_uu = k_s[0, 0]
    k_ub = k_s[0, 3]
    k_bb = k_s[3, 3]

    # shortcuts
    pi = np.pi
    ns = len(wheel.spokes)
    R = wheel.rim.radius
    l = wheel.spokes[0].length
    EI = wheel.rim.young_mod * wheel.rim.I22
    EIw = wheel.rim.young_mod * wheel.rim.Iw
    GJ = wheel.rim.shear_mod * wheel.rim.I11
    

    rx = np.sqrt(wheel.rim.I22 / wheel.rim.area)
    ry = np.sqrt(wheel.rim.I33 / wheel.rim.area)

    if approx == 'linear':

        T_cn = [calc_Tc_mode_lin(n) for n in range(2, N+1)]
        T_c = min(T_cn)
        n_c = T_cn.index(T_c) + 2

    elif approx == 'small_mu':

        T_c = 11.875 * GJ/(ns*R**2) * np.power(k_uu*R**4/GJ, 2.0/3.0)
        n_c = np.power(k_uu*R**4/(2*GJ), 1.0/6.0)

    else:

        T_cn = [calc_Tc_mode_quad(n) for n in range(2, N+1)]
        T_c = min(T_cn)
        n_c = T_cn.index(T_c) + 2

    return T_c, n_c


def calc_continuum_stiff(wheel, tension=0.0):

    def k_spoke(s):
        'Stiffness matrix for a single spoke.'

        d = s.rim_pt[2]
        n = np.append(s.n, 0)

        k_sp = s.EA / s.length * np.outer(n, n) +\
            tension / s.length * np.diag([1, 1, 1, 0])

        k_sp[0:3, 3] = d * k_sp[0:3, 1]
        k_sp[3, 0:3] = d * k_sp[0:3, 1]
        k_sp[3, 3] = d**2 * k_sp[1, 1]

        return k_sp

    s1 = wheel.spokes[0]
    s2 = wheel.spokes[1]

    return len(wheel.spokes) / (4*np.pi*wheel.rim.radius) * \
        (k_spoke(s1) + k_spoke(s2))


def calc_Pn_lat(wheel):
    'Lateral Pippard number (ratio of length scale to spoke spacing).'

    k_sp = wheel.calc_continuum_stiff()
    k_uu = k_sp[0, 0]

    n_spokes = len(wheel.spokes)
    EI = wheel.rim.young_mod * wheel.rim.I22
    GJ = wheel.rim.shear_mod * wheel.rim.I11
    cc = (GJ/EI)/(GJ/EI + 1)

    Pn_lat = n_spokes/(2*np.pi*wheel.rim.radius) *\
        np.power(4*EI / k_uu, 0.25) * cc

    return Pn_lat


def calc_Pn_rad(wheel):
    'Radial Pippard number (ratio of length scale to spoke spacing).'

    k_sp = wheel.calc_continuum_stiff()
    k_vv = k_sp[1, 1]

    n_spokes = len(wheel.spokes)
    EI = wheel.rim.young_mod * wheel.rim.I33

    Pn_rad = n_spokes/(2*np.pi*wheel.rim.radius) *\
        np.power(4*EI / k_vv, 0.25)

    return Pn_rad


def calc_lambda_lat(wheel):
    'Calculate lambda = k_uu*R^4/EI_lat'

    k_sp = wheel.calc_continuum_stiff()
    k_uu = k_sp[0, 0]

    return k_uu*wheel.rim.radius**4 / (wheel.rim.young_mod * wheel.rim.I22)


def calc_lambda_rad(wheel):
    'Calculate lambda = k_vv*R^4/EI_rad'

    k_sp = wheel.calc_continuum_stiff()
    k_vv = k_sp[1, 1]

    return k_vv*wheel.rim.radius**4 / (wheel.rim.young_mod * wheel.rim.I33)


def print_continuum_stats(wheel):
    'Print summary information about the wheel.'

    print 'lambda (lat) :', wheel.calc_lambda_lat()
    print 'lambda (rad) :', wheel.calc_lambda_rad()
    print 'R/le (lat)   :', np.power(wheel.calc_lambda_lat(), 0.25)
    print 'R/le (rad)   :', np.power(wheel.calc_lambda_rad(), 0.25)
    print 'Pn_lat       :', wheel.calc_Pn_lat()
    print 'Pn_rad       :', wheel.calc_Pn_rad()