import numpy as np


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