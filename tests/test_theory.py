import pytest
from bikewheelcalc import *


# -----------------------------------------------------------------------------
# Buckling tests
# -----------------------------------------------------------------------------

def test_Tc_linear(std_ncross):
    'Linear approximation'

    w = std_ncross(0)
    Tc_ = calc_buckling_tension(w, approx='linear', N=20)

    K_s = 210e9*np.pi/4*(1.8e-3)**2 / np.hypot(0.3 - 0.025, 0.025)

    R = w.rim.radius
    EI = w.rim.young_mod*w.rim.I_lat
    GJ = w.rim.shear_mod*w.rim.J_tor

    c1_2 = (0.025/np.hypot(0.3 - 0.025, 0.025))**2
    c2 = (0.3 - 0.025)/np.hypot(0.3 - 0.025, 0.025)
    kuu = 36*K_s*c1_2 / (2*np.pi*R)
    kT = (1 - c1_2)/c2 / np.hypot(0.3 - 0.025, 0.025)

    Kb = np.pi*EI/R**3*(2**2-1)**2
    Kt = np.pi*GJ/R**3*2**2*(2**2-1)**2

    Kn0 = np.pi*R*kuu + Kb*Kt/(Kb+Kt)

    Tc = 2*R/36. * Kn0/(2**2 - R*kT)

    assert np.allclose(Tc, Tc_[0])

def test_Tc_lin_quad(std_ncross):
    'Linear and Quadratic should give same result when y0~0'

    w = std_ncross(0)
    w.rim.sec_params = {'y_0': 0.00001}
    Tc_lin = calc_buckling_tension(w, approx='linear', N=20)
    Tc_quad = calc_buckling_tension(w, approx='quadratic', N=20)

    assert np.abs((Tc_lin[0] - Tc_quad[0]) / Tc_quad[0]) < 0.001

def test_Tc_modemat_lin(std_ncross):
    'Mode matrix method should give same result as linear.'

    w = std_ncross(3)
    Tc_ = calc_buckling_tension(w, approx='linear', N=20)

    Tc_mm = calc_buckling_tension_modematrix(w,
                                             smeared_spokes=True,
                                             coupling=True,
                                             r0=False,
                                             N=20)

    assert np.allclose(Tc_[0], Tc_mm)

def test_Tc_modemat_quad(std_ncross):
    'Mode matrix method should give same result as quadratic.'

    w = std_ncross(3)
    w.rim.sec_params = {'y_0': 0.005}

    # Calculate buckling tension from generalized eigenvalue approach
    Tc_mm = calc_buckling_tension_modematrix(w,
                                             smeared_spokes=True,
                                             coupling=False,
                                             r0=False,
                                             N=6)

    # Check against quadratic solution INCLUDING y0^2 term
    n = 2
    ns = len(w.spokes)
    R = w.rim.radius
    CT = w.rim.shear_mod*w.rim.J_tor
    EI = w.rim.young_mod*w.rim.I_lat
    y0 = w.rim.sec_params['y_0']

    kbar = w.calc_kbar(tension=False)
    kuu, kup, kpp = (kbar[0, 0], kbar[0, 3], kbar[3, 3])
    kT = (2*pi*R/ns)*w.calc_kbar_geom()[0, 0]

    A = -y0*R*(n**2 - R*kT + n**4*y0/R)

    B = (-EI/R*(n**2 + n**4*y0/R - R*kT)
         -CT*n**2/R*(n**2 - R*kT + y0/R*(2*n**2 - 1))
         -R*kpp*(n**2 - R*kT) + 2*R*y0*kup*n**2 + R**2*y0*kuu)

    C = (EI*CT*n**2/R**4*(n**2-1)**2
         +EI*(kuu + 2*kup/R*n**2 + kpp/R**2*n**4)
         +CT*n**2*(kuu + 2*kup/R + kpp/R**2))

    # Solve for the smaller root
    Tc_qd = (2*pi*R/ns) * (-B - np.sqrt(B**2 - 4*A*C))/(2*A)

    assert np.allclose(Tc_qd, Tc_mm)


def test_Klat_uncoupled(std_ncross):
    'Check that calc_lat_stiff() and Eqn. (2.71) give same result'

    w = std_ncross(0)
    w.rim.sec_params['y_0'] = 0.
    w.apply_tension(100.)

    # Analytical solution
    R = w.rim.radius
    EI = w.rim.young_mod*w.rim.I_lat
    GJ = w.rim.shear_mod*w.rim.J_tor

    kuu = w.calc_kbar(tension=True)[0, 0]
    Tb = np.sum([s.tension*s.n[1] for s in w.spokes]) / (2.*np.pi*R)

    Kb = lambda n: np.pi*EI     /R**3*(n**2 - 1)**2
    Kt = lambda n: np.pi*GJ*n**2/R**3*(n**2-1)**2

    K0 = 2*np.pi*R*kuu
    K1 = np.pi*R*kuu - np.pi*Tb
    Kn = lambda n: np.pi*R*kuu + Kb(n)*Kt(n)/(Kb(n)+Kt(n)) - np.pi*n**2*Tb

    K_lat_mode = 1./(1./K0 + 1./K1 + np.sum([1./Kn(n) for n in range(2, 11)]))

    # Theory
    K_lat_mm = calc_lat_stiff(wheel=w, N=10,
                              smeared_spokes=True, tension=True,
                              buckling=True, coupling=False)

    assert np.allclose(K_lat_mode, K_lat_mm)

def test_Krad_rotsymm(std_ncross):
    'Check that stiffness is identical at each spoke'

    w = std_ncross(0)
    w.rim.sec_params['y_0'] = 0.
    w.apply_tension(1.)

    Krad = [calc_rad_stiff(w, theta=s.rim_pt[1], N=36, tension=True,
                           smeared_spokes=False, coupling=False, r0=True)
            for s in w.spokes[:5]]

    assert np.allclose(Krad, Krad[0])
