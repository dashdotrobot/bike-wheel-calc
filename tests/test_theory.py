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
    EI = w.rim.young_mod*w.rim.I22
    GJ = w.rim.shear_mod*w.rim.I11

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
    w.rim.sec_params = {'y_s': 0.00001}
    Tc_lin = calc_buckling_tension(w, approx='linear', N=20)
    Tc_quad = calc_buckling_tension(w, approx='quadratic', N=20)

    assert np.abs((Tc_lin[0] - Tc_quad[0]) / Tc_quad[0]) < 0.001

# def test_Tc_MM_uncoupled(std_ncross):
#     'Test '
#     w = std_ncross(0)

#     # Tc_mm = calc_buckling_tension_modematrix(smeared_spokes=True, )

#     assert False
