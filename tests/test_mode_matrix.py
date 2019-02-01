import pytest
import sys
import numpy as np
from bikewheelcalc import *


# -----------------------------------------------------------------------------
# Stiffness matrix tests
# -----------------------------------------------------------------------------

def test_K_matl_geom(std_ncross):
    'Check that _geom and _matl stiffness matrices are consistent'

    w = std_ncross(3)
    w.rim.sec_params['y_s'] = 0.001
    w.apply_tension(100.)

    mm = ModeMatrix(w, N=20)

    K1 = (mm.K_rim(tension=True, r0=True) +
          mm.K_spk(tension=True, smeared_spokes=False))

    K2 = (mm.K_rim(tension=False, r0=True) -
          100.*mm.K_rim_geom(r0=True) +
          mm.K_spk(tension=False, smeared_spokes=False) +
          100.*mm.K_spk_geom(smeared_spokes=False))

    assert np.allclose(K1, K2)

def test_K_radial_singular(std_ncross):
    'Radial-spoked wheel at T=0 should have a singular matrix'

    w = std_ncross(0)
    w.rim.sec_params['y_s'] = 0.001
    w.apply_tension(0.)

    mm = ModeMatrix(w, N=20)

    K = (mm.K_rim(tension=False, r0=True) +
         mm.K_spk(tension=False, smeared_spokes=True))

    assert np.linalg.cond(K) > 1./sys.float_info.epsilon

def test_K_pos_def(std_ncross):
    'Check properties of stiffness matrix'

    # Tangent-spoked wheel should have a positive-definite stiffness matrix
    w = std_ncross(3)
    w.rim.sec_params['y_s'] = 0.0
    w.apply_tension(0.)

    mm = ModeMatrix(w, N=36)

    K = (mm.K_rim(tension=False, r0=True) +
         mm.K_spk(tension=False, smeared_spokes=False))

    # Symmetric
    assert np.allclose(K, K.transpose())

    # Positive eigenvalues
    assert np.all(np.linalg.eigvals(K) > -1e-8)

def test_K_lat(std_ncross):
    'Check that ModeMatrix and Eqn. (2.71) give same result'

    w = std_ncross(0)
    w.rim.sec_params['y_s'] = 0.
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


    # Mode Matrix method
    mm = ModeMatrix(w, N=10)
    ix_uc = mm.get_ix_uncoupled(dim='lateral')
    F_ext = mm.F_ext([0.], np.array([[1., 0., 0., 0.]]))[ix_uc]
    K = mm.get_K_uncoupled(K=(mm.K_rim(tension=True, r0=False) +
                              mm.K_spk(tension=True, smeared_spokes=True)),
                           dim='lateral')

    d = np.linalg.solve(K, F_ext)

    K_lat_mm = 1. / mm.B_theta(0.)[:, ix_uc].dot(d)[0]

    assert np.allclose(K_lat_mode, K_lat_mm)


# -----------------------------------------------------------------------------
# Spoke adjustment tests
# -----------------------------------------------------------------------------

def test_A_adj(std_ncross):
    'Check properties of the spoke adjustment matrix'

    w = std_ncross(3)
    mm = ModeMatrix(w, N=10)

    A = mm.A_adj()

    # Spoke adjustment has the same effect as a force applied along the spoke vector
    s = w.spokes[5]  # Chose an arbitrary spoke
    assert np.allclose(mm.A_adj()[:, 5],
                       mm.F_ext(theta=s.rim_pt[1],
                                f=s.EA/s.length * np.append(s.n, 0.)))

def test_spoke_tension(std_ncross):
    'Check that Spoke.calc_tension_change() and ModeMatrix.spoke_tension_change give same result.'

    w = std_ncross(3)
    mm = ModeMatrix(w, N=10)

    # No deformation, only adjustment
    dm = np.zeros(4 + 8*10)
    a = np.zeros(len(w.spokes))
    a[5] = 0.001

    dT = mm.spoke_tension_change(dm, a)
    dT_s = w.spokes[5].calc_tension_change([0., 0., 0., 0.], a[5])

    # Check correct result for single spoke
    assert np.allclose(dT[5], dT_s)

    # Check all others are zero
    assert np.allclose(dT[:5], 0.)
    assert np.allclose(dT[6:], 0.)

def test_uniform_tension(std_ncross):
    'Check radial and tension influence functions against analytical solution'

    w = std_ncross(0.)
    w.apply_tension(0.001)  # Small tension to make K invertible

    mm = ModeMatrix(w, N=24)
    K = mm.K_rim() + mm.K_spk()

    theta_s = [s.rim_pt[1] for s in w.spokes]

    # Tighten all spokes by one millimeter
    a = 0.001*np.ones(len(w.spokes))
    dm = np.linalg.solve(K, mm.A_adj().dot(a))
    v = mm.B_theta(theta_s, comps=1).dot(dm)
    T = mm.spoke_tension_change(dm, a)

    # Theoretical solution
    R = w.rim.radius
    ns = len(w.spokes)
    ls = w.spokes[0].length
    EAr = w.rim.young_mod*w.rim.area
    EAs = w.spokes[0].EA
    n2 = w.spokes[0].n[1]

    v_theor = 0.001 / (2*np.pi*ls*EAr/(ns*n2*R*EAs) + n2)
    T_theor = EAs/ls*(0.001 - v_theor*n2)

    # All tensions same
    assert np.allclose(T - np.mean(T), 0.)

    # All displacements same
    assert np.allclose(v - np.mean(v), 0.)

    # Matches analytical solution within 0.1%
    assert np.abs((v[0] - v_theor) / v[0]) < 0.001
    assert np.abs((T[0] - T_theor) / T[0]) < 0.001
