import pytest
import sys
import numpy as np
from bikewheelcalc import *


# -----------------------------------------------------------------------------
# Stiffness matrixtests
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
    w.rim.sec_params['y_s'] = 0.001
    w.apply_tension(0.)

    mm = ModeMatrix(w, N=36)

    K = (mm.K_rim(tension=False, r0=True) +
         mm.K_spk(tension=False, smeared_spokes=True))

    # Symmetric
    assert np.allclose(K, K.transpose())

    # Positive eigenvalues
    assert np.all(np.linalg.eigvals(K) > 0.)

def test_K_lat(std_ncross):
    'Check that ModeMatrix and Eqn. (2.71) give same result'

    w = std_ncross(0)
    w.rim.sec_params['y_s'] = 0.
    w.apply_tension(100.)

    def get_Kn_from_K(KK, n):
        Ksub = KK[2+4*(n-1):2+4*(n-1)+4, 2+4*(n-1):2+4*(n-1)+4]
        k = np.linalg.solve(Ksub, np.array([1., 0., 0., 0.]))

        return 1./k[0]


    # Analytical solution
    R = w.rim.radius
    EI = w.rim.young_mod*w.rim.I22
    GJ = w.rim.shear_mod*w.rim.I11

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
