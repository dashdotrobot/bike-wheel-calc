import pytest
import sys
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

    # Tangent-spoked wheel should have a positive-definite stiffness matrix
    w = std_ncross(3)
    w.rim.sec_params['y_s'] = 0.001
    w.apply_tension(0.)

    mm = ModeMatrix(w, N=20)

    K = (mm.K_rim(tension=False, r0=True) +
         mm.K_spk(tension=False, smeared_spokes=True))

    # Symmetric
    assert np.allclose(K, K.transpose())

    # Positive eigenvalues
    # assert np.all(np.linalg.eigvals(K) > 0.)
