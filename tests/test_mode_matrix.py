import pytest
from bikewheelcalc import *


# -----------------------------------------------------------------------------
# Stiffness matrixtests
# -----------------------------------------------------------------------------

def test_K_rim_matl_geom(std_ncross):
    'Check that _geom and _matl stiffness matrices are consistent'

    w = std_ncross(0)
    w.rim.sec_params['y_s'] = 0.001
    w.apply_tension(1000.)

    mm = ModeMatrix(w, N=20)

    K1 = (mm.K_rim(tension=True, r0=True) +
          0*mm.K_spk(tension=True, smeared_spokes=False))

    K2 = (mm.K_rim(tension=False, r0=True) -
          1000.*mm.K_rim_geom(r0=True) +
          0*mm.K_spk(tension=False, smeared_spokes=False) +
          0*1000.*mm.K_spk_geom(smeared_spokes=False))

    assert np.allclose(K1[0, 0], K2[0, 0])

def test_K_wheel(std_ncross):
    'Check properties of stiffness matrix'

    w = std_ncross(3)
    w.rim.sec_params['y_s'] = 0.001
    w.apply_tension(0.)

    mm =ModeMatrix(w, N=20)

    # K = mm.K_rim(tension=False, r0=True) + mm.K_

    assert True