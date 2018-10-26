import pytest
from bikewheelcalc import *


# -----------------------------------------------------------------------------
# Stiffness matrixtests
# -----------------------------------------------------------------------------

def test_K_matl_geom(std_ncross):
    'Check that _geom and _matl stiffness matrices are consistent'

    w = std_ncross(0)
    w.apply_tension(100.)

    mm = ModeMatrix(w, N=20)

    assert np.allclose(mm.K_rim(tension=True, r0=True) +
                       mm.K_spk(tension=True, smeared_spokes=False),
                       mm.K_rim(tension=False, r0=True) -
                       100.*mm.K_rim_geom(r0=True) +
                       mm.K_spk(tension=False, smeared_spokes=False) +
                       100.*mm.K_spk_geom(smeared_spokes=False))
