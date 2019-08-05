import pytest
import warnings
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub


# -----------------------------------------------------------------------------
# Wheel tension tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize('n_cross', [0, 1, 2, 3])
def test_apply_tension_T_avg(std_ncross, n_cross):
    'Test that the apply_tension() method gives the correct T_avg'

    w = std_ncross(n_cross)

    # Make hub asymmetric
    w.hub.width_left = 0.03
    w.hub.width_right = 0.02
    w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset_lat=0.)

    w.apply_tension(T_avg=100.0)

    Tavg = np.sum([s.n[1]*s.tension for s in w.spokes]) / len(w.spokes)

    assert np.allclose(Tavg, 100.0)
    assert np.allclose(np.sum([s.n[0]*s.tension for s in w.spokes[::2]]),
                       -np.sum([s.n[0]*s.tension for s in w.spokes[1::2]]))

@pytest.mark.parametrize('n_cross', [0, 1, 2, 3])
def test_apply_tension_right(std_ncross, n_cross):
    'Test that the apply_tension() method works with T_right'

    w = std_ncross(n_cross)

    # Make hub asymmetric
    w.hub.width_left = 0.03
    w.hub.width_right = 0.02
    w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset_lat=0.)

    w.apply_tension(T_right=100.0)

    assert np.allclose([s.tension for s in w.spokes[1::2]], 100.0)
    assert np.allclose(np.sum([s.n[0]*s.tension for s in w.spokes[::2]]),
                       -np.sum([s.n[0]*s.tension for s in w.spokes[1::2]]))

@pytest.mark.parametrize('n_cross', [0, 1, 2, 3])
def test_apply_tension_left(std_ncross, n_cross):
    'Test that the apply_tension() method works with T_left'

    w = std_ncross(n_cross)

    # Make hub asymmetric
    w.hub.width_left = 0.03
    w.hub.width_right = 0.02
    w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset_lat=0.)

    w.apply_tension(T_left=100.0)

    assert np.allclose([s.tension for s in w.spokes[0::2]], 100.0)
    assert np.allclose(np.sum([s.n[0]*s.tension for s in w.spokes[::2]]),
                       -np.sum([s.n[0]*s.tension for s in w.spokes[1::2]]))

def test_apply_tension_none(std_ncross):
    'Test that the apply_tension() method throws error with no arguments'

    w = std_ncross(1)

    with pytest.raises(TypeError):
        w.apply_tension()
