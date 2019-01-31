import pytest
from bikewheelcalc import BicycleWheel, Rim, Hub


# -----------------------------------------------------------------------------
# Test fixtures
#------------------------------------------------------------------------------

@pytest.fixture
def std_ncross():
    'Define a function which returns a Standard Wheel with the specified spoke pattern'

    def _build_wheel(n_cross=0):
        w = BicycleWheel()
        w.hub = Hub(diameter=0.050, width=0.05)
        w.rim = Rim(radius=0.3, area=100e-6,
                    I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                    young_mod=69e9, shear_mod=26e9)

        w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset=0.)

        return w

    return _build_wheel

@pytest.fixture
def std_no_spokes():
    'Define a function which returns a Standard Wheel with no spokes'

    def _build_wheel():
        w = BicycleWheel()
        w.hub = Hub(diameter=0.050, width=0.05)
        w.rim = Rim(radius=0.3, area=100e-6,
                    I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                    young_mod=69e9, shear_mod=26e9)

        return w

    return _build_wheel
