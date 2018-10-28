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
                    I11=25./26e9, I22=200./69e9, I33=100./69e9, Iw=0.0,
                    young_mod=69e9, shear_mod=26e9)


        w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset=0.)

        return w

    return _build_wheel
