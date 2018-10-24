import pytest
from bikewheelcalc import BicycleWheel, Rim, Hub


# -------------------------------------------------------------------------------
# Test fixtures
#------------------------------------------------------------------------------
@pytest.fixture
def std_radial():
    'Return a Standard Bicycle Wheel with radial spokes'

    w = BicycleWheel()
    w.hub = Hub(diam1=0.050, width1=0.025)
    w.rim = Rim(radius=0.3, area=100e-6,
                I11=25., I22=200., I33=100., Iw=0.0,
                young_mod=69e9, shear_mod=26e9)

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset=0.)

@pytest.fixture
def std_3cross():
    'Return a Standard Bicycle Wheel with 3-cross spokes'

    w = BicycleWheel()
    w.hub = Hub(diam1=0.050, width1=0.025)
    w.rim = Rim(radius=0.3, area=100e-6,
                I11=25., I22=200., I33=100., Iw=0.0,
                young_mod=69e9, shear_mod=26e9)

    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.8e-3, young_mod=210e9, offset=0.)


# -----------------------------------------------------------------------------
# Hub tests
# -----------------------------------------------------------------------------
def test_hub_symm():
    'Initialize a symmetric hub using flange diameter and width'

    h = Hub(diameter=0.05, width=0.05)

    assert h.width_left == 0.025
    assert h.width_right == 0.025
    assert h.diameter_left == 0.05
    assert h.diameter_right == 0.05

def test_hub_asymm():
    'Initialize an asymmetric hub using two explicit diameters and widths'

    h = Hub(diameter_left=0.04, diameter_right=0.06, width_left=0.03, width_right=0.02)

    assert h.width_left == 0.03
    assert h.width_right == 0.02
    assert h.diameter_left == 0.04
    assert h.diameter_right == 0.06

def test_hub_asymm_offset():
    'Initialize an asymmetric hub using a width and an offset'

    h = Hub(diameter=0.05, width=0.05, offset=0.01)

    assert h.width_left == 0.035
    assert h.widtH_right == 0.015
