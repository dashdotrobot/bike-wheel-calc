import pytest
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub


# -------------------------------------------------------------------------------
# Test fixtures
#------------------------------------------------------------------------------
@pytest.fixture
def std_radial():
    'Return a Standard Bicycle Wheel with radial spokes'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                I11=25., I22=200., I33=100., Iw=0.0,
                young_mod=69e9, shear_mod=26e9)

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset=0.)

    return w

@pytest.fixture
def std_3cross():
    'Return a Standard Bicycle Wheel with 3-cross spokes'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                I11=25., I22=200., I33=100., Iw=0.0,
                young_mod=69e9, shear_mod=26e9)

    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.8e-3, young_mod=210e9, offset=0.)

    return w


# -----------------------------------------------------------------------------
# Hub tests
# -----------------------------------------------------------------------------

def test_hub_symm():
    'Initialize a symmetric hub using flange diameter and width'

    h = Hub(diameter=0.05, width=0.05)

    assert h.diameter_left == 0.05
    assert h.diameter_right == 0.05
    assert np.allclose(h.width_left, 0.025)
    assert np.allclose(h.width_right, 0.025)

def test_hub_asymm():
    'Initialize an asymmetric hub using two explicit diameters and widths'

    h = Hub(diameter_left=0.04, diameter_right=0.06, width_left=0.03, width_right=0.02)

    assert h.diameter_left == 0.04
    assert h.diameter_right == 0.06
    assert h.width_left == 0.03
    assert h.width_right == 0.02

def test_hub_asymm_offset():
    'Initialize an asymmetric hub using a width and an offset'

    h = Hub(diameter=0.05, width=0.05, offset=0.01)

    assert np.allclose(h.width_left, 0.035)
    assert np.allclose(h.width_right, 0.015)

# -----------------------------------------------------------------------------
# Wheel tests
# -----------------------------------------------------------------------------

def test_radial_geom(std_radial):
    'Initialize a wheel and check that the basic geometry is correct'

    # Check number of spokes
    assert len(std_radial.spokes) == 36

    # Check spoke angle alpha
    assert np.allclose([np.dot(s.n, np.array([0., 1., 0.]))
                        for s in std_radial.spokes],
                        (0.3 - 0.025)/np.hypot(0.3 - 0.025, 0.025))

# Test calc_k() method for a single spoke
def test_calc_k(std_radial, std_3cross):
    'Check that calc_k() works properly for each spoke'

    k_EA_theor = 210e9*np.pi/4*(1.8e-3)**2 / np.hypot(0.3 - 0.025, 0.025)
    k_T_theor = 100. / np.hypot(0.3 - 0.025, 0.025)

    std_radial.apply_tension(100.)

    s = std_radial.spokes[0]
    k = s.calc_k(tension=True)

    d_matl = np.append(s.n, 0.)
    dF_matl = k.dot(d_matl)

    # Check direction and magnitude of dF_matl
    assert np.allclose(np.cross(dF_matl[:3], s.n), 0.)
    assert np.allclose(np.dot(dF_matl[:3], s.n), k_EA_theor)

    d_geom = np.append(np.array([1., 0., 0.]) -
                       np.dot(np.array([1., 0., 0.]), s.n)*s.n,
                       0.)
    dF_geom = k.dot(d_geom)

    # Check direction and magnitude of dF_geom
    assert np.allclose(np.dot(dF_geom[:3], s.n), 0.)
    assert np.allclose(np.sqrt(np.dot(dF_geom, dF_geom)), k_T_theor)

# Test calc_kbar() method for the entire wheel