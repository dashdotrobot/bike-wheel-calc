import pytest
import warnings
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub


# -----------------------------------------------------------------------------
# Mass properties tests
# -----------------------------------------------------------------------------

def test_mass_rim_only():
    'Check that wheel mass returns rim mass if no spoke density is given'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9, density=1.)

    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.8e-3, young_mod=210e9, offset_lat=0.)

    # Should return a warning that some spoke densities are not specified
    with pytest.warns(UserWarning):
        m_wheel = w.calc_mass()

    assert np.allclose(m_wheel, 2*np.pi*0.3*100e-6)

def test_mass_spokes_only():
    'Check that spoke masses are correctly calculated'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9)

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset_lat=0., density=1.0)

    # Calculate mass of a single spoke
    m_spk = np.hypot(0.3 - 0.025, 0.025) * np.pi/4*(1.8e-3)**2 * 1.0

    # Should return a warning that the rim density is not specified
    with pytest.warns(UserWarning):
        m_wheel = w.calc_mass()

    assert np.allclose(m_wheel, 36.*m_spk)

def test_I_rim_only():
    'Check that wheel inertia returns rim inertia if no spoke density is given'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9, density=1.)

    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.8e-3, young_mod=210e9, offset_lat=0.)

    # Should return a warning that some spoke densities are not specified
    with pytest.warns(UserWarning):
        I_wheel = w.calc_rot_inertia()

    assert np.allclose(I_wheel, (2*np.pi*0.3*100e-6)*0.3**2)

def test_I_spokes_only():
    'Check that spoke inertias are correctly calculated'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9)

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset_lat=0., density=1.0)

    # Calculate inertia of a single spoke
    m_spk = np.hypot(0.3 - 0.025, 0.025) * np.pi/4*(1.8e-3)**2 * 1.0
    I_spk = m_spk*(0.3 - 0.025)**2/12. + m_spk*(0.5*(0.025 + 0.3))**2

    # Should return a warning that the rim density is not specified
    with pytest.warns(UserWarning):
        I_wheel = w.calc_rot_inertia()

    assert np.allclose(I_wheel, 36.*I_spk)

def test_I_wheel_less_than_max():
    'Check that rotational inertia is less than theoretical maximum'

    w = BicycleWheel()
    w.hub = Hub(diameter=0.050, width=0.05)
    w.rim = Rim(radius=0.3, area=100e-6,
                J_tor=25./26e9, I_lat=200./69e9, I_rad=100./69e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9, density=1.0)

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset_lat=0., density=1.0)

    I_wheel = w.calc_rot_inertia()

    assert I_wheel < w.calc_mass()*w.rim.radius**2
