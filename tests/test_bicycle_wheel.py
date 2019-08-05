import pytest
import warnings
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub


# -----------------------------------------------------------------------------
# Hub tests
# -----------------------------------------------------------------------------

def test_hub_symm():
    'Initialize a symmetric hub using flange diameter and width'

    h = Hub(diameter=0.05, width=0.05)

    assert h.diameter_nds == 0.05
    assert h.diameter_ds == 0.05
    assert np.allclose(h.width_nds, 0.025)
    assert np.allclose(h.width_ds, 0.025)

def test_hub_asymm():
    'Initialize an asymmetric hub using two explicit diameters and widths'

    h = Hub(diameter_nds=0.04, diameter_ds=0.06, width_nds=0.03, width_ds=0.02)

    assert h.diameter_nds == 0.04
    assert h.diameter_ds == 0.06
    assert h.width_nds == 0.03
    assert h.width_ds == 0.02

def test_hub_asymm_offset():
    'Initialize an asymmetric hub using a width and an offset'

    h = Hub(diameter=0.05, width=0.05, offset=0.01)

    assert np.allclose(h.width_nds, 0.035)
    assert np.allclose(h.width_ds, 0.015)


# -----------------------------------------------------------------------------
# Spoke tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize('n_cross', [0, 1, 2, 3])
def test_calc_k(std_ncross, n_cross):
    'Check that calc_k() works properly for each spoke'

    w = std_ncross(n_cross)
    w.apply_tension(100.)

    s = w.spokes[0]
    k = s.calc_k(tension=True)

    k_EA_theor = 210e9*np.pi/4*(1.8e-3)**2 / s.length
    k_T_theor = s.tension / s.length

    # Material stiffness
    d_matl = np.append(s.n, 0.)
    dF_matl = k.dot(d_matl)

    assert np.allclose(np.cross(dF_matl[:3], s.n), 0.)
    assert np.allclose(np.dot(dF_matl[:3], s.n), k_EA_theor)

    # Geometric stiffness
    d_geom = np.append(np.array([1., 0., 0.]) -
                       np.dot(np.array([1., 0., 0.]), s.n)*s.n,
                       0.)
    d_geom = d_geom / np.sqrt(d_geom[0]**2 + d_geom[1]**2 + d_geom[2]**2)
    dF_geom = k.dot(d_geom)

    assert np.allclose(np.dot(dF_geom[:3], s.n), 0.)
    assert np.allclose(np.sqrt(np.dot(dF_geom, dF_geom)), k_T_theor)

def test_calc_k_geom(std_ncross):
    'Check that calc_k() and calc_k_geom() are consistent'

    w = std_ncross(0)
    w.apply_tension(100.)

    s = w.spokes[0]

    assert np.allclose(s.calc_k(tension=True),
                       s.calc_k(tension=False) + s.tension*s.calc_k_geom())

def test_calc_kbar_geom(std_ncross):
    'Check that calc_kbar() and calc_kbar_geom() are consistent'

    w = std_ncross(0)
    w.apply_tension(100.)

    assert np.allclose(w.calc_kbar(tension=True),
                       w.calc_kbar(tension=False) + 100.*w.calc_kbar_geom())

@pytest.mark.parametrize('n_cross', [0, 1, 2, 3])
def test_calc_kbar_symm_nooffset(std_ncross, n_cross):
    'Compare kbar for radial spokes against theory'

    w = std_ncross(n_cross)

    c1, c2, c3 = w.spokes[0].n
    l = w.spokes[0].length
    Ks = 210e9*np.pi/4*(1.8e-3)**2 / w.spokes[0].length
    R = 0.3
    ns = 36

    kbar_theor = ns*Ks/(2*np.pi*R) * np.diag([c1**2, c2**2, c3**2, 0.])

    kbar = w.calc_kbar(tension=False)

    assert np.allclose(kbar, kbar_theor)

@pytest.mark.parametrize('offset', [-0.05, -0.01, 0., 0.01, 0.05])
def test_calc_kbar_offset_zero_eig(std_ncross, offset):
    'Check that u-phi submatrix has a zero eigenvalue'

    w = std_ncross(1)
    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset=offset)

    kbar = w.calc_kbar(tension=False)
    w, v = np.linalg.eig(kbar[np.ix_([0, 3], [0, 3])])

    assert np.allclose(np.min(w), 0.)

def test_calc_tension_change(std_ncross):
    'Check basic properties of Spoke.calc_tension_change()'

    w = std_ncross(3)
    s = w.spokes[5]  # random spoke number

    # Tighten spoke with fixed ends
    dT = s.calc_tension_change([0., 0., 0., 1.], a=0.001)
    assert np.allclose(dT, s.EA/s.length*0.001)

    # Check tension change for 1% extension
    dT = s.calc_tension_change(-0.01*s.length*s.n)
    assert np.allclose(dT, 0.01*s.EA)

    # No tension change for displacement perpendicular to spoke vector
    u = 0.001*np.cross([0., 0., 1.], s.n)
    dT = s.calc_tension_change(u)
    assert np.allclose(dT, 0.)


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
    w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset=0.)

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
    w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset=0.)

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
    w.lace_cross(n_spokes=36, n_cross=n_cross, diameter=1.8e-3, young_mod=210e9, offset=0.)

    w.apply_tension(T_left=100.0)

    assert np.allclose([s.tension for s in w.spokes[0::2]], 100.0)
    assert np.allclose(np.sum([s.n[0]*s.tension for s in w.spokes[::2]]),
                       -np.sum([s.n[0]*s.tension for s in w.spokes[1::2]]))

def test_apply_tension_none(std_ncross):
    'Test that the apply_tension() method throws error with no arguments'

    w = std_ncross(1)

    with pytest.raises(TypeError):
        w.apply_tension()


# -----------------------------------------------------------------------------
# Spoke lacing geometry tests
# -----------------------------------------------------------------------------

def test_radial_geom(std_ncross):
    'Initialize a wheel and check that the basic geometry is correct'

    w = std_ncross(0)

    # Check number of spokes
    assert len(w.spokes) == 36

    # Check spoke angle alpha
    assert np.allclose([np.dot(s.n, np.array([0., 1., 0.]))
                        for s in w.spokes],
                        (0.3 - 0.025)/np.hypot(0.3 - 0.025, 0.025))

def test_lace_cross_nds(std_no_spokes):

    w = std_no_spokes()

    w.lace_cross_nds(n_spokes=18, n_cross=3, diameter=2.0e-3, young_mod=210e9,
                     offset=0.01, offset_rad=0.01)

    # Check rim theta positions
    assert np.allclose([s.theta for s in w.spokes], np.arange(0., 2*np.pi, 2*np.pi/18.))

    # Check spoke vectors for leading spokes
    n_ll = np.array([0.025 - 0.01,
                     0.3 - 0.025*np.cos(2*np.pi/18*3) - 0.01,
                     0.025*np.sin(2*np.pi/18*3)])
    n_lt = np.array([0.025 - 0.01,
                     0.3 - 0.025*np.cos(2*np.pi/18*3) - 0.01,
                     -0.025*np.sin(2*np.pi/18*3)])

    assert np.all([np.allclose(s.n*s.length, n_ll) for s in w.spokes[::2]])
    assert np.all([np.allclose(s.n*s.length, n_lt) for s in w.spokes[1::4]])


def test_lace_cross_ds(std_no_spokes):

    w = std_no_spokes()

    w.lace_cross_ds(n_spokes=18, n_cross=3, diameter=2.0e-3, young_mod=210e9,
                    offset=0.01, offset_rad=0.01)

    # Check rim theta positions
    assert np.allclose([s.theta for s in w.spokes], np.arange(2*np.pi/36., 2*np.pi, 2*np.pi/18.))

    # Check spoke vectors for leading spokes
    n_rl = np.array([-0.025 + 0.01,
                     0.3 - 0.025*np.cos(2*np.pi/18*3) - 0.01,
                     0.025*np.sin(2*np.pi/18*3)])
    n_rt = np.array([-0.025 + 0.01,
                     0.3 - 0.025*np.cos(2*np.pi/18*3) - 0.01,
                     -0.025*np.sin(2*np.pi/18*3)])

    assert np.all([np.allclose(s.n*s.length, n_rl) for s in w.spokes[::2]])
    assert np.all([np.allclose(s.n*s.length, n_rt) for s in w.spokes[1::4]])


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

    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.8e-3, young_mod=210e9, offset=0.)

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

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset=0., density=1.0)

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

    w.lace_cross(n_spokes=36, n_cross=3, diameter=1.8e-3, young_mod=210e9, offset=0.)

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

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset=0., density=1.0)

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

    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset=0., density=1.0)

    I_wheel = w.calc_rot_inertia()

    assert I_wheel < w.calc_mass()*w.rim.radius**2
