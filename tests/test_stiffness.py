import pytest
import warnings
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub


# -----------------------------------------------------------------------------
# Spoke stiffness tests
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
    w.lace_radial(n_spokes=36, diameter=1.8e-3, young_mod=210e9, offset_lat=offset)

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
