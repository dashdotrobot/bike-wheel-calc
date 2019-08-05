import pytest
import warnings
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub


# -----------------------------------------------------------------------------
# Hub geometry tests
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

    w.lace_cross_side(n_spokes=18, n_cross=3, side=1, diameter=2.0e-3, young_mod=210e9,
                      offset_lat=0.01, offset_rad=0.01)

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
    assert np.all([np.allclose(s.n*s.length, n_lt) for s in w.spokes[1::2]])


def test_lace_cross_ds(std_no_spokes):

    w = std_no_spokes()

    w.lace_cross_side(n_spokes=18, n_cross=3, side=-1, offset=np.pi/18, diameter=2.0e-3, young_mod=210e9,
                      offset_lat=0.01, offset_rad=0.01)

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
    assert np.all([np.allclose(s.n*s.length, n_rt) for s in w.spokes[1::2]])

