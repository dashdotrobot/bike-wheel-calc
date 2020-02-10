import pytest
import warnings
import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix

# -----------------------------------------------------------------------------
# Internal force tests
# -----------------------------------------------------------------------------

def test_diametral_compression(ring_no_spokes):
    'Diametral compression of a ring with 2 radial spokes.'

    w = ring_no_spokes()

    mm = ModeMatrix(w, N=128)
    F_ext = (mm.F_ext([0.], np.array([[0., 1., 0., 0.]])) +
             mm.F_ext([np.pi], np.array([[0., 1., 0., 0.]])))
    K = (mm.K_rim(tension=False, r0=False) +
         mm.K_spk(tension=True, smeared_spokes=False))

    d = np.linalg.solve(K, F_ext)

    # Normal force at pi/2 and 3pi/2 should be -0.5
    assert np.allclose(mm.normal_force([np.pi/2, 3*np.pi/2], d),
                       -0.5, rtol=1e-3)

    # Zero shear at pi/2 and 3pi/2
    assert np.allclose(mm.shear_force_rad([np.pi/2, 3*np.pi/2], d),
                       0.)

    # Bending moment at pi/2
    assert np.allclose(mm.moment_rad(np.pi/2, d),
                       0.5 - 1./np.pi, rtol=1e-3)

    # Bending moment at pi/4
    assert np.allclose(mm.moment_rad(np.pi/4, d),
                       np.cos(np.pi/4)/2 - 1./np.pi, rtol=1e-3)

def test_four_pt_bend(ring_no_spokes):
    'Test out-of-plane forces with four-pt-bend on rim.'

    w = ring_no_spokes()

    mm = ModeMatrix(w, N=128)
    F_ext = (mm.F_ext(0., [1., 0., 0., 0.]) +
             mm.F_ext(np.pi, [1., 0., 0., 0.]) +
             mm.F_ext(np.pi/2, [-1., 0., 0., 0.]) +
             mm.F_ext(3*np.pi/2, [-1., 0., 0., 0.]))

    K = (mm.K_rim(tension=False, r0=False) +
         mm.K_spk(tension=True, smeared_spokes=True))

    d = np.linalg.solve(K, F_ext)

    # Lateral bending moment at supports (may be inaccurate due to cusp)
    assert np.allclose(mm.moment_lat([0., np.pi], d), -0.5, rtol=2)
    assert np.allclose(mm.moment_lat([np.pi/2, 3*np.pi/2], d), 0.5, rtol=2)

    # Lateral bending moment between supports
    assert np.allclose(mm.moment_lat([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4],
                                     d), 0.)

    # Lateral shear force
    assert False

    # Twisting moment
    assert False
