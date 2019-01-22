from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix
import matplotlib.pyplot as plt
import numpy as np


def calc_rad_stiff(wheel):
    'Calculate radial stiffness.'

    # Create a ModeMatrix model with 24 modes
    mm = ModeMatrix(wheel, N=24)

    # Calculate stiffness matrix
    K = mm.K_rim(tension=True) + mm.K_spk(smeared_spokes=False, tension=True)

    # Create a unit radial load pointing radially inwards at theta=0
    F_ext = mm.F_ext(0., np.array([0., 1., 0., 0.]))

    # Solve for the mode coefficients
    dm = np.linalg.solve(K, F_ext)

    return 1e-6 / mm.rim_def_rad(0., dm)[0]


def calc_lat_stiff(wheel):
    'Calculate lateral (side-load) stiffness.'

    # Create a ModeMatrix model with 24 modes
    mm = ModeMatrix(wheel, N=24)

    # Calculate stiffness matrix
    K = mm.K_rim(tension=True) + mm.K_spk(smeared_spokes=False, tension=True)

    # Create a unit lateral load at theta=0
    F_ext = mm.F_ext(0., np.array([1., 0., 0., 0.]))

    # Solve for the mode coefficients
    dm = np.linalg.solve(K, F_ext)

    return 1e-3 / mm.rim_def_lat(0., dm)[0]


def calc_rot_stiff(wheel):
    'Calculate rotational (wind-up) stiffness.'

    # Create a ModeMatrix model with 24 modes
    mm = ModeMatrix(wheel, N=24)

    # Calculate stiffness matrix
    K = mm.K_rim(tension=True) + mm.K_spk(smeared_spokes=False, tension=True)

    # Create a unit tangential load at theta=0
    F_ext = mm.F_ext(0., np.array([0., 0., 1., 0.]))

    # Solve for the mode coefficients
    dm = np.linalg.solve(K, F_ext)

    return 1e-3*np.pi/180*wheel.rim.radius / mm.rim_def_tan(0., dm)[0]


# Create 5 wheels with different lacing patterns:
# 0 = radial spokes
# 1 = 1-cross
# 2 = 2-cross ... etc
wheels = [BicycleWheel() for i in range(5)]

for i, w in enumerate(wheels):

    # Create hub
    w.hub = Hub(width=0.05, diameter=0.05)

    # Create rim
    w.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9)

    # Generate spoking pattern
    w.lace_cross(n_spokes=36, n_cross=i, diameter=2.0e-3, young_mod=210e9)

    # Add some spoke tension
    w.apply_tension(800.)


# Radial stiffness is the ratio of a radial force applied to the rim, to the
# resulting displacement.
stiff_rad = [calc_rad_stiff(w) for w in wheels]  # Units: [kN/mm]

# Lateral stiffness is the ratio of a force applied to the rim parallel to
# the axle, to the resulting displacement.
stiff_lat = [calc_lat_stiff(w) for w in wheels]  # Units: [N/mm]

# Rotational stiffness is the ratio of applied torque to hub twist in degrees.
# This can be calculated by calculating the tangential displacement for a given
# tangential load, and converting to degrees of rotation
stiff_rot = [calc_rot_stiff(w) for w in wheels]  # Units: [kN / degree]

# Print a table of stiffnesses
print('\n\n')
print('wheel | rotational [N-m/degree] | radial [N/m] | lateral [N/m]')
print('--------------------------------------------------------------')
for i in range(len(wheels)):
    print('  {i:2d}          {r:5.3e}             {rad:5.3e}      {lat:4.3e}'
          .format(i=i, r=stiff_rot[i], rad=stiff_rad[i], lat=stiff_lat[i]))

# Create bar graphs of stiffnesses
fig, ax = plt.subplots(nrows=3, figsize=(5, 7))

ax[0].bar(range(len(stiff_rot)), stiff_rot)
ax[0].set_title('Rotational stiffness')
ax[0].set_ylabel('[kN/degree]')

ax[1].bar(range(len(stiff_rad)), stiff_rad)
ax[1].set_title('Radial stiffness')
ax[1].set_ylabel('[kN/mm]')

ax[2].bar(range(len(stiff_lat)), stiff_lat)
ax[2].set_title('Lateral stiffness')
ax[2].set_ylabel('[N/mm]')
ax[2].set_xlabel('Spoke crossings')

plt.tight_layout()
plt.show()
