from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix
import matplotlib.pyplot as plt
import numpy as np


# Create an example wheel and rim
wheel = BicycleWheel()
wheel.hub = Hub(width=0.05, diameter=0.05)
wheel.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9)
wheel.lace_cross(n_spokes=36, n_cross=3, diameter=2.0e-3, young_mod=210e9)


# Create a ModeMatrix model with 24 modes
mm = ModeMatrix(wheel, N=24)

# Create a 500 Newton pointing radially inwards at theta=0
F_ext = mm.F_ext(0., np.array([0., 500., 0., 0.]))

# Calculate stiffness matrix
K = mm.K_rim(tension=False) + mm.K_spk(smeared_spokes=False, tension=False)

# Solve for the mode coefficients
dm = np.linalg.solve(K, F_ext)

# Get radial deflection
theta = np.linspace(-np.pi, np.pi, 100)
rad_def = mm.rim_def_rad(theta, dm)

# Calculate change in spoke tensions
dT = [-s.EA/s.length *
      np.dot(s.n,
             mm.B_theta(s.rim_pt[1], comps=[0, 1, 2]).dot(dm))
      for s in wheel.spokes]


# Plot radial deformation and tension change
fig, ax = plt.subplots(nrows=2, figsize=(5, 5))

ax[0].plot(theta, 1000.*rad_def)
ax[0].set_xlim(-np.pi, np.pi)
ax[0].set_ylabel('Radial deflection [mm]')

ax[1].bar(np.arange(-np.pi, np.pi, 2*np.pi/36), np.roll(dT, 18),
		  width=1.5*np.pi/36)
ax[1].set_xlim(-np.pi, np.pi)
ax[1].set_xlabel('theta')
ax[1].set_ylabel('Change in spoke tension [N]')

ax[0].set_title('Effect of a 500 N radial load')

plt.tight_layout()
plt.show()
