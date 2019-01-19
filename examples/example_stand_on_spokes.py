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
K = mm.K_rim(tension=False) + mm.K_spk(smeared_spokes=True, tension=False)

# Solve for the mode coefficients
dm = np.linalg.solve(K, F_ext)

# Calculate the rotational stiffness
# Get radial deflection
theta = np.linspace(-np.pi, np.pi, 100)
rad_def = mm.rim_def_rad(theta, dm)


# Draw deformed wheel
f1 = plt.figure(1)
plt.plot(theta, 1000.*rad_def)

plt.xlabel('theta')
plt.ylabel('Radial deflection [mm]')

# f2 = plt.figure(2)
# soln.plot_spoke_tension(fig=f2)
# f2.gca().set_yticklabels([])

plt.tight_layout()
plt.show()
