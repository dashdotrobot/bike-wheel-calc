from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix
import matplotlib.pyplot as plt
import numpy as np


# Create an example wheel and rim
wheel = BicycleWheel()
wheel.hub = Hub(diameter=0.050, width=0.05)
wheel.rim = Rim(radius=0.3, area=100e-6,
                I_lat=200./69e9, I_rad=100./69e9, J_tor=25./26e9, I_warp=0.0,
                young_mod=69e9, shear_mod=26e9)

diam_flange = np.linspace(0.01, 0.1, 10)
rot_stiff = []

for d in diam_flange:

    # Create hub and spokes for each flange diameter
    wheel.hub = Hub(width=0.025, diameter=d)
    wheel.lace_cross(n_spokes=36, n_cross=3, diameter=2.0e-3, young_mod=210e9)

    # Create a ModeMatrix model with 24 modes
    mm = ModeMatrix(wheel, N=24)

    # Create a unit tangential force
    F_ext = mm.F_ext(0., np.array([0., 0., 1., 0.]))

    # Calculate stiffness matrix
    K = mm.K_rim(tension=False) + mm.K_spk(smeared_spokes=True, tension=False)

    # Solve for the mode coefficients
    dm = np.linalg.solve(K, F_ext)

    # Calculate the rotational stiffness
    rot_stiff.append(np.pi/180*wheel.rim.radius/mm.rim_def_tan(0., dm)[0])

plt.plot(diam_flange * 100, rot_stiff, 'ro')
plt.xlabel('Flange diameter [cm]')
plt.ylabel('Rotationoal stiffness [N / degree]')

print('Flange diam [cm] | Rot. stiffness [N/degree]')
print('----------------------------------------------------')
for d, r in zip(diam_flange, rot_stiff):
    print('{0:11.1f}      | {1:4.3e}'.format(d*100, r))

plt.tight_layout()
plt.show()
