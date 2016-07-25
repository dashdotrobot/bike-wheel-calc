import bikewheelcalc as bc
import matplotlib.pyplot as pp
import numpy as np


# Create an example wheel and rim
wheel = bc.BicycleWheel()
wheel.rim = wheel.Rim.general(radius=0.3,
                              area=82.0e-6,
                              I11=5620e-12,
                              I22=1187e-12,
                              I33=1124e-12,
                              Iw=0.0,
                              young_mod=69.0e9,
                              shear_mod=26.0e9)

diam_flange = np.linspace(0.01, 0.1, 10)

rot_stiff = []

for d in diam_flange:

    # Create hub and spokes for each flange diameter
    wheel.hub = wheel.Hub(diam1=d, width1=0.03)
    wheel.lace_cross(n_spokes=36, n_cross=3, diameter=2.0e-3,
                     young_mod=210e9, offset=0.0)

    # Create FEM model
    fem = bc.BicycleWheelFEM(wheel)

    # Rigid body to constrain hub nodes
    r_hub = bc.RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    r_rim = bc.RigidBody('rim', [0, 0, 0], fem.get_rim_nodes())
    fem.add_rigid_body(r_hub)
    fem.add_rigid_body(r_rim)

    fem.add_constraint(r_rim.node_id, range(6))      # fix rim
    fem.add_constraint(r_hub.node_id, [2, 3, 4])     # fix hub z, roll, and yaw
    fem.add_constraint(r_hub.node_id, 5, np.pi/180)  # rotate by 1 degree

    soln = fem.solve()

    rot_stiff.append(soln.nodal_rxn[r_rim.node_id, 5])

pp.plot(diam_flange * 100, rot_stiff, 'ro')
pp.xlabel('flange diameter [cm]')
pp.ylabel('wind-up stiffness [N-m / degree]')

print 'flange diam [cm] | rotational stiffness [N-m/degree]'
print '----------------------------------------------------'
for d, r in zip(diam_flange, rot_stiff):
    print '{0:11.1f}      | {1:4.3e}'.format(d*100, r)

pp.show()
