#!/usr/bin/env python

from classes.bikewheelfem import *
from classes.wheelgeometry import WheelGeometry
import matplotlib.pyplot as pp


# Initialize wheel geometry from wheel files
geom = WheelGeometry(wheel_file='wheel_36_x3.txt')

# Rim section and material properties
r_sec = RimSection(area=82.0e-6,      # cross-sectional area
                   I11=5620.0e-12,    # area moment of inertia (twist)
                   I22=1187.0e-12,    # area moment of inertia (wobble)
                   I33=1124.0e-12,    # area moment of inertia (squish)
                   young_mod=69.0e9,  # Young's modulus - aluminum
                   shear_mod=26.0e9)  # shear modulus - aluminum

# spoke section and material properties
s_sec = SpokeSection(2.0e-3,  # spoke diameter
                     210e9)   # Young's modulus - steel


diam_flange = np.linspace(0.01, 0.1, 10)

rot_stiff = []

for d in diam_flange:

    # Create FEM model
    geom.d1_hub = d
    geom.d2_hub = d
    fem = BicycleWheelFEM(geom, r_sec, s_sec)

    # Rigid body to constrain hub nodes
    r_hub = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    r_rim = RigidBody('rim', [0, 0, 0], fem.get_rim_nodes())
    fem.add_rigid_body(r_hub)
    fem.add_rigid_body(r_rim)

    fem.add_constraint(r_rim.node_id, range(6))   # fix rim
    fem.add_constraint(r_hub.node_id, [2, 3, 4])  # fix hub z, roll, and yaw

    fem.add_constraint(r_hub.node_id, 5, np.pi/180)  # rotate by 1 degree

    soln = fem.solve()
    print(soln.nodal_rxn)
    print(soln.dof_rxn)

    i_rxn = 9  # index of reaction torque on rim

    rot_stiff.append(soln.nodal_rxn[i_rxn])

    pp.plot(d*100, soln.nodal_rxn[i_rxn], 'ro')

pp.xlabel('flange diameter [cm]')
pp.ylabel('wind-up stiffness [N-m / degree]')

pp.show()

print(rot_stiff)