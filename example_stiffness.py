#!/usr/bin/env python

import bikewheelcalc as bc
import matplotlib.pyplot as pp
import numpy as np


# Initialize wheel geometry from wheel files
geom = []
geom.append(bc.WheelGeometry(wheel_file='wheel_36_x1.txt'))
geom.append(bc.WheelGeometry(wheel_file='wheel_36_x2.txt'))
geom.append(bc.WheelGeometry(wheel_file='wheel_36_x3.txt'))
geom.append(bc.WheelGeometry(wheel_file='wheel_36_x4.txt'))
geom.append(bc.WheelGeometry(wheel_file='wheel_36_crowsfoot.txt'))


# Rim section and material properties
r_sec = bc.RimSection(area=82.0e-6,      # cross-sectional area
                   I11=5620.0e-12,    # area moment of inertia (twist)
                   I22=1187.0e-12,    # area moment of inertia (wobble)
                   I33=1124.0e-12,    # area moment of inertia (squish)
                   young_mod=69.0e9,  # Young's modulus - aluminum
                   shear_mod=26.0e9)  # shear modulus - aluminum

# spoke section and material properties
s_sec = bc.SpokeSection(2.0e-3,  # spoke diameter
                     210e9)   # Young's modulus - steel

# Rotational stiffness is the ratio of applied torque to hub twist in degrees.
# This can be calculated by twisting the hub a fixed amount and measuring the
# torque. Alternatively, one could apply a fixed torque and measure the degree
# of hub twist.
stiff_rot = []  # Units: [N-m / degree]

# Lateral stiffness is the ratio of a force applied to the rim parallel to
# the axle, to the resulting displacement. The hub is rigidly clamped.
stiff_lat = []  # Units: [N/m]

# Radial stiffness is the ratio of a radial force applied to the rim, to the
# resulting displacement. The hub is rigidly clamped.
stiff_rad = []  # Units: [N/m]


for g in geom:

    fem = bc.BicycleWheelFEM(g, r_sec, s_sec)

    # Create a rigid body to constrain the hub nodes
    r_hub = bc.RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    fem.add_rigid_body(r_hub)

    # Calculate radial stiffness. Apply an upward force to the bottom node
    fem.add_constraint(r_hub.node_id, range(6))
    fem.add_force(0, 1, 1)

    soln = fem.solve()
    stiff_rad.append(1.0 / np.abs(soln.nodal_disp[0, 1]))

    # Remove forces and boundary conditions
    fem.remove_bc(range(fem.n_nodes), range(6))


    # Calculate lateral stiffness. Apply a sideways force to the bottom node
    fem.add_constraint(r_hub.node_id, range(6))
    fem.add_force(0, 2, 1)

    soln = fem.solve()
    stiff_lat.append(1.0 / np.abs(soln.nodal_disp[0, 2]))

    # Remove forces and boundary conditions
    fem.remove_bc(range(fem.n_nodes), range(6))


    # Calculate rotational stiffness. Fix both hub and rim and rotate the hub
    r_rim = bc.RigidBody('rim', [0, 0, 0], fem.get_rim_nodes())
    fem.add_rigid_body(r_rim)

    fem.add_constraint(r_rim.node_id, range(6))   # fix rim
    fem.add_constraint(r_hub.node_id, [2, 3, 4])  # fix hub z, roll, and yaw

    fem.add_constraint(r_hub.node_id, 5, np.pi/180)  # rotate by 1 degree

    soln = fem.solve() 
    stiff_rot.append(soln.nodal_rxn[r_rim.node_id, 5])

# Print a table of stiffnesses
print '\n\n'
print 'wheel | rotational [N-m/degree] | radial [N/m] | lateral [N/m]'
print '--------------------------------------------------------------'
for i in range(len(geom)):
    print '  {i:2d}          {r:5.3e}             {d:5.3e}      {l:4.3e}'.format(i=i, r=stiff_rot[i], d=stiff_rad[i], l=stiff_lat[i])

# Create bar graphs of stiffnesses
pp.bar(range(len(stiff_rot)), stiff_rot)
pp.ylabel('Rotational stiffness [N-m / degree]')

pp.figure()
pp.bar(range(len(stiff_rad)), stiff_rad)
pp.ylabel('Radial stiffness [N-m]')

pp.figure()
pp.bar(range(len(stiff_lat)), stiff_lat)
pp.ylabel('Lateral stiffness [N-m]')


pp.show()
