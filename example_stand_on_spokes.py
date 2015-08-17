#!/usr/bin/env python

import bikewheelcalc as bc
import matplotlib.pyplot as pp


# Initialize wheel geometry from wheel files
geom_36x3 = bc.WheelGeometry(wheel_file='wheel_36_x3.txt')

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


# Create finite-element model from wheel
fem = bc.BicycleWheelFEM(geom_36x3, r_sec, s_sec)

# Create a rigid body for the hub nodes and constrain it
R1 = bc.RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
fem.add_rigid_body(R1)
fem.add_constraint(R1.node_id, range(6))

# Add an upward force, distributed over bottom 3 nodes
fem.add_force(0, 1, 500)
fem.add_force(1, 1, 250)
fem.add_force(35, 1, 250)

soln = fem.solve(pretension=1000)

# Draw deformed wheel
f1 = pp.figure(1)
f_def = soln.plot_deformed_wheel(scale_rad=0.1)
pp.axis('off')
pp.savefig('def_36x3.png')

# Plot spoke tension
f2 = pp.figure(2)
soln.plot_spoke_tension(fig=f2)
f2.gca().set_yticklabels([])

pp.show()
