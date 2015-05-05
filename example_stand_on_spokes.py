#!/usr/bin/env python

from classes.bikewheelfem import *
from classes.wheelgeometry import WheelGeometry
import matplotlib.pyplot as pp


# Initialize wheel geometry from wheel files
geom_36x3 = WheelGeometry(wheel_file='wheel_36_x3.txt')
geom_crow = WheelGeometry(wheel_file='wheel_36_crowsfoot.txt')

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


# --- 36-spoke cross-3 lacing --------------------------------
fem = BicycleWheelFEM(geom_36x3, r_sec, s_sec)

# Create a rigid body for the hub nodes and constrain it
R1 = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
fem.add_rigid_body(R1)
fem.add_constraint(R1.node_id, range(6))

# Add an upward force, distributed over bottom 3 nodes
fem.add_force(0, 1, 500)
fem.add_force(1, 1, 250)
fem.add_force(35, 1, 250)

soln = fem.solve()

# Draw deformed wheel
f1 = pp.figure(1)
f_def = soln.plot_deformed_wheel(scale_rad=0.05)
pp.axis('off')
pp.savefig('def_36x3.png')

# Plot spoke tension
f2 = pp.figure(2)

# Move bars so loaded spokes appear in center
ind_shift = (np.arange(len(soln.spokes_t)) + 18) % 36
pp.bar(range(len(soln.spokes_t)), soln.spokes_t[ind_shift])
pp.xlabel('spoke number')
pp.ylabel('change in spoke tension [Newtons]')
pp.savefig('spoke_t_36x3.png')


# --- 36-spoke crows-foot lacing -----------------------------
fem = BicycleWheelFEM(geom_crow, r_sec, s_sec)

R1 = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
fem.add_rigid_body(R1)
fem.add_constraint(R1.node_id, range(6))

fem.add_force(0, 1, 500)
fem.add_force(1, 1, 250)
fem.add_force(35, 1, 250)

soln = fem.solve()

# Draw deformed wheel
f3 = pp.figure(3)
f_def = soln.plot_deformed_wheel(scale_rad=0.05)
pp.axis('off')
pp.savefig('def_crow.png')

# Plot spoke tension
f4 = pp.figure(4)

# Move bars so loaded spokes appear in center
ind_shift = (np.arange(len(soln.spokes_t)) + 18) % 36
pp.bar(range(len(soln.spokes_t)), soln.spokes_t[ind_shift])
pp.xlabel('spoke number')
pp.ylabel('change in spoke tension [Newtons]')
pp.savefig('spoke_t_crow.png')

pp.show()
