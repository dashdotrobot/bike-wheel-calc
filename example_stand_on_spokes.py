#!/usr/bin/env python

from bikewheelfem import *
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

d_rim = 0.6
d_hub = 0.04

file_36x3 = 'wheel_36_x3.txt'
file_crow = 'wheel_36_crowsfoot.txt'

geom_36x3 = BicycleWheelGeom(wheel_file=file_36x3)
geom_crow = BicycleWheelGeom(wheel_file=file_crow)

r_sec = RimSection(82.0e-6,     # cross-sectional area
                   5620.0e-12,  # I11 (torsion)
                   1187.0e-12,  # I22 (out-of-plane bending, wobble)
                   1124.0e-12,  # I33 (in-plane bending, squish)
                   69.0e9,      # Young's modulus - aluminum
                   26.0e9)      # shear modulus

s_sec = SpokeSection(2.0e-3,   # diameter
                     210e9)    # Young's modulus - steel


# --- 36-spoke cross-3 lacing --------------------------------
fem = BicycleWheelFEM(geom_36x3, r_sec, s_sec)

R1 = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
fem.add_rigid_body(R1)

fem.add_constraint(R1.node_id, range(6))  # Fix hub
fem.add_force(0,1,500)
fem.add_force(1,1,250)
fem.add_force(35,1,250)

soln = fem.solve()

# Draw deformed wheel
f1 = pp.figure(1)
f_def = soln.plot_deformed_wheel(scale_rad=0.05)
pp.axis('off')
pp.savefig('def_36x3.png')

# Plot spoke tension
f2 = pp.figure(2)

# Move bars so loaded spokes appear in center
ind_shift = (np.arange(len(soln.spoke_t)) + 18) % 36
pp.bar(range(len(soln.spoke_t)), soln.spoke_t[ind_shift])
pp.xlabel('spoke number')
pp.ylabel('change in spoke tension [Newtons]')
pp.savefig('spoke_t_36x3.png')


# --- 36-spoke crows-foot lacing -----------------------------
fem = BicycleWheelFEM(geom_crow, r_sec, s_sec)

R1 = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
fem.add_rigid_body(R1)

fem.add_constraint(R1.node_id, range(6))  # Fix hub
fem.add_force(0,1,500)
fem.add_force(1,1,250)
fem.add_force(35,1,250)

soln = fem.solve()

# Draw deformed wheel
f3 = pp.figure(3)
f_def = soln.plot_deformed_wheel(scale_rad=0.05)
pp.axis('off')
pp.savefig('def_36x3.png')

# Plot spoke tension
f4 = pp.figure(4)

# Move bars so loaded spokes appear in center
ind_shift = (np.arange(len(soln.spoke_t)) + 18) % 36
pp.bar(range(len(soln.spoke_t)), soln.spoke_t[ind_shift])
pp.xlabel('spoke number')
pp.ylabel('change in spoke tension [Newtons]')
pp.savefig('spoke_t_36x3.png')

pp.show()