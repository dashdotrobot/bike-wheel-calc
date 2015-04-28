#!/usr/bin/env python

from bikewheelfem import *
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

d_rim = 0.6
d_hub = 0.04

wheel_file = 'wheel_36x3.txt'

geom = BicycleWheelGeom(wheel_file=wheel_file)

r_sec = RimSection(82.0e-6,     # cross-sectional area
                   5620.0e-12,  # I11 (torsion)
                   1187.0e-12,  # I22 (out-of-plane bending, wobble)
                   1124.0e-12,  # I33 (in-plane bending, squish)
                   69.0e9,      # Young's modulus - aluminum
                   26.0e9)      # shear modulus

s_sec = SpokeSection(2.0e-3,   # diameter
                     210e9)    # Young's modulus - steel

fem = BicycleWheelFEM(geom, r_sec, s_sec)

R1 = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
fem.add_rigid_body(R1)

fem.add_constraint(R1.node_id, range(6))  # Fix hub
fem.add_force(0,1,500)
fem.add_force(2,1,250)
fem.add_force(35,1,250)

soln = fem.solve()

f1 = pp.figure(1)

f_def = soln.plot_deformed_wheel()
pp.axis('off')

pp.savefig('def.png')

f2 = pp.figure(2)
ind_shift = (np.arange(len(soln.spoke_t)) + 18) % 36
print(ind_shift)
pp.bar(range(len(soln.spoke_t)), soln.spoke_t[ind_shift])

pp.xlabel('spoke number')
pp.ylabel('spoke tension')

pp.savefig('spoke_t.png')

pp.show()