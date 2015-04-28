#!/usr/bin/env python

from wheelfem import *
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


flange_height = np.linspace(0.01, 0.1,10)

for f in flange_height:

    # Create FEM model
    geom.d1_hub = f
    geom.d2_hub = f
    fem = BicycleWheelFEM(geom, r_sec, s_sec)

    # Rigid body to constrain hub nodes
    R1 = RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    fem.add_rigid_body(R1)

    fem.add_constraint(R1.node_id, range(6))  # Fix hub
    fem.add_constraint(0, 0, 1)

    soln = fem.solve()
    # print(soln.nodal_rxn)
    # print(soln.dof_rxn)

    i_rxn = soln.dof_rxn.index(0)

    pp.plot(f, soln.nodal_rxn[i_rxn],'ro')


pp.show()