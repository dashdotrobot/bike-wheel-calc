#!/usr/bin/env python

import bikewheelcalc as bc
import numpy as np


def calc_lat_stiff(geom, rim_sec, spoke_sec, pretension=None):

    fem = bc.BicycleWheelFEM(geom, rim_sec, spoke_sec)

    # Create a rigid body to constrain the hub nodes
    r_hub = bc.RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    fem.add_rigid_body(r_hub)

    # Calculate lateral stiffness. Apply a sideways force to the bottom node
    fem.add_constraint(r_hub.node_id, range(6))
    fem.add_force(0, 2, 1.0e-4)

    soln = fem.solve(verbose=False, pretension=pretension)
    stiff_lat = 1.0e-4 / np.abs(soln.nodal_disp[0, 2])

    return stiff_lat


def calc_rot_stiff(geom, rim_sec, spoke_sec, pretension=None):

    fem = bc.BicycleWheelFEM(geom, rim_sec, spoke_sec)

    # create rigid bodies for hub and rim
    r_hub = bc.RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    fem.add_rigid_body(r_hub)

    r_rim = bc.RigidBody('rim', [0, 0, 0], fem.get_rim_nodes())
    fem.add_rigid_body(r_rim)

    # fix rim and hub, then rotate hub
    fem.add_constraint(r_rim.node_id, range(6))             # fix rim
    fem.add_constraint(r_hub.node_id, [2, 3, 4])            # z, roll, and yaw
    fem.add_constraint(r_hub.node_id, 5, 1.0e-4*np.pi/180)  # rotate hub

    soln = fem.solve(pretension=pretension)
    stiff_rot = soln.nodal_rxn[r_rim.node_id, 5]

    return stiff_rot


def calc_rad_stiff(geom, rim_sec, spoke_sec, pretension=None):

    fem = bc.BicycleWheelFEM(geom, rim_sec, spoke_sec)

    # create a rigid body to constrain the hub nodes
    r_hub = bc.RigidBody('hub', [0, 0, 0], fem.get_hub_nodes())
    fem.add_rigid_body(r_hub)

    # constrain hub and displace bottom-most node upwards
    fem.add_constraint(r_hub.node_id, range(6))
    fem.add_constraint(0, 1, 1.0e-4)

    soln = fem.solve(verbose=False, pretension=pretension)
    stiff_rad = soln.nodal_rxn[0, 1] / 1.0e-4

    return stiff_rad
