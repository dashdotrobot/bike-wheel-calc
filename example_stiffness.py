import bikewheelcalc as bc
import matplotlib.pyplot as plt
import numpy as np


# Create 5 wheels
wheels = [bc.BicycleWheel() for i in range(5)]

for i, w in enumerate(wheels):

    # Create hub
    w.hub = bc.Hub(diam1=0.04, width1=0.03)

    # Create rim
    w.rim = bc.Rim.general(radius=0.3,
                           area=82.0e-6,
                           I11=5620e-12,
                           I22=1187e-12,
                           I33=1124e-12,
                           Iw=0.0,
                           young_mod=69.0e9,
                           shear_mod=26.0e9)

    # Generate spoking pattern
    w.lace_cross(n_spokes=36, n_cross=i, diameter=2.0e-3,
                 young_mod=210e9, offset=0.0)

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

for w in wheels:

    fem = bc.BicycleWheelFEM(w)

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
    fem.add_force(0, 2, 1.0)

    soln = fem.solve()
    stiff_lat.append(1.0 / np.abs(soln.nodal_disp[0, 2]))

    # Remove forces and boundary conditions
    fem.remove_bc(range(fem.n_nodes), range(6))

    # Calculate rotational stiffness
    # Fix both hub and rim and rotate the hub
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
for i in range(len(wheels)):
    print '  {i:2d}          {r:5.3e}             {rad:5.3e}      {lat:4.3e}'\
        .format(i=i, r=stiff_rot[i], rad=stiff_rad[i], lat=stiff_lat[i])

# Create bar graphs of stiffnesses
plt.bar(range(len(stiff_rot)), stiff_rot)
plt.ylabel('Rotational stiffness [N-m / degree]')

plt.figure()
plt.bar(range(len(stiff_rad)), stiff_rad)
plt.ylabel('Radial stiffness [N-m]')

plt.figure()
plt.bar(range(len(stiff_lat)), stiff_lat)
plt.ylabel('Lateral stiffness [N-m]')

plt.show()
