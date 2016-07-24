import bikewheelcalc as bc
import matplotlib.pyplot as pp


# Create an example wheel
wheel = bc.BicycleWheel()
wheel.hub = wheel.Hub(diam1=0.04, width1=0.03)
wheel.rim = wheel.Rim.general(radius=0.3,
                              area=82.0e-6,
                              I11=5620e-12,
                              I22=1187e-12,
                              I33=1124e-12,
                              Iw=0.0,
                              young_mod=69.0e9,
                              shear_mod=26.0e9)

wheel.lace_cross(n_spokes=36, n_cross=3, diameter=2.0e-3,
                 young_mod=210e9, offset=0.0)

# Create finite-element model from wheel
fem = bc.BicycleWheelFEM(wheel, verbose=True)

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
f_def = soln.plot_deformed_wheel(rel_scale=0.1)
pp.axis('off')

f2 = pp.figure(2)
soln.plot_spoke_tension(fig=f2)
f2.gca().set_yticklabels([])

pp.show()
