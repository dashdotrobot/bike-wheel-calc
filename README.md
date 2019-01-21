# bike-wheel-calc
Stress analysis of bicycle wheels using the Mode Matrix method.

## Overview

__bike-wheel-calc__ is a Python module for simulating the quasistatic structural behavior of wire-spoked bicycle wheels. Using the `ModeMatrix object`, the deformation of the wheel can be calculated given a set of external forces applied to the rim. In addition, predefined routines in the `theory` submodule may be used to calculate the wheel stiffness, buckling tension, and other properties.

Some example calculations that can be performed with bike-wheel-calc are:
* calculating change in spoke tension under rider weight, acceleration, braking, etc
* calculating the wheel stiffness, including rotational, lateral, and radial
* determining the effect of breaking a spoke
* comparing the performance of different spoke lacing patterns
* comparing the response of drive-side vs. non-drive-side spokes

You can try out __bike-wheel-calc__ with an interactive, graphical, online app for simulating wheels based on the code [here](http://www.bicyclewheel.info).

## Installation

### With `pip`

1. Download the entire repository.
2. Open a console (with Python) and navigate to the root directory
3. Install the package using

```
pip install .
```

## Usage

### The `BicycleWheel` class

The BicycleWheel objects defines the geometry, material properties, and spoke arrangement of the wheel. First, create an empty wheel object:

```python
wheel = BicycleWheel()
```

Next, define the hub using the subclass Hub:

```python
wheel.hub = wheel.Hub(diameter=0.040, width=0.050)
```

This creates a hub with a flange diameter of 40 mm and a total width of 50 mm (flange to flange). Optionally, you can specify `diameter_ds`, `diameter_nds`, `width_ds`, and `width_nds`. to create a hub with different flange diameters and/or rim dish.

Next, define the rim.

```python
wheel.rim = Rim.general(radius=0.3,        # [m] radius at the beam axis
                        area=100.0e-6,     # [m^2] cross-sectional area of the rim
                        I_lat=1500e-12,    # [m^4] Second moment of area for lateral bending
                        I_rad=3000e-12,    # [m^4] Second moment of area for radial bending
                        J_tor=500e-12,     # [m^4] Torsion constant
                        Iw=0.0,            # [m^6] Warping constant
                        young_mod=69.0e9,  # [N/m^2] Young's modulus
                        shear_mod=26.0e9)  # [N/m^2] Shear modulus
```

The [torsion constant](https://en.wikipedia.org/wiki/Torsion_constant), and [second moments of area](https://en.wikipedia.org/wiki/Second_moment_of_area) are geometric properties of the rim cross-section that determine its stiffness. `Iw` is the warping constant (set to zero if you are unsure), `young_mod` is the rim material [Young's Modulus](http://en.wikipedia.org/wiki/Young%27s_modulus), and `shear_mod` is the rim material [Shear Modulus](http://en.wikipedia.org/wiki/Shear_modulus).

Next, create spokes by either using a predefined spoke configuration:

```python
wheel.lace_cross(n_spokes=36, n_cross=3, diameter=2.0e-3, young_mod=210e9)
```

or by defining spokes manually:

```python
rim_pt = (r_rim, theta_rim, z_rim)
hub_pt = (r_hub, theta_hub, z_hub)
d = 2.0e-3
wheel.spokes.append(wheel.Spoke(rim_pt, hub_pt, diameter=d, young_mod=210e9))
...
```

The first two arguments are tuples defining the position of the spoke nipple and hub eyelet, respectively, in polar coordinates (R, theta, z). The spoke nipple does not need to lie on the rim centroid line. For example, a "fat-bike" wheel typically has spoke nipples which are offset from the centerline of the rim to provide torsional stability to the rim.

(Optional) Finally, apply spoke tension

```python
wheel.apply_tension(800.)  # Apply 800 Newtons average radial tension.
```




### 2 Create BicycleWheelFEM object (finite-element solver)

The BicycleWheelFEM object represents the model that you are solving, including the nodes (points) and elements (spokes and rim segments), and the constraints and forces (boundary conditions).

```python
fem = BicycleWheelFEM(wheel)
```

### 3 Add constraints

By default, the hub nodes aren't connected to anything, so before applying forces, you should probably attach them all together. Do this by defining a `RigidBody` object which contains all the hub nodes. The reason this isn't done by default is to allow you to simulate more complicated effects, like the torsional flexibility of the hub in transferring torque from the sprocket side to the left side).

```python
R_hub = RigidBody(name='hub', pos=[0, 0, 0], nodes=fem.get_hub_nodes())
fem.add_rigid_body(R_hub)
```

You could also choose the nodes to constrain manually by entering a list of nodes IDs, e.g. `R_hub = RigidBody('name', pos=[0, 0, 0], nodes=[12, 13, 14, 22, 25, ...])`. The rigid body contains a special reference node at the position `pos` which you will use to add constraints or forces to the rigid body.

Before you apply forces to the wheel, you need to add constraints to specify how it is held (you can't stretch a spring without holding the other side fixed).

> **Degrees of Freedom.** Each node in the model has [6 degrees of freedom](http://en.wikipedia.org/wiki/Degrees_of_freedom_%28mechanics%29#Six_degrees_of_freedom) (displacement along the x, y, and z axes, and rotation around the x, y, and z axes). So an un-connected set of _N_ nodes would have _6N_ degrees of freedom. However, the connections between nodes (rim segments or spokes) reduce the total degrees of freedom. A properly connected finite-element model only has _6_ degrees of freedom left, so in general you must constrain 6 degrees of freedom in order to solve the model. You can do this by constraining all 6 DOFs for a single node (for example, the hub reference node) or you can constrain the displacement DOFs for 3 separate nodes.

Constrain a node using the BicycleWheelFEM object's `add_constraint()` function.

```python
# constrain the X and Y position of node 0.
fem.add_constraint(0, [0, 1])

# constrain the X, Y, and Z position, and rotation around Z of nodes 0 through 5
fem.add_constraint(range(6), [0, 1, 2, 5])

# set the x-displacement of node 5 to 1 mm
fem.add_constraint(5, 0, 0.001)

# constrain all DOFs of the hub reference node
fem.add_constraint(R_hub.node_id, range(6))
```

Note that if the third argument is omitted, the DOF will be fixed at zero (i.e. no movement or rotation). The units for displacement DOFs is meters and the units for rotational DOFs is [radians](http://en.wikipedia.org/wiki/Radian).

If your model is not properly constrained, the solver may not be able to find a valid solution.

### 4 Add forces or torques

A fully constrained model will produce a solution, but the solution may not be very interesting unless you add _forces and torques_. These forces might, for example, represent the upward force from the ground, the torque applied by the sprocket or a disc brake, or lateral forces during cornering.

> Forces and torques can only be applied to nodes. If you want to apply a force to a point on the rim in between two spokes, simply create a spoke nipple there using the `geom.add_nipple()` function, but don't connect a spoke to that point.

The numbering scheme for forces follows the numbering scheme for constraint DOFs: [0, 1, 2] are the forces along the x, y, and z directions, and [3, 4, 5] are the torques about the x, y, and z axes.

```python
# apply an upward force of 100 N to the bottom-most node
fem.add_force(0, 0, 100)

# apply a torque of 50 N-m to the hub around the axle (z-axis)
fem.add_force(R_hub.node_id, 5, 50)
```

**Note** that you can add a force to a node which has been constrained, so long as the force is applied to a degree of freedom which has not been constrained.

```python
# Hold the position of the hub fixed, but apply a twisting torque of 50 N-m
fem.add_constraint(R_hub.node_id, [0 1 2])
fem.add_force(R_hub.node_id, 5, 50)
```

### 5 Solve and extract results

Once you have created your model and added constraints and forces, solving the equations is very straightforward.

```python
solution = fem.solve()
```

The `solution` object is an instance of the class `FEMSolution` which contains all the numerical results. It also has a simple function to plot the deformed (exaggerated) shape of the wheel. The nodal displacements and rotations can be extracted from an (N x 6)-dimensional array, where N is the number of nodes.

```python
# Get the x-displacement of node 12
x_disp_12 = solution.nodal_disp[12][0]

# Get the rotation of the hub around the axle (z-axis)
hub_rot = solution.nodal_disp[R_hub.node_id][5]

# Get the total (vector) displacement of node 4
d_4 = solution.nodal_disp[4][0:3]
```

Each constrained node has a reaction force (or torque) associated with it. The reaction force is the force that would have to be applied to the node in order to keep it in the desired position. Reaction forces are only defined for degrees of freedom which have been constrained. The shape of the `nodal_rxn` array is the same size and shape as the `nodal_disp` array, but all the entries corresponding to unconstrained DOFs will be zero.

> **Note** An applied force can produce both a reaction force and a reaction torque. Imagine you are holding a fishing rod. When a fish grabs the line, you have to apply a force (pulling the fish towards you), but at the same time you have to apply a torque to keep the rod from twisting out of your hands.

## Contents

* LICENSE - MIT license
* README.md - this file
* example_"".py - Example scripts
* bikewheelcalc/
 * bicyclewheel.py - BicycleWheel class. Basic wheel and properties definition.
 * bikewheelfem.py - BicycleWheelFEM class. Core finite-element solver routines
 * femsolution.py - FEMSolution class. Result database and post-processing / visualization methods.
 * helpers.py - utility methods
 * rigidbody.py - RigidBody class. A rigid body object constrains multiple nodes to move rigidly.
