# bike-wheel-calc
Finite-element stress analysis for bicycle wheel implemented in Python and NumPy

bike-wheel-calc is a Python module for calculating the stresses and strains in a
wire (thin-spoked) wheel. I am making every effort to make the code usable and
self-explanatory to bike enthusiasts and mechanics, regardless of background in
stress analysis or numerical methods.

## Overview

**bike-wheel-calc** is a flexible, open-source, 3-dimensional finite-element solver for calculating the stresses and deformations of a thin-spoked bicycle wheel. By changing the geometry and material properties of the rim and spokes, and the spoke lacing pattern, almost any bicycle wheel can be simulated. The wheel can be arbitrarily constrained to represent real-world scenarios, and arbitrary forces and torques can be applied to points on the rim or hub.

Some example calculations that can be performed with bike-wheel-fem are:
* calculating change in spoke tension under rider weight, acceleration, braking, etc
* calculating the wheel stiffness, including rotational, lateral, and radial
* determining the effect of breaking a spoke
* comparing the performance of different spoke lacing patterns
* comparing the response of drive-side vs. non-drive-side spokes

Although some elementary visualization routines are included in the FEMSolution class, this package is meant for computation, not for advanced visualization, plotting, or data post-processing. The code merely computes the displacements, rotations, and stresses in each component. The code can easily be extended by writing custom visualization or post-processing routines.

## Requirements

* Python 2.7.x or later [(www.python.org)](www.python.org)
* NumPy [(www.numpy.org)](www.numpy.org)
* SciPy [(www.scipy.org)](www.scipy.org)
* matplotlib [(www.matplotlib.org)](www.matplotlib.org) - required for plotting and visualization

All of these packages and more are included in the popular [Anaconda distribution](https://store.continuum.io/cshop/anaconda/). 

## Installation

Download all the files into one directory. You can put the folder bikewheelcalc on your module path, or keep it in the same directory as where you write your Python scripts.

## Usage

To perform a typical computation, you should (1) define the wheel properties, (2) define the spoke and rim material properties, (3) add constraints ("clamp" or "fix" the wheel at some points), (4) add forces or torques, and finally, (5) solve!

### 1 Defining wheel geometry and material properties

A BicycleWheel objects defines the geometry, material properties, and spoke arrangement of the wheel. First, create an empty wheel object:

```python
wheel = BicycleWheel()
```

Next, define the hub using the subclass Hub:

```python
wheel.hub = wheel.Hub(diam1=0.04, width1=0.03)
```

You can specify a different flange diameter and width for the non-drive side by specifying the optional diam2 and width2 keywords. All dimensions are in meters (30 millimeters = 0.03 meters).

Next, define the rim using one of the available constructors. The most general constructor is the general() constructor."

```python
wheel.rim = wheel.Rim.general(radius=0.3,
                              area=82.0e-6,
                              I11=5620e-12,
                              I22=1187e-12,
                              I33=1124e-12,
                              Iw=0.0,
                              young_mod=69.0e9,
                              shear_mod=26.0e9)
```

`radius` is the radius of the rim centroid (the geometric center of the rim cross-section), `area` is the cross-sectional area, `I11` is the [torsion constant](https://en.wikipedia.org/wiki/Torsion_constant), `I22` is the [second moment of area](https://en.wikipedia.org/wiki/Second_moment_of_area) for out-of-plane bending, `I33` is the second moment of area for in-plane bending, `Iw` is the warping constant (set to zero if you are unsure), `young_mod` is the rim material [Young's Modulus](http://en.wikipedia.org/wiki/Young%27s_modulus), and `shear_mod` is the rim material [Shear Modulus](http://en.wikipedia.org/wiki/Shear_modulus).

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
