# bike-wheel-fem
Finite-element stress analysis for bicycle wheel implemented in Python and NumPy

bike-wheel-fem is a Python module for calculating the stresses and strains in a
wire (thin-spoked) wheel. I am making every effort to make the code usable and
self-explanatory to bike enthusiasts and mechanics, regardless of background in
stress analysis or numerical methods.

## Overview

**bike-wheel-fem** is a flexible, open-source, 3-dimensional finite-element solver for calculating the stresses and deformations of a thin-spoked bicycle wheel. By changing the geometry and material properties of the rim and spokes, and the spoke lacing pattern, almost any bicycle wheel can be simulated. The wheel can be arbitrarily constrained to represent real-world scenarios, and arbitrary forces and torques can be applied to points on the rim or hub.

Some example calculations that can be performed with bike-wheel-fem are:
* calculating change in spoke tension under rider weight, acceleration, cornering, etc
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

Download all the files into one directory. You can put the module
bikewheelfem.py anywhere on your module path, or keep it in the same directory
as where you write python scripts.

## Usage

To perform a typical computation, you should (1) define the wheel geometry, (2) define the spoke and rim material properties, (3) add constraints ("clamp" or "fix" the wheel at some points), (4) add forces or torques, and finally, (5) solve!

### 1 Defining wheel geometry

A WheelGeometry object defines the geometry of a bicycle wheel including rim diameter, hub diameter (drive-side and non-drive-side), hub width (drive-side and non-drive-side), and spoke connections.

Wheel geometry can be defined through the constructor, or by loading a separate wheel file. Spokes can either be specified in the wheel file, or added manually.

```python
geom = WheelGeometry(wheel_file='wheel_36_x3.txt')
```

Or:

```python
geom = WheelGeometry(rim_diam=0.6, hub_diam=0.04, hub_width=0.035, n_spokes=36)
geom.add_spoke(31, 1)
geom.add_spoke(32, 2)
geom.add_spoke(9, 3)
...
```

Diameters should be specified in meters. If the `n_spokes` parameter is omitted and a wheel file is not provided, the locations of the hub eyelets and spoke nipples must be defined manually.

```python
geom = WheelGeometry(rim_diam=0.6, hub_diam=0.04, hub_width=0.035)
geom.add_nipple([0, 10, 20, 30, ...])
geom.add_eyelet([0, 10, 20, 30, ...], [1, -1, 1, -1, ...])
```

The first argument is a list of the angular positions (starting from 0 degrees = bottom center), while the second argument to `add_eyelet()` is a list of the same length specifying the side (1 = drive side, -1 = non-drive side).

### 2 Define spoke and rim material properties

The SpokeSection object defines the spoke diameter and the spoke material [Young's Modulus](http://en.wikipedia.org/wiki/Young%27s_modulus).

```python
spoke_section = SpokeSection(d_spoke=0.002, young_mod=210e9)
```

The RimSection object defines the material properties and cross-section geometric properties of the rim.

```python
rim_section = RimSection(area=82.0e-6,      # cross-sectional area
                         I11=5620.0e-12,    # area moment of inertia (twist)
                         I22=1187.0e-12,    # area moment of inertia (wobble)
                         I33=1124.0e-12,    # area moment of inertia (squish)
                         young_mod=69.0e9,  # Young's modulus - aluminum
                         shear_mod=26.0e9)  # shear modulus - aluminum
```

The [area moment of inertia](http://en.wikipedia.org/wiki/Second_moment_of_area) is a geometric property of the cross-section. It is easy to calculate for simple geometric shapes (circles, boxes, etc) but can be quite complicated for generic hollow sections, like most bicycle rims. The quantity `I22` is actually the [polar moment of inertia](http://en.wikipedia.org/wiki/Polar_moment_of_inertia), which is related to the twisting of the rim. Rim twist probably plays a smaller role in wheel deformation under most circumstances, and it may be safe to simply choose a large value (larger than `I22` or `I33`) for simple in-plane problems like rider weight, acceleration, or braking loads.

### 3 Add constraints



### 4 Add forces or torques

### 5 Solve and extract results

## The Model

The code is based on an object-oriented framework. The class hierarchy for a typical wheel calculation is as follows:

* BicycleWheelFEM - Finite-element solver
  * BicycleWheelGeom

## Contents

* LICENSE - MIT license
* README.md - this file
* example_"".py - Example scripts
* wheel_"".txt - Example wheel definition files
* classes/
 * bikewheelfem.py - BicycleWheelFEM class. Core finite-element solver routines
 * femsolution.py - FEMSolution class. Result database and post-processing / visualization methods.
 * helpers.py - utility methods
 * rigidbody.py - RigidBody class. A rigid body object constrains multiple nodes to move rigidly.
 * rimsection.py - RimSection class. Rim material and section properties
 * spokesection.py - SpokeSection class. Spoke material and section properties
 * wheelgeometry.py - WheelGeometry class. Methods for parsing wheel files
