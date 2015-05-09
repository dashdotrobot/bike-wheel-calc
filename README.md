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

### 2 Define spoke and rim material properties

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
