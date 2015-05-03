# bike-wheel-fem
Finite-element stress analysis for bicycle wheel implemented in Python and NumPy

bike-wheel-fem is a Python module for calculating the stresses and strains in a
wire (thin-spoked) wheel. I am making every effort to make the code usable and
self- explanatory to bike enthusiasts and mechanics, regardless of background in
stress analysis or numerical methods.

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

## Usage



## The Model

The code is based on an object-oriented framework. The class hierarchy for a typical wheel calculation is as follows:

* BicycleWheelFEM - Finite-element solver
  * BicycleWheelGeom
