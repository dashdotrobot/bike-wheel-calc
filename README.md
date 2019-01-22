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
wheel.spokes.append(wheel.Spoke(rim_pt, hub_pt, diameter=2.0e-3, young_mod=210e9))
...
```

The first two arguments are tuples defining the position of the spoke nipple and hub eyelet, respectively, in polar coordinates (R, theta, z). The spoke nipple does not need to lie on the rim centroid line. For example, a "fat-bike" wheel typically has spoke nipples which are offset from the centerline of the rim to provide torsional stability to the rim.

(Optional) Finally, apply spoke tension

```python
wheel.apply_tension(800.)  # Apply 800 Newtons average radial tension.
```

### 2 Create a ModeMatrix object

The ModeMatrix class contains the methods and objects to implement the the Mode Matrix method for stress analysis of the wheel, developed by Matthew Ford in his Ph.D. thesis [[1]](#references). In this method, the deformations functions are approximated by sine and cosine functions, e.g.

```
u(t) = u_0 + SUM(u_n_c*cos(n*theta) + u_n_s*sin(n*theta))
```

where `n` goes from 1 to `N` in the `SUM`. The highest mode number `N` is chosen to achieve the desired precision. More modes requires more computational power, but results in a more precise solution.

In the Mode Matrix method, we solve for the coefficients `u_0, u_1_c, u_1_s, u_2_c, u_2_s, ...`. If the highest mode is `N`, there are a total of `8N + 4` modes: That's one coefficient for each sine/cosine, each degree of freedom, and each mode = `2*4*N`, plus 4 coefficients for `u_0, v_0, w_0, phi_0` for the zero-mode.

Create a `ModeMatrix` object as follows:

```python
mm = ModeMatrix(wheel, N=24)
```

### 3 Compute the mode stiffness matrix with the desired approximations

The mode stiffness matrix is composed of two parts: the rim stiffness matrix and the spoke stiffness matrix:

```python
K = (mm.K_rim(tension=True, r0=True) +
     mm.K_spk(tension=True, smeared_spokes=True)
```

The option `tension=True` on the rim stiffness matrix takes into account the effect of spoke tension and the compressive stress in the rim on lateral stiffness. The option `tension=True` on the spoke stiffness matrix takes into account the stiffening effect of tension on the spoke system lateral stiffness. The option `smeared_spokes=True` indicates that the wheel should be treated as if it had infinite spokes, smeared out into an effective "disc" with the same average stiffness as the original spokes. Choose `smeared_spokes=False` to get all the effects of discrete spokes.

### 4 Add forces or torques

Apply forces and torques by creating a `F_ext` object for each force, and adding together as many forces as desired.

```python
# 120 Newton lateral force and 500 Newton radial force from the road
F_contact = mm.F_ext(theta=0., f=[120., 500., 0., 0.])

# Pair of forces from rim braking
F_brake_1 = mm.F_ext(theta=0., f=[0., 0., -100., 0.])     # Tangential force at the road
F_brake_2 = mm.F_ext(theta=3.1415, f=[0., 0., 100., 0.])  # Tangential force at the rim brakes

F = F_contact + F_brake_1 + F_brake_2
```

In this scenario, we have a created a radial and lateral force at the road contact point (theta=0), and a pair of forces representing a rim brakig scenario: -100 Newtons at the road contact point and 100 Newtons at the brake pads (theta=pi)

> Note, you can also add a torque to the rim (say, if one flange of the rim is loaded) by specifying `f=[0., 0., 0., T]`.

### 5 Solve and extract results

The stiffness matrix `K`, force vector `F` and mode coefficient vector `d` satisfy the matrix equation

```
K.dot(d) = F
```

Solve for the mode coefficients `d` using

```
d = numpy.linalg.solve(K, F)
```

## References

[1] Matthew Ford, [Reinventing the Wheel: Stress Analysis, Stability, and Optimization of the Bicycle Wheel](https://github.com/dashdotrobot/phd-thesis/releases/download/v1.0/Ford_BicycleWheelThesis_v1.0.pdf), Ph.D. Thesis, Northwestern University (2018)