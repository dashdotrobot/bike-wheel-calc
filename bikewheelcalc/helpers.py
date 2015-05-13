import numpy as np
from scipy import interpolate

def skew_symm(v):
    'Create a skew-symmetric tensor V from vector v such that V*u = v cross u.'

    return np.matrix([[0,     v[2], -v[1]],
                      [-v[2], 0,     v[0]],
                      [v[1], -v[0],  0]])


def interp_periodic(x, y, xx, period=2*np.pi):
    'Interpolate a periodic function with cubic spline matching slope at ends.'

    # Pad data by wrapping beginning and end to match derivatives
    x_pad = np.concatenate(([x[-1] - period], x, [x[0] + period, x[1] + period]))
    y_pad = np.concatenate(([y[-1]], y, [y[0], y[1]]))

    f_spline = interpolate.splrep(x_pad, y_pad)

    return interpolate.splev(xx, f_spline)

