import numpy as np
from scipy.interpolate import interp1d

class Piecewise:
    def __init__(self, t, v):
        self.t = t
        self.v = v

    def __call__(self, t):
        return np.interp(t, self.t, self.v)

class Polynomial:
    @classmethod
    def from_fit(cls, t, v, n, t0=None, v0=None, t1=None, v1=None, t2=None, v2=None):
        """Fit best polynomial of degree n to data points (t, v) and optional fixed point (t0, v0) and derivative (t1, v1) and second derivative (t2, v2)
        """
        coeffs = np.polyfit(t, v, n)
        return cls(coeffs)

    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, x):
        return sum(c * x ** i for i, c in enumerate(self.coeffs))

class Spline:
    @classmethod
    def from_fit(cls, t, v):
        spline=interp1d(t, v, kind="cubic")
        return cls(spline)

    def __init__(self, spline):
        self.spline = spline

    def __call__(self, t):
        self.spline(t)