import numpy as np
from scipy.interpolate import interp1d


class Piecewise:
    def __init__(self, t, v):
        self.t = t
        self.v = v

    def __call__(self, t):
        return np.interp(t, self.t, self.v)


class Polynomial:

    def __init__(self, coeffs):
        self.coeffs = coeffs

    @classmethod
    def from_fit(
        cls, t, v, n, t0=None, v0=None, t1=None, v1=None, t2=None, v2=None
    ):
        """Fit best polynomial of degree n to data points (t, v) and optional fixed point (t0, v0) and derivative (t1, v1) and second derivative (t2, v2)"""
        coeffs = np.polyfit(t, v, n)
        return cls(coeffs)

    def __call__(self, x):
        return sum(c * x**i for i, c in enumerate(self.coeffs))


class Spline:

    def __init__(self, spline):
        self.spline = spline

    @classmethod
    def from_fit(cls, t, v):
        spline = interp1d(t, v, kind="cubic")
        return cls(spline)

    def __call__(self, t):
        self.spline(t)


def solveconst(A, b, C, d):
    """Solve constrained least square problem using Lagrange multipliers
    min(|| A x - b ||_2) and Cx=b
    x: n
    A: m x n
    b: m
    C: l x n
    d: l

    L = x A.T A x - 2 A.T x + b.T b + l C x
    Equivalent to:
    (A.T A   C.T )  (x)  = (A.T b)
    (C       0   )  (l)  = (d)
    """
    nl, nx = C.shape
    m = np.hstack([np.dot(A.T, A), C.T])
    m = np.vstack([m, np.hstack([C, np.zeros((nl, nl))])])
    n = np.hstack([np.dot(A.T, b), d])
    sol = np.linalg.solve(m, n)
    return sol[:nx]


def makeA(x, N):
    return np.column_stack([x**i for i in range(N + 1)])


def makeAp(x, N):
    return np.column_stack(
        [np.zeros(len(x))] + [i * x ** (i - 1) for i in range(1, N + 1)]
    )


def makeApp(x, N):
    z = [np.zeros(len(x))] * 2
    z += [i * (i - 1) * x ** (i - 2) for i in range(2, N + 1)]
    return np.column_stack(z)


def poly_val(p, x):
    return np.sum([p[i] * x**i for i in range(len(p))], axis=0)


def poly_print(p, x="x", power="**", mul="*"):
    res = ["%.10e" % p[0]]
    if len(p) > 1:
        res.append("%+.10e%s%s" % (p[1], mul, x))
    for i in range(2, len(p)):
        res.append("%+.10e%s%s%s%d" % (p[i], mul, x, power, i))
    return "".join(res)


def poly_fit(
    order, xdata, ydata, x0=[], y0=[], xp0=[], yp0=[], xpp0=[], ypp0=[]
):
    A = makeA(xdata, order)
    b = ydata
    C0 = makeA(np.array(x0), order)
    C1 = makeAp(np.array(xp0), order)
    C2 = makeApp(np.array(xpp0), order)
    C = np.vstack([C0, C1, C2])
    d = np.hstack([y0, yp0, ypp0])
    p = solveconst(A, b, C, d)
    return p
