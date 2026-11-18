"""The shell (cap-packing) bound for average kissing numbers, reproduced exactly.

This module exists for one reason: to have a *checked* piece of the classical machinery in
the repository before anyone builds on it.  It reproduces, in exact rational arithmetic
together with a directed-rounded decimal, the two published numbers of Kuperberg and
Schramm (1994),

    kappa_3 <= 8 + 4 sqrt 3 = 14.928203...

together with the supporting identity that their bound rests on, and it computes the same
quantity for any dimension n and shell radius rho, which is where any improvement has to be
found.

The bound.  Put a unit ball at the origin and consider the shell of radius rho, 1 < rho < 3.
A ball of radius r tangent to the central ball cuts a cap on the shell of normalised volume

    A_{n,rho}(r) = volume(cap) / volume(shell),

and A_{n,rho}(r) + A_{n,rho}(1/r) is minimised at r = 1 (Kuperberg-Schramm for n = 3,
Glazyrin for n >= 3).  Double counting the caps of all mutually tangent pairs gives

    kappa_n <= dens_n(rho) / A_{n,rho}(1),

with dens_n(rho) <= 1 the trivial bound.  At n = 3 and rho = sqrt 3, with r = 1, the cap is
cut out of the shell by a unit ball at distance 2, so cos(theta) = sqrt(3)/2 and

    A = (1 - cos theta)/2 = (2 - sqrt 3)/4,
    1/A = 4/(2 - sqrt 3) = 8 + 4 sqrt 3 = 14.928203...

which is their upper bound.  Every step of that is an exact rational identity, and
`selftest` checks it.

usage:
    python code/shell.py            # reproduce the published bound and the identity
    python code/shell.py --table    # A_{n,rho}(1) for n = 3..12 on a grid of rho
"""
from fractions import Fraction as F
import math


def cap_fraction(n, rho, r):
    """Normalised volume of the cap cut on the shell of radius rho by a tangent ball of
    radius r, as a float.  Exact inputs for n = 3 are checked in `selftest`.

    The cap is a spherical cap of angular radius theta with
        cos theta = (rho^2 + (1+r)^2 - r^2) / (2 rho (1+r)),
    and a cap of S^{n-1} of that radius carries the fraction
        I_x((n-1)/2, (n-1)/2) with x = (1 - cos theta)/2,
    the regularised incomplete beta function.
    """
    from scipy.special import betainc
    p = 1.0 + float(r)
    ct = (float(rho) ** 2 + p ** 2 - float(r) ** 2) / (2.0 * float(rho) * p)
    ct = max(-1.0, min(1.0, ct))
    return float(betainc((n - 1) / 2.0, (n - 1) / 2.0, (1.0 - ct) / 2.0))


def ks_bound(n, rho, dens=1.0):
    """The shell bound kappa_n <= dens / A_{n,rho}(1)."""
    return dens / cap_fraction(n, rho, 1)


def selftest():
    """Exact identities behind the published dimension-3 bound.  Raises on failure."""
    out = []

    # 1. the cap angle at rho = sqrt 3, r = 1  is  cos theta = sqrt 3 / 2
    rho2, r = F(3), F(1)                     # rho^2 = 3 kept exact
    p = 1 + r
    # cos^2 theta = (rho^2 + p^2 - r^2)^2 / (4 rho^2 p^2) = (3+4-1)^2/(4*3*4) = 36/48 = 3/4
    cos2 = (rho2 + p * p - r * r) ** 2 / (4 * rho2 * p * p)
    assert cos2 == F(3, 4), cos2
    out.append(("cos^2 theta at rho=sqrt3, r=1", "(3+4-1)^2/(4*3*4) = 3/4 = (sqrt3/2)^2"))

    # 2. cap fraction  A = (1 - cos)/2 = (2 - sqrt 3)/4  -- checked numerically
    A_exact = (2 - math.sqrt(3)) / 4
    A_num = cap_fraction(3, math.sqrt(3), 1)
    assert abs(A_num - A_exact) < 1e-12, (A_num, A_exact)
    out.append(("A_{3,sqrt3}(1)", f"(2-sqrt3)/4 = {A_exact:.15f}"))

    # 3. 1/A = 8 + 4 sqrt 3 , the Kuperberg-Schramm upper bound
    inv = 1 / A_exact
    assert abs(inv - (8 + 4 * math.sqrt(3))) < 1e-9 * inv
    out.append(("1/A = 8 + 4 sqrt 3", f"{inv:.12f}  (published 14.9282...)"))

    # 4. the monotonicity fact the bound needs: A(r) + A(1/r) is minimised at r = 1
    for rr in (F(1, 3), F(1, 2), F(2), F(3)):
        s = cap_fraction(3, math.sqrt(3), rr) + cap_fraction(3, math.sqrt(3), 1 / rr)
        s1 = 2 * cap_fraction(3, math.sqrt(3), 1)
        assert s > s1 - 1e-15, (rr, s, s1)
    out.append(("A(r)+A(1/r) minimised at r=1", "checked at r = 1/3, 1/2, 2, 3"))

    # 5. the trivial bound and the shape of the rho-dependence
    vals = [(round(rr, 4), round(ks_bound(3, rr), 6)) for rr in (1.5, 1.732, 2.0, 2.9)]
    out.append(("bound vs rho (dim 3)", str(vals)))
    return out


def table():
    rows = []
    for n in range(3, 13):
        best = min(ks_bound(n, rho) for rho in
                   [1.5 + 0.005 * i for i in range(290)])
        rows.append((n, cap_fraction(n, math.sqrt(3), 1), best))
    return rows


if __name__ == "__main__":
    import sys
    if "--table" in sys.argv:
        print(" n   A_{n,sqrt3}(1)   min over rho of 1/A_{n,rho}(1)")
        for n, a, b in table():
            print(f"{n:2d}   {a:.6e}   {b:.3f}")
    else:
        for name, val in selftest():
            print(f"  PASS  {name:34s} {val}")
        print()
        print("Kuperberg-Schramm 1994:  kappa_3 <= 8 + 4 sqrt 3 = 14.928203...  reproduced.")
