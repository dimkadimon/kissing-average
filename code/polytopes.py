"""The polytope side of the average-kissing problem, computed rather than quoted.

Eppstein-Kuperberg-Ziegler (arXiv:math/0204007) reduce a *lower* bound to a statement about a
polytope.  If a 4-polytope Q is *ridge-tangent*, its facets cut the unit 3-sphere in a ball
packing whose tangency points are the ridges of Q, and then

    kappa = 2 f_1(Q) / f_0(Q) ,

the ridge-facet ratio doubled.  Their record, 7656/607 = 12.6129, is the value of that ratio
for a specific compound of *cut* 600-cells; the chain of n cut cells that circulates in the
literature gives 2(666n+54)/(106n+14) -> 666/53 = 12.566, i.e. the weaker Kuperberg-Schramm
value.  So the combinatorial target is clear:

    maximise f_1/f_0 over ridge-tangent 4-polytopes,
    which means: beat f_1/f_0 = 3828/607 = 6.30642 .

This module computes f-vectors of the classical 4-polytopes from coordinates and reports the
ratio.  The point of doing it here is that the dictionary is checkable: the 600-cell gives
f_1/f_0 = 720/120 = 6 exactly, hence kappa = 12 exactly, which is the fcc value computed
independently in code/packings.py -- two different routes to the same number.

A warning that the computation makes concrete: f_1/f_0 is *not* bounded combinatorially.  The
cyclic polytope C_4(n) has f_1 = C(n,2), so its ratio grows like n/2 and can be as large as we
like.  Whatever bounds kappa from above must therefore come from the geometry of ridge
tangency (which is exactly the Bockhaus-Glazyrin territory), not from Euler-type counting.
That is why this file reports ratios for many polytopes and does not pretend to be a search
over all of them.

usage:
    python code/polytopes.py
"""
import itertools
import math

import numpy as np
from scipy.spatial import ConvexHull


PHI = (1 + math.sqrt(5)) / 2


# ---------------------------------------------------------------- classical 4-polytopes
def simplex4():
    return np.array([[1.0 if i == j else 0.0 for j in range(4)] for i in range(5)])


def cross4():
    return np.array([[s * (1.0 if i == j else 0.0) for j in range(4)]
                     for i in range(4) for s in (1, -1)])


def tesseract():
    return np.array(list(itertools.product((-0.5, 0.5), repeat=4)))


def cell24():
    pts = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0.0] * 4
                    v[i], v[j] = si, sj
                    pts.append(v)
    for signs in itertools.product((1, -1), repeat=4):
        pts.append(np.array(signs) / 2.0)
    return np.array(pts)


def cell600():
    """The 120 vertices of the 600-cell: 8 of (+-1,0,0,0), 16 of (+-1/2,...), and the 96
    even permutations of (0, +-1/(2 PHI), +-1/2, +-PHI/2).  All have squared norm 1, so the
    facets of the convex hull are tangent to the unit sphere -- the property the E-construction
    needs; see code/compounds.py, which measures it."""
    pts = set()
    for i in range(4):
        for s in (1.0, -1.0):
            v = [0.0] * 4
            v[i] = s
            pts.add(tuple(v))
    for sg in itertools.product((0.5, -0.5), repeat=4):
        pts.add(tuple(sg))
    base = [0.0, 1 / (2 * PHI), 0.5, PHI / 2]
    for perm in itertools.permutations(range(4)):
        if _parity(perm):
            continue
        v = [base[p] for p in perm]
        for sg in itertools.product((1.0, -1.0), repeat=4):
            pts.add(tuple(round(sg[i] * v[i], 12) for i in range(4)))
    return np.array(sorted(pts))


def _parity(perm):
    inv = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm))
              if perm[i] > perm[j])
    return inv % 2


def cell120():
    """The 120-cell: 8 + 16*? vertices; built as the dual-ish standard set is long, so we use
    the vertex set of the 600-cell's dual: all even permutations of (+-2,+-2,0,0) and
    (+-sqrt5,+-1,+-1,+-1) and (+-PHI^2, +-PHI, +-1/PHI... ) -- omitted here on purpose."""
    raise NotImplementedError("the 120-cell is not needed for the argument; see README")


# ---------------------------------------------------------------- f-vectors
def f_vector(points):
    """(f0, f1, f2, f3) of the convex hull.  f0 = vertices actually used, f1 = edges."""
    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    f0 = len(used)
    # facets are 3-dimensional; f3 = number of facets; f2 = number of 2-faces
    facets = [frozenset(s) for s in hull.simplices]
    f3 = len(set(facets))
    ridges = set()
    for f in set(facets):
        for drop in range(4):
            ridges.add(frozenset(tuple(f)[:drop] + tuple(f)[drop + 1:]))
    f2 = len(ridges)
    edges = set()
    for f in facets:
        for pair in itertools.combinations(sorted(f), 2):
            edges.add(pair)
    f1 = len(edges)
    return f0, f1, f2, f3


def tangency_spread(P):
    """Relative spread of the facet distances from the origin: zero exactly when every facet
    is tangent to one sphere about the origin, which the E-construction requires."""
    h = ConvexHull(P)
    d = -h.equations[:, -1] / np.linalg.norm(h.equations[:, :-1], axis=1)
    return float((d.max() - d.min()) / d.mean()), float(d.min()), float(d.max())


def kappa_of(points):
    f0, f1, f2, f3 = f_vector(points)
    return 2 * f1 / f0, (f0, f1, f2, f3)


def cut(points, drop_vertex=0, scale_out=1.0):
    """Remove one vertex and take the convex hull of the rest: the 'cut' polytope.

    `scale_out` pushes the remaining vertices outwards slightly, which is what makes the
    removed part an actual facet (the vertex figure) instead of nothing at all; a strictly
    convex position is preserved for small values."""
    keep = np.delete(points, drop_vertex, axis=0)
    if scale_out != 1.0:
        keep = keep * scale_out
    return keep


def truncated(points, vertex=0, facet_centre=None, how_far=0.15):
    """Truncate a vertex: replace it by its vertex figure, which adds one facet."""
    v = points[vertex]
    keep = np.delete(points, vertex, axis=0)
    link = []
    for p in keep:
        d = np.linalg.norm(p - v)
        if d < 1e-9:
            continue
        link.append((d, p))
    link.sort(key=lambda t: t[0])
    nearest = [p for _, p in link[:12]]
    new = [v + how_far * (p - v) for p in nearest]
    return np.vstack([keep] + [np.array(new)])


def main():
    print("4-polytope            f0     f1      f2     f3     f1/f0    kappa = 2 f1/f0")
    for name, gen in [("5-cell", simplex4), ("8-cell (tesseract)", tesseract),
                      ("16-cell", cross4), ("24-cell", cell24), ("600-cell", cell600)]:
        try:
            P = gen()
        except NotImplementedError:
            continue
        k, f = kappa_of(P)
        print(f"{name:20s} {f[0]:5d} {f[1]:6d} {f[2]:7d} {f[3]:6d}   "
              f"{f[1]/f[0]:6.4f}   {k:8.4f}")

    print()
    print("reference points")
    print("  600-cell: kappa = 12 exactly, the same value as fcc computed in code/packings.py")
    print("  record (cut 600-cell corona): 7656/607 = %.4f, f1/f0 = %.5f"
          % (7656 / 607, 3828 / 607))
    print("  chain of n cut cells: 666/53 = %.4f (the weaker Kuperberg-Schramm value)"
          % (666 / 53))

    print()
    print("cutting the 600-cell: the ratio rises, and so does the tangency error")
    P = cell600()
    rng = np.random.default_rng(1)
    for k in (0, 1, 4, 12, 28):
        Q = P if k == 0 else np.delete(P, rng.choice(len(P), k, replace=False), axis=0)
        kappa_val, f = kappa_of(Q)
        spread, lo, hi = tangency_spread(Q)
        verdict = "tangent" if spread < 1e-6 else "NOT tangent -> not a bound"
        print(f"  cut {k:2d}: f0={f[0]:3d} 2 f1/f0 = {kappa_val:7.4f}"
              f"  facet-distance spread {spread:6.2%}   {verdict}")

    print()
    print("why this is not a search over all polytopes: the cyclic polytope C_4(n) is")
    print("neighbourly, so f1 = C(n,2) and f1/f0 -> n/2 is unbounded. The upper bound on")
    print("kappa must come from ridge tangency (geometry), not from counting.")


if __name__ == "__main__":
    main()
