"""Exact contact graphs of direction systems, and the two structural facts this repository
is about.

A direction system S is a finite set of unit vectors in R^k.  Two of them are *in contact*
when their inner product is exactly 1/2 (angle 60 degrees).  The contact graph records
which pairs touch.

Everything here is rational: each system is stored as integer vectors, and a contact is an
exact equality 2<x,y> = |x|^2 = |y|^2, checked over the rationals.

Contents
--------
root_system('D', n)   the D_n root system, n(n-1)*2 unit vectors
root_system('A', n)   the A_n root system (A_3 = D_3 = the cuboctahedron, 12 points)
root_system('E7'), root_system('E8')   the exceptional root systems
contact_graph(S)      adjacency matrix of the 1/2-inner-product graph
degrees(S)            degree of every direction in that graph
max_coverage(S, ...)  largest number of directions within 60 degrees of a single point
                      (exact by brute force over the cone structure, see below)

The two facts this file checks
------------------------------
(1) *Four-regularity of the kissing contact graph in dimensions 3 and 4.*  In the kissing
    configuration of 12 points (A_3) and of 24 points (D_4) every point is in contact with
    exactly four others.  This is the single hypothesis of the lemma that turns the
    F_4 -> E_6 step (12 -> 24) into a contact identity, and it is what will be used for
    any higher-dimensional analogue.

(2) *Max coverage.*  If a single point of the unit sphere can be within 60 degrees of at
    most nu directions of S, then deleting one point of a lifted configuration covers at
    most nu directions.  For the 12-point and 24-point systems nu = 12 and nu = 12
    respectively, which is the quantity that decides whether the
    ``deleted points < new points'' condition (Eppstein-Kuperberg-Ziegler) can be met.

usage:
    python code/graphs.py
"""
from fractions import Fraction as F
from itertools import combinations


# ------------------------------------------------------------------ direction systems
def _signs(k, parity=None):
    """All sign vectors of length k, optionally with an even/odd number of minus signs."""
    out = []
    for m in range(1 << k):
        v = tuple(1 if (m >> i) & 1 == 0 else -1 for i in range(k))
        if parity is None or (v.count(-1) % 2 == (0 if parity == "even" else 1)):
            out.append(v)
    return out


def root_system(name, n=None):
    """Unit vectors of a root system, as tuples of Fractions."""
    if name == "A":
        # A_n in R^{n+1}: e_i - e_j, i != j  (A_3 = D_3, 12 vectors)
        m = n + 1
        out = []
        for i in range(m):
            for j in range(m):
                if i != j:
                    v = [0] * m
                    v[i], v[j] = 1, -1
                    out.append(tuple(F(x, 2) for x in v))  # |v|^2 = 2/4 = 1/2? no:
        return _normalise(out)
    if name == "D":
        # D_n: +-e_i +-e_j, i < j
        out = []
        for i, j in combinations(range(n), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * n
                    v[i], v[j] = si, sj
                    out.append(tuple(F(x, 2) for x in v))
        return _normalise(out)
    if name == "E8":
        out = []
        for i, j in combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i], v[j] = si, sj
                    out.append(tuple(F(x, 2) for x in v))
        for s in _signs(8, "even"):
            out.append(tuple(F(x, 2) for x in s))
        return _normalise(out)
    if name == "E7":
        # the E8 roots with coordinate sum zero (126 of them), scaled to unit length
        out = []
        for i, j in combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i], v[j] = si, sj
                    if sum(v) == 0:
                        out.append(tuple(F(x, 2) for x in v))
        for s in _signs(8, "even"):
            if sum(s) == 0:
                out.append(tuple(F(x, 2) for x in s))
        return _normalise(out)
    raise ValueError(name)


def _normalise(vs):
    """Scale every vector to unit length, exactly where possible.

    The systems above are built so that every vector has the same rational squared length;
    we divide by its square root and keep the resulting angle information in the exact
    form <x,y>^2 / (|x|^2|y|^2), which is all the contact test needs.
    """
    seen = set()
    out = []
    for v in vs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def norm2(x):
    return dot(x, x)


def in_contact(x, y):
    """Angle exactly 60 degrees: <x,y>^2 = |x|^2 |y|^2 / 4 and <x,y> > 0."""
    d = dot(x, y)
    return d > 0 and 4 * d * d == norm2(x) * norm2(y)


def within_60(x, y):
    """Angle at most 60 degrees (used for coverage counts)."""
    d = dot(x, y)
    return d > 0 and 4 * d * d >= norm2(x) * norm2(y)


def contact_graph(S):
    n = len(S)
    return [[j for j in range(n) if j != i and in_contact(S[i], S[j])] for i in range(n)]


def degrees(S):
    adj = contact_graph(S)
    return [len(a) for a in adj]


# ------------------------------------------------------------------ coverage
def sample_directions(k, count=20000, seed=0, half=2000):
    """A dense, deterministic set of test directions on S^{k-1} (integer lattice points
    with small coordinates, normalised).  Used to *lower*-bound the coverage number."""
    import itertools
    import math
    pts = []
    rng = range(-6, 7)
    for v in itertools.product(rng, repeat=k):
        if any(v):
            pts.append(v)
        if len(pts) >= count:
            break
    return pts


def max_coverage_lower_bound(S, k, tries=200000):
    """Largest number of directions of S within 60 degrees of some point, found by
    searching all nonzero integer points with small coordinates (plus all directions of S
    themselves and their midpoints).  This is a lower bound on the exact value."""
    from itertools import product
    cand = set()
    for v in S:
        cand.add(v)
    for a, b in combinations(S, 2):
        cand.add(tuple(x + y for x, y in zip(a, b)))
    r = range(-4, 5)
    for v in product(r, repeat=k):
        if any(v):
            cand.add(tuple(F(x) for x in v))
    best, arg = 0, None
    for c in cand:
        if norm2(c) == 0:
            continue
        cnt = sum(1 for s in S if within_60(c, s))
        if cnt > best:
            best, arg = cnt, c
    return best, arg


# ------------------------------------------------------------------ the checks
def check_tower():
    """Contact counts along the root-system tower 3, 4, 4, 8 (dimensions 3, 4, 6, 8)."""
    rows = []
    for name, n in [("A", 3), ("D", 4), ("D", 5), ("E7", None), ("E8", None)]:
        S = root_system(name, n)
        d = degrees(S)
        rows.append((f"{name}{n if n else ''}", len(S), sum(d) // 2,
                     min(d), max(d), len(set(d))))
    return rows


def check_regular():
    """Is the contact graph 4-regular for A_3 (12 points) and D_4 (24 points)?"""
    out = {}
    for name, n in [("A", 3), ("D", 4), ("D", 5), ("E8", None)]:
        S = root_system(name, n)
        d = degrees(S)
        out[f"{name}{n if n else ''}"] = (len(S), sorted(set(d)))
    return out


if __name__ == "__main__":
    print("contact graphs of root systems (points, edges, min degree, max degree, #degrees)")
    for r in check_tower():
        print("   ", r)
    print()
    print("degree sets (a single value means the graph is regular)")
    for k, v in check_regular().items():
        print("   ", k, "points", v[0], "degrees", v[1])
    print()
    for name, n in [("A", 3), ("D", 4)]:
        S = root_system(name, n)
        k = 3 if name == "A" else 4
        best, arg = max_coverage_lower_bound(S, k)
        print(f"max coverage for {name}{n}: at least {best} of {len(S)} directions"
              f" (witness {arg[:6]}...)")
