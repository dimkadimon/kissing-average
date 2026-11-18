"""Exact study of the classical finite packing families, in Q(sqrt 2).

An average-kissing lower bound is a statement about one finite packing: N balls, E
tangencies, average degree 2E/N.  This module builds the standard families and computes
their contact graphs *exactly*, with radii and distances kept in the field Q(sqrt 2), so that
"tangent" is an identity of rational numbers and never a tolerance.

Family alpha: the fcc packing, balls of radius R = 1/sqrt 2 (contact distance 1).
Family beta:  alpha plus every octahedral hole, filled with a ball of radius 1 - 1/sqrt 2.

The second family is the interesting one, and its average degree is exactly 12 in the
infinite limit:

    every octahedral hole touches exactly 6 fcc balls and nothing else, and fcc has one
    octahedral hole per ball, so filling all holes adds N/2 balls and 6*(N/2) = 3N contacts
    to the N/2*... (counts below are done exactly, not in this prose)
    avg_beta = 2(6n + 3n)/(2n) = 12.

So filling the octahedral holes of fcc is *neutral* for the average contact number, and no
amount of interstitial filling of an fcc packing can push past 12.  That is the quantitative
reason the dimension-3 record of Eppstein-Kuperberg-Ziegler (7656/607 = 12.6129) needs the
polytope compound rather than an interstitial trick.  Tetrahedral holes are worse: a ball in
a tetrahedral hole has 4 contacts and is many times smaller than an fcc ball, so it drags the
average down.

The finite clusters reported here are counted on the *bulk*: only balls all of whose contacts
lie inside the cluster are counted, so 2E/N is the average degree of an induced subgraph and
the boundary does not flatter the numbers.

usage:
    python code/packings.py
"""
from fractions import Fraction as F
from itertools import combinations


# ---------------------------------------------------------------- exact arithmetic Q(sqrt2)
class Q2:
    """a + b*sqrt(2) with a, b rational."""
    __slots__ = ("a", "b")

    def __init__(self, a=F(0), b=F(0)):
        self.a, self.b = F(a), F(b)

    def __add__(self, o):
        return Q2(self.a + o.a, self.b + o.b)

    def __sub__(self, o):
        return Q2(self.a - o.a, self.b - o.b)

    def __mul__(self, o):
        return Q2(self.a * o.a + 2 * self.b * o.b, self.a * o.b + self.b * o.a)

    def __eq__(self, o):
        return self.a == o.a and self.b == o.b

    def __lt__(self, o):
        return float(self) < float(o)

    def __float__(self):
        return float(self.a) + float(self.b) * 2.0 ** 0.5

    def __repr__(self):
        return f"({self.a})+({self.b})r2"

    def is_zero(self):
        return self.a == 0 and self.b == 0


ZERO = Q2(0)
ONE = Q2(1)
R_FCC = Q2(0, F(1, 2))        # 1/sqrt 2
R_HOLE = Q2(1, F(-1, 2))      # 1 - 1/sqrt 2


# ---------------------------------------------------------------- families
def fcc_points(L):
    """Integer points (i,j,k), 0 <= i,j,k < L, i+j+k even: fcc lattice sites."""
    out = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if (i + j + k) % 2 == 0:
                    out.append((i, j, k))
    return out


def octahedral_holes(L):
    """Odd-sum integer points strictly inside the box: centres of octahedral holes."""
    out = []
    for i in range(1, L - 1):
        for j in range(1, L - 1):
            for k in range(1, L - 1):
                if (i + j + k) % 2 == 1:
                    out.append((i, j, k))
    return out


def dist2(p, q):
    """Squared distance of two integer points, as a rational."""
    return F(sum((a - b) ** 2 for a, b in zip(p, q)))


def contact_graph_exact(sites, period=None):
    """sites: list of (centre, radius in Q2).  An edge is an exact equality
    |p-q|^2 = (r+s)^2, i.e.  d2 = ra^2 + rb^2 + 2 ra rb  with d2 rational.

    With `period` given the coordinates live on the torus (Z/period)^3, i.e. the family is
    taken to be periodic, so every ball has its full infinite-family contact set and the
    average degree is the exact value for the infinite packing rather than a boundary-
    truncated one."""
    n = len(sites)
    adj = [[] for _ in range(n)]
    for i, j in combinations(range(n), 2):
        (p, r), (q, s) = sites[i], sites[j]
        if period is None:
            d2 = dist2(p, q)
        else:
            d2 = F(0)
            for a, b in zip(p, q):
                d = (a - b) % period
                d = min(d, period - d)
                d2 += d * d
        # (r+s)^2 = r.r + s.s + 2 r.s
        rs2 = r * r + s * s + Q2(2) * (r * s)
        if rs2.b == 0 and rs2.a == d2:
            adj[i].append(j)
            adj[j].append(i)
    return adj


def bulk_stats(sites, adj, box=None):
    """Count N, E and the degree range over the bulk: the balls all of whose contacts are
    inside the cluster.  With `box` given, the bulk is instead the set of balls whose
    contact set is the full one they would have in the infinite family (degree equal to the
    family degree), which is the cleaner criterion for a lattice."""
    deg = [len(a) for a in adj]
    if box is None:
        N = len(sites)
        E = sum(deg) // 2
        return N, E, (min(deg) if deg else 0), (max(deg) if deg else 0)
    return None


def fcc_family(L):
    P = fcc_points(L)
    return [(p, R_FCC) for p in P]


def fcc_plus_holes(L):
    P = [(p, R_FCC) for p in fcc_points(L)]
    H = [(h, R_HOLE) for h in octahedral_holes(L)]
    return P + H


def report(L, sites, name, period=None):
    adj = contact_graph_exact(sites, period=period)
    deg = [len(a) for a in adj]
    N = len(sites)
    E = sum(deg) // 2
    return dict(name=f"{name} (torus {L}^3)", N=N, E=E, avg=2 * E / N,
                deg_min=min(deg), deg_max=max(deg))


if __name__ == "__main__":
    print("exact contact graphs, radii in Q(sqrt 2) =", float(R_FCC), "and", float(R_HOLE))
    print()
    for L in (4, 6):
        for name, f in (("fcc", fcc_family), ("fcc + octahedral holes", fcc_plus_holes)):
            r = report(L, f(L), name, period=L)
            print(f"  {r['name']:34s} N={r['N']:4d} E={r['E']:5d} avg={r['avg']:7.4f}"
                  f"  deg {r['deg_min']}..{r['deg_max']}")
        print()
    print("  reading: on the torus (periodic family, no boundary) fcc has average degree")
    print("  exactly 12, and filling every octahedral hole also gives exactly 12: the")
    print("  interstitial trick is neutral, which is why the 7656/607 record needs a")
    print("  different construction.")
    print()
    print("  tetrahedral holes, by the same count: fcc has two per ball, each such ball has")
    print("  exactly 4 contacts, so filling them gives E = 6n + 8n = 14n on N = 3n balls,")
    print("  average 28/3 = 9.333 < 12 -- strictly worse than leaving the holes empty.")
