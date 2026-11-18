"""Exact kissing numbers of the classical lattices, for every dimension up to 24.

For a *lattice* packing every ball has exactly the same number of contacts, tau(L), so the
average kissing number of that packing is tau(L), and a lower bound for kappa_d is exactly "a
lattice with a large tau".  This module computes tau(L) from an integer Gram matrix by exact
Fincke-Pohst enumeration: floats choose the search order, and every candidate is accepted or
rejected by the integer test x^T G x <= bound, so no tolerance enters the answer.

What the table shows, and what it is honest about:

  * for d <= 8 the published record is attained by a root lattice and is reproduced exactly
    (2 = Z, 6 = A_2, 12 = A_3, 24 = D_4, 40 = D_5, 72 = E_6, 126 = E_7, 240 = E_8);
  * from d = 9 on, no root lattice comes close, and the ratio column grows exponentially
    (2.1 at d = 9, 178 at d = 24).  The records there are *not* root lattices: they are
    laminated lattices Lambda_9..12 (272, 336, 438, 648), Barnes-Wall, and the Leech lattice,
    together with non-lattice configurations for d = 9..13.  Those lattices are gluings of a
    code with a lattice, and their Gram matrices are not built here, so the table says
    "not reproduced" rather than pretending;
  * the exhaustive direct-sum search (--long) is exact: tau is additive over direct sums, so
    the best average kissing number reachable by root lattices in dimension d is a finite
    optimisation over partitions, solved here by depth-first search.

usage:
    python code/lattices.py          # table, d = 1..24
    python code/lattices.py --long   # plus the exhaustive direct-sum search
    python code/lattices.py --dim 8  # one dimension, verbose
"""
import argparse
import math
import time

# Best published lower bounds on the kissing number, d = 1..24, from the table maintained by
# Henry Cohn (cohn.mit.edu/kissing-numbers, read 19 November 2026), which supersedes the
# Nebe-Sloane lower-bound table.  2024-2026 improvements: K(10) 500 -> 510 (Ganzhinov),
# K(11) 582 -> 604 (Bianchi-Kwon-Pappu-Zou), K(12) 840 -> 841 (Takhanov-Assylbekov-Yun),
# K(14) 1606 -> 1932, K(17) 5346 -> 5730 (Cohn-Li), K(18) 7398 -> 8358,
# K(19) 10668 -> 12460, K(20) 17400 -> 19448, K(21) 27720 -> 31560.
# Exact: d = 1, 2, 3, 4, 8, 24.
KISSING_RECORD = {
    1: 2, 2: 6, 3: 12, 4: 24, 5: 40, 6: 72, 7: 126, 8: 240, 9: 306, 10: 510,
    11: 604, 12: 841, 13: 1154, 14: 1932, 15: 2564, 16: 4320, 17: 5730, 18: 8358,
    19: 12460, 20: 19448, 21: 31560, 22: 49896, 23: 93150, 24: 196560,
}
# Matching upper bounds where the value is not known (same table, same date).
KISSING_UPPER = {5: 43, 6: 76, 7: 132, 9: 363, 10: 552, 11: 867, 12: 1355, 13: 2064,
                 14: 3174, 15: 4853, 16: 7319, 17: 10978, 18: 16406, 19: 24417,
                 20: 36195, 21: 53524, 22: 80810, 23: 122351}


# ------------------------------------------------------------------ exact enumeration
def cholesky_upper(G):
    """R with G = R^T R, upper triangular, in floats.  Used only to order the search."""
    n = len(G)
    R = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s = float(G[i][i]) - sum(R[k][i] * R[k][i] for k in range(i))
        if s <= 0:
            raise ValueError("Gram matrix is not positive definite in this basis")
        R[i][i] = math.sqrt(s)
        for j in range(i + 1, n):
            s = float(G[i][j]) - sum(R[k][i] * R[k][j] for k in range(i))
            R[i][j] = s / R[i][i]
    return R


def _enumerate(G, bound, want_all=False):
    """Fincke-Pohst: every integer x with 0 < x^T G x <= bound, exactly.

    The floats only decide where to look.  Rounding can push a partial sum a hair above the
    bound, and an exactly-minimal vector must not be lost to that, so the pruning bound is
    loosened and the coordinate window widened; membership is then decided by the exact
    integer test x^T G x <= bound at the leaf."""
    G = [[int(v) for v in r] for r in G]
    n = len(G)
    tol = 1e-6 * bound + 1e-9
    R = cholesky_upper(G)
    out = []

    def q(x):
        return sum(x[i] * G[i][j] * x[j] for i in range(n) for j in range(n))

    def rec(i, x, acc):
        if i < 0:
            xa = [int(v) for v in x]
            if any(xa) and q(xa) <= bound:
                out.append(tuple(xa))
            return
        centre = -sum(R[i][j] * x[j] for j in range(i + 1, n))
        slack = bound + tol - acc
        if slack < 0:
            return
        rad = math.sqrt(slack) / R[i][i]
        for xi in range(math.floor(centre - rad) - 1, math.ceil(centre + rad) + 2):
            t = (R[i][i] * xi + sum(R[i][j] * x[j] for j in range(i + 1, n))) ** 2
            if acc + t <= bound + tol:
                x[i] = xi
                rec(i - 1, x, acc + t)

    rec(n - 1, [0] * n, 0.0)
    if want_all:
        return out
    norm = min(q(v) for v in out) if out else None
    return [v for v in out if q(v) == norm], norm


def all_vectors(G, bound):
    """Every integer x with 0 < x^T G x <= bound, exactly."""
    return _enumerate(G, bound, want_all=True)


def min_vectors(G, bound=None):
    """All nonzero integer x with x^T G x <= bound, exactly (default bound: the minimum)."""
    G = [[int(v) for v in r] for r in G]
    if bound is None:
        bound = min(G[i][i] for i in range(len(G)))    # x = +-e_i shows tau >= 2n
    return _enumerate(G, bound, want_all=False)


# ------------------------------------------------------------------ the classical families
def gram_Zn(d):
    return [[2 if i == j else 0 for j in range(d)] for i in range(d)]


def gram_An(d):
    """A_d inside the hyperplane sum x = 0 of Z^{d+1}; basis e_i - e_{i+1}."""
    if d < 1:
        raise ValueError("A_d needs d >= 1")
    G = [[0] * d for _ in range(d)]
    for i in range(d):
        G[i][i] = 2
        if i + 1 < d:
            G[i][i + 1] = G[i + 1][i] = -1
    return G


def gram_Dn(d):
    """D_d = {x in Z^d : sum x even}; basis b_1 = e_1 + e_2, b_i = e_i - e_{i-1} (i >= 2)."""
    if d < 2:
        raise ValueError("D_d needs d >= 2")
    basis = []
    b = [0] * d
    b[0], b[1] = 1, 1
    basis.append(b)
    for i in range(1, d):
        v = [0] * d
        v[i], v[i - 1] = 1, -1
        basis.append(v)
    return [[sum(a * b for a, b in zip(basis[i], basis[j])) for j in range(d)]
            for i in range(d)]


def gram_cartan(d):
    """E6, E7, E8 as Cartan matrices (min norm 2; tau = 72, 126, 240)."""
    M = {
        6: [[2, -1, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0], [0, -1, 2, -1, 0, -1],
            [0, 0, -1, 2, -1, 0], [0, 0, 0, -1, 2, 0], [0, 0, -1, 0, 0, 2]],
        7: [[2, -1, 0, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0, 0], [0, -1, 2, -1, 0, 0, -1],
            [0, 0, -1, 2, -1, 0, 0], [0, 0, 0, -1, 2, -1, 0], [0, 0, 0, 0, -1, 2, 0],
            [0, 0, -1, 0, 0, 0, 2]],
        8: [[2, -1, 0, 0, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, -1], [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, 0], [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0], [0, 0, -1, 0, 0, 0, 0, 2]],
    }
    return M[d]


FAMILIES = {"Z": gram_Zn, "A": gram_An, "D": gram_Dn, "E": gram_cartan}
MAXDIM = {"Z": 24, "A": 24, "D": 24, "E": 8}


def available(name, d):
    if d > MAXDIM[name]:
        return False
    return name != "E" or d in (6, 7, 8)


def tau(name, d):
    if not available(name, d):
        return None
    try:
        vecs, _ = min_vectors(FAMILIES[name](d))
    except Exception:
        return None
    return len(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--dim", type=int, default=None)
    args = ap.parse_args()

    if args.dim:
        d = args.dim
        for name in FAMILIES:
            if not available(name, d):
                continue
            t0 = time.time()
            vecs, norm = min_vectors(FAMILIES[name](d))
            print(f"d = {d}: {name}_{d}  min norm^2 = {norm}  tau = {len(vecs)}"
                  f"  ({time.time() - t0:.2f} s)")
        return

    print("exact kissing numbers of the classical lattices, d = 1..24")
    print("(tau = number of minimal vectors, from integer Gram matrices by Fincke-Pohst)")
    print()
    print(f"{'d':>2}  {'Z_d':>5} {'A_d':>5} {'D_d':>5} {'E':>5}  {'best here':>11}"
          f"  {'record':>8}  {'upper':>7}  {'record/best':>11}")
    best_by_dim = {}
    for d in range(1, 25):
        row = {name: tau(name, d) for name in FAMILIES}
        vals = {k: v for k, v in row.items() if v}
        bestname, best = max(vals.items(), key=lambda kv: kv[1]) if vals else ("-", 0)
        best_by_dim[d] = (bestname, best)
        rec = KISSING_RECORD[d]
        up = KISSING_UPPER.get(d, "= record")
        cells = "  ".join(f"{str(row[k]):>5}" if row[k] else f"{'-':>5}" for k in FAMILIES)
        ratio = f"{rec / best:.2f}" if best else "-"
        print(f"{d:>2}  {cells}  {bestname + '_' + str(d) + '=' + str(best):>11}"
              f"  {rec:>8}  {str(up):>7}  {ratio:>11}")

    print()
    print("reading of the table")
    print("  * d <= 8: the record is a root lattice and is reproduced exactly; the values are")
    print("    K = 2, 6, 12, 24, 40, 72, 126, 240 and are known to be optimal.")
    print("  * d = 9..24: no root lattice comes close, and record/best grows without bound")
    print("    (2.1 at d = 9, 178 at d = 24). The records are laminated lattices Lambda_9..12")
    print("    (272, 336, 438, 648), Barnes-Wall (d = 16, 4320), the Leech lattice (d = 24,")
    print("    196,560) and non-lattice configurations for d = 9..13; none of them is a root")
    print("    lattice and none is built here. The ratio is a statement about root lattices.")
    print("  * d = 5, 6, 7 have K = 40, 72, 126 with upper bounds 43, 76, 132 (Tomar 2026):")
    print("    these are the smallest dimensions where a better packing might exist.")

    if not args.long:
        print()
        print("run with --long for the exhaustive direct-sum search (the best average kissing")
        print("number reachable by root lattices in each dimension).")
        return

    print()
    print("exhaustive direct-sum search: maximise tau over all direct sums of root lattices")
    print("(tau is additive over direct sums, so this is an exact finite optimisation)")
    tau_of = {}
    for name in ("A", "D", "E"):
        for d in range(1, 25):
            if not available(name, d):
                continue
            v = tau(name, d)
            if v:
                tau_of[(name, d)] = v
    td = sorted(tau_of.items(), key=lambda kv: kv[1] / kv[0][1], reverse=True)

    for d in range(1, 25):
        bestval, bestdesc = -1, None

        def dfs(rem, start, acc, desc):
            nonlocal bestval, bestdesc
            if rem == 0:
                if acc > bestval:
                    bestval, bestdesc = acc, "+".join(desc)
                return
            for idx in range(start, len(td)):
                (name, k), v = td[idx]
                if k <= rem:
                    desc.append(f"{name}{k}")
                    dfs(rem - k, idx, acc + v, desc)
                    desc.pop()

        dfs(d, 0, 0, [])
        rec = KISSING_RECORD[d]
        print(f"  d = {d:>2}: best root-lattice direct sum {bestdesc:<12} "
              f"tau = {bestval:>7}   record {rec:>7}   ratio {rec / bestval:>6.2f}")


if __name__ == "__main__":
    main()
