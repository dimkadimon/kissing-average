"""Periodic (not necessarily lattice) packings: average contact number, computed directly.

The point of this module.  Bockhaus-Glazyrin's lower-bound column (their Table 1, and every
lower bound in results/records.txt) is the *lattice* kissing number, because for a lattice
packing every ball has the same number of contacts.  But the quantity being bounded is the
*average* contact number over a packing, and a periodic packing with several orbits does not
have to be balanced: with orbits of contact counts c_0, ..., c_{k-1} the average is
(c_0 + ... + c_{k-1})/k, which can exceed every individual c_i.  That is exactly the mechanism
by which non-lattice packings beat lattice ones for the kissing number in high dimensions, and
it is available here too.  So this module searches periodic packings with k orbits on top of a
good lattice and asks whether the average can be pushed above tau(L).

Everything is computed in coordinates: the lattice is given by a Gram matrix G, a basis B with
B B^T = G is taken from the Cholesky factor, sites are B w + s_i, and a contact is a pair at
squared distance mu (the minimal norm) to within 1e-9.  The enumeration box is checked to be
large enough: a contact whose lattice coordinate sits on the boundary of the box would be a
sign the box is too small, and that is reported rather than silently dropped.

usage:
    python code/periodic.py --check          # one orbit must reproduce tau exactly
    python code/periodic.py --dim 9 --k 2 --budget 120
"""
import argparse
import itertools
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from lattices import FAMILIES, all_vectors, min_vectors      # noqa: E402


def basis_of(G):
    """The map from lattice coordinates to space: with G = L L^T, the point of coordinates w is
    L^T w, so |L^T w|^2 = w^T G w.  Returning L means a row-vector product W @ L gives exactly
    (L^T w)^T, which is what lattice_points needs."""
    return np.linalg.cholesky(np.array(G, dtype=float))


def min_norm(G):
    return float(min(sum(v[i] * G[i][j] * v[j] for i in range(len(G)) for j in range(len(G)))
                     for v in min_vectors(G)[0]))


def contacts_of(B, shifts, mu):
    """Contacts per orbit pair for the packing U_i (L + s_i), balls of radius sqrt(mu)/2.

    For orbits (i, j) with shift difference delta, a contact is a lattice coordinate w with
    |B^T w + delta|^2 = mu.  Only w with w^T G w <= (sqrt(mu) + |delta|)^2 can qualify, so that
    ball is enumerated exactly (lattices.all_vectors) and every candidate is tested by the
    identity  |B^T w + delta|^2 = w^T G w + 2<B^T w, delta> + |delta|^2.

    The Gram matrix is integral here (mu = 2), so the bound is an integer and the enumeration
    is exact; no scaling is applied, because scaling the Gram matrix changes which integer
    vectors are integral points of the lattice."""
    k = len(shifts)
    d = B.shape[0]
    G = [[int(round(float((B @ B.T)[i][j]))) for j in range(d)] for i in range(d)]
    counts = np.zeros((k, k), dtype=int)
    min_d2 = math.inf
    for i in range(k):
        for j in range(k):
            delta = np.asarray(shifts[j], dtype=float) - np.asarray(shifts[i], dtype=float)
            rad = math.sqrt(mu) + float(np.linalg.norm(delta))
            bound = int(math.floor(rad * rad)) + 1
            W = all_vectors(G, bound)
            d2 = np.empty(len(W))
            for t, w in enumerate(W):
                p = B.T @ np.array(w, dtype=float)
                d2[t] = float(p @ p) + 2.0 * float(p @ delta) + float(delta @ delta)
            # Sites at distance 0 are a genuine problem when i != j: the two orbits coincide
            # and the "packing" has duplicate balls.  Only the self-distance w = 0 (i == j) is
            # allowed to be ignored, and it is excluded by requiring a nonzero coordinate.
            if i == j:
                d2 = np.where(d2 < 1e-12, math.inf, d2)
            elif len(d2):
                min_d2 = min(min_d2, 0.0)          # coincident sites: always infeasible
            counts[i, j] = int((np.abs(d2 - mu) < 1e-9 * mu).sum())
            finite = d2[np.isfinite(d2)]
            if len(finite):
                min_d2 = min(min_d2, float(finite.min()))
    E = sum(counts[i][i] for i in range(k)) / 2.0 \
        + sum(counts[i][j] for i in range(k) for j in range(k) if i != j) / 2.0
    return dict(counts=counts, edges=E, average=2.0 * E / k, min_d2=min_d2,
                per_orbit=counts.sum(axis=1))


def report(G, shifts, label=""):
    B = basis_of(G)
    mu = min_norm(G)
    st = contacts_of(B, shifts, mu)
    st["tau"] = len(min_vectors(G)[0])
    st["mu"] = mu
    st["feasible"] = st["min_d2"] >= mu - 1e-9
    return st


def check():
    print("validation: a single orbit must reproduce tau(L) and have min distance mu")
    print(f"{'lattice':>8} {'tau':>5} {'computed':>9} {'min d^2 / mu':>13}")
    for name, d in (("D", 5), ("E", 6), ("E", 7), ("E", 8), ("D", 4), ("A", 3)):
        G = FAMILIES[name](d)
        st = report(G, [[0.0] * d])
        print(f"{name + str(d):>8} {st['tau']:>5} {st['average']:>9.1f} "
              f"{st['min_d2'] / st['mu']:>13.9f}")


def optimise(G, k, budget, seed=0, verbose=True, grid=None):
    """Search shifts for a k-orbit periodic packing with maximal average contact number."""
    rng = np.random.default_rng(seed)
    d = len(G)
    B = basis_of(G)
    mu = min_norm(G)
    tau = len(min_vectors(G)[0])
    best = (-1, None)
    t0 = time.time()
    grid = grid if grid is not None else [0.0, 0.25, 1 / 3, 0.5]
    tries = 0
    while time.time() - t0 < budget:
        # random shift set, then local improvement
        shifts = [[0.0] * d] + [list(rng.choice(grid, size=d)) for _ in range(k - 1)]
        st = contacts_of(B, shifts, mu)
        if st["min_d2"] < mu - 1e-9:
            continue
        cur, curS = st["average"], shifts
        for _ in range(60):
            improved = False
            for i in range(1, k):
                for axis in range(d):
                    for val in grid:
                        trial = [list(s) for s in curS]
                        trial[i][axis] = val
                        s2 = contacts_of(B, trial, mu)
                        if s2["min_d2"] >= mu - 1e-9 and s2["average"] > cur + 1e-9:
                            cur, curS, improved = s2["average"], trial, True
            if not improved:
                break
        tries += 1
        if cur > best[0]:
            best = (cur, curS)
            if verbose:
                print(f"    k={k}: average {cur:.4f} vs tau {tau} "
                      f"({time.time() - t0:.0f}s, {tries} restarts)", flush=True)
    return best, tau, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--k", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--budget", type=float, default=60.0)
    args = ap.parse_args()
    if args.check:
        check()
        return
    d = args.dim
    name = "E" if d in (6, 7, 8) else "D"
    G = FAMILIES[name](d)
    tau = len(min_vectors(G)[0])
    print(f"d = {d}, base lattice {name}{d}, tau = {tau} (the published lower bound)")
    for k in args.k:
        (val, shifts), tau, tries = optimise(G, k, args.budget)
        verdict = "IMPROVES the lattice bound" if val > tau + 1e-9 else "no improvement"
        print(f"  k = {k} orbits: best average {val:.4f} vs tau {tau}  -> {verdict}"
              f"  ({tries} restarts)")


if __name__ == "__main__":
    main()
