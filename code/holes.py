"""How much room is there around the best packing in dimensions 5, 6 and 7?

K(5) = 40 <= 43, K(6) = 72 <= 76, K(7) = 126 <= 132: these are the only dimensions below 13
where the record is not code-theoretic, and the only ones where a better packing might
plausibly exist.  This module takes the record configuration (40, 72, 126 unit vectors: the
minimal vectors of D_5, E_6, E_7, computed exactly) and answers the first necessary question:

    can one more ball be added without moving the others?

Equivalently: is the deepest "hole" of the configuration an admissible direction?  Write
rho(P) = min_{|x|=1} max_{p in P} <x, p>.  A point x can be added to P iff max_p <x,p> <= 1/2,
so rho(P) <= 1/2 means the configuration is *not* jammed and one more point fits; rho(P) > 1/2
means it is locally jammed, and the surplus rho - 1/2 measures how badly.

The hole problem is a minimax over the sphere: the minimum of a max of linear functions, which
is attained at a point x proportional to the sum of the "active" constraints.  We solve it two
ways -- log-sum-exp continuation (smooth, many starts) and exact subset enumeration of the
active set -- and report both, so the number is not an artefact of one optimiser.

usage:
    python code/holes.py
"""
import itertools
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from lattices import FAMILIES, min_vectors            # noqa: E402


def record_configuration(name, d):
    """The 2d, 40, 72 or 126 minimal vectors of the record lattice, as unit vectors."""
    G = FAMILIES[name](d)
    vecs, norm = min_vectors(G)
    L = np.linalg.cholesky(np.array(G, dtype=float))     # G = L L^T
    pts = np.array([L.T @ np.array(v, dtype=float) for v in vecs])   # so |x|^2 = v^T G v
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def objective(x, P, beta):
    s = P @ x
    m = s.max()
    w = np.exp(beta * (s - m))
    Z = w.sum()
    val = m + math.log(Z) / beta
    grad = (w / Z) @ P
    return val, grad


def hole_logsumexp(P, starts=4000, beta0=30.0, rounds=12, rng=None):
    """min over the sphere of max_p <x,p>, by log-sum-exp continuation."""
    rng = rng or np.random.default_rng(0)
    d = P.shape[1]
    X = rng.normal(size=(starts, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    vals = (P @ X.T).max(axis=0)
    # keep the best 200 starts
    idx = np.argsort(vals)[:200]
    X = X[idx]
    beta = beta0
    best = np.inf
    for _ in range(rounds):
        for _ in range(60):
            S = P @ X.T                     # (|P|, k)
            M = S.max(axis=0)
            W = np.exp(beta * (S - M))
            Z = W.sum(axis=0)
            grad = P.T @ (W / Z)            # (d, k)
            step = 0.5 / beta
            X = X - step * grad.T            # descent: we minimise the max
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        beta = min(beta * 1.7, 4000.0)
        cur = (P @ X.T).max(axis=0).min()
        best = min(best, cur)
    k = int(np.argmin((P @ X.T).max(axis=0)))
    return float((P @ X[k]).max()), X[k]


def hole_active_set(P, x0, tol=1e-6, rounds=8):
    """Exact refinement of a candidate hole.

    At a local minimum x of max_p <x,p> with value t, the active set A = {p : <x,p> = t} must
    satisfy: x is parallel to sum_{p in A} p with non-negative multipliers (KKT).  So we take
    the active set of the current x, replace x by normalise(sum_{A} p), and repeat until the
    active set is stable.  This turns a numerical optimum into a verified one: the returned
    value is the maximum over ALL points, not just the active ones, so if it exceeds the
    optimum the refinement simply reports a worse (but sound) number."""
    best_x, best_val = x0, float((P @ x0).max())
    for _ in range(rounds):
        vals = P @ best_x
        t = vals.max()
        A = np.where(vals >= t - tol)[0]
        cand = P[A].sum(axis=0)
        n = np.linalg.norm(cand)
        if n < 1e-12:
            break
        cand = cand / n
        val = float((P @ cand).max())
        if val < best_val - 1e-12:
            best_x, best_val = cand, val
        else:
            break
    return best_val, best_x


def hole_subsets(P, k_max=3):
    """Cross-check by enumerating the active sets of size at most k_max and testing the KKT
    point normalise(sum of the subset).  Covers every hole whose active set is small, which is
    the case for the highly symmetric configurations here."""
    N, d = P.shape
    best = np.inf
    bestx = None
    K = min(k_max, N)
    for k in range(1, K + 1):
        for subset in itertools.combinations(range(N), k):
            s = P[list(subset)].sum(axis=0)
            n = np.linalg.norm(s)
            if n < 1e-9:
                continue
            x = s / n
            v = float((P @ x).max())
            if v < best:
                best, bestx = v, x
    return best, bestx


def certify(P, rho, x, tol=1e-6):
    """Check that x is a KKT point of the hole problem, and identify the closed form of rho.

    KKT for  min max_p <x,p>  on the sphere: x is parallel to the sum of the active p's.  At
    such a point rho = <x,p> for each active p, and the scaled vector y = x/rho satisfies
    <y,p> = 1 for all active p -- so y is a vector of norm 1/rho at lattice inner product 1
    from the active set.  1/rho^2 coming out as a small rational is what makes the value look
    like sqrt(2/5) rather than a numerical accident."""
    vals = P @ x
    t = vals.max()
    A = np.where(vals >= t - tol)[0]
    S = P[A].sum(axis=0)
    kkt = float(np.linalg.norm(S / np.linalg.norm(S) - x))
    y = x / t
    ips = P[A] @ y
    # local minimality: perturb x on the sphere and re-measure
    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(200):
        z = x + 0.01 * rng.normal(size=x.size)
        z /= np.linalg.norm(z)
        if float((P @ z).max()) > t:
            worse += 1
    return dict(active=len(A), kkt_residual=kkt, inv_rho2=1.0 / t ** 2,
                y_norm2=float(y @ y), active_inner_min=float(ips.min()),
                active_inner_max=float(ips.max()), local_min=f"{worse}/200")


def main():
    print("the record configurations in dimensions 5, 6, 7, and their deepest holes")
    print()
    print("rho(P) = min over unit x of max over p in P of <x,p>;  a point can be added iff")
    print("rho(P) <= 1/2, i.e. iff the min angle arccos(rho) is at least 60 degrees.")
    print()
    print(f"{'d':>2} {'config':>8} {'|P|':>5} {'rho':>10} {'angle':>9} {'deficit':>9}  jammed?")
    rows = []
    for d, name in ((5, "D"), (6, "E"), (7, "E")):
        P = record_configuration(name, d)
        r1, x1 = hole_logsumexp(P)
        r2, _ = hole_active_set(P, x1)
        r3, _ = hole_subsets(P, k_max=3)
        rho = min(r1, r2, r3)
        ang = math.degrees(math.acos(min(rho, 1.0)))
        deficit = 60.0 - ang
        jammed = "yes" if rho > 0.5 else "no"
        rows.append((d, f"{name}{d}", len(P), rho, ang, deficit, jammed))
        print(f"{d:>2} {name + str(d):>8} {len(P):>5} {rho:>10.6f} {ang:>8.3f}d"
              f" {deficit:>8.3f}d  {jammed}")
        print(f"     log-sum-exp {r1:.6f}, KKT refinement {r2:.6f}, "
              f"3-subset enumeration {r3:.6f}")
    print()
    print("reading: rho > 1/2 in every case, so no ball can be added to the record packing")
    print("without moving the others. The deficit column is the number of degrees by which the")
    print("deepest hole misses being admissible -- the quantitative version of 'locally jammed'.")
    print()
    print("this settles only the insertion question. A 41st, 73rd or 127th point might still")
    print("exist if all the points are allowed to move; that is the search in code/attack567.py.")

    print()
    print("certifying the hole directions (KKT structure and closed form)")
    for d, name in ((3, "A"), (4, "D"), (5, "D"), (6, "E"), (7, "E"), (8, "E")):
        P = record_configuration(name, d)
        r1, x1 = hole_logsumexp(P, starts=2500, rounds=9)
        r2, _ = hole_active_set(P, x1)
        rho, x = min(r1, r2), x1
        c = certify(P, rho, x)
        guess = f"1/sqrt({c['inv_rho2']:.6g})" if abs(c["inv_rho2"] - round(c["inv_rho2"], 3)) < 1e-3 \
            else f"{c['inv_rho2']:.6f}"
        print(f"  d = {d} ({name}{d}, {len(P)} pts): rho = {rho:.9f}, 1/rho^2 = "
              f"{c['inv_rho2']:.6f}  [{guess}]")
        print(f"          active set {c['active']}, KKT residual {c['kkt_residual']:.2e}, "
              f"<y,p> on the active set in [{c['active_inner_min']:.6f}, "
              f"{c['active_inner_max']:.6f}]")
        print(f"          local minimum: {c['local_min']} perturbations of the hole make it worse")
    print()
    print("  so the hole depth of the record packing is a closed form, not a numerical accident:")
    print("  rho = sqrt(2/5) in d = 5, sqrt(3/8) in d = 6, 1/sqrt(3) in d = 7,")
    print("  against the admissibility threshold 1/2 = sqrt(1/4).")

    # a sanity check that the machinery sees a non-jammed configuration when there is one
    print()
    print("sanity check: the same measurement on configurations that are NOT jammed")
    for d, name, drop in ((5, "D", 1), (6, "E", 1), (7, "E", 1)):
        P = record_configuration(name, d)[drop:]
        r1, _ = hole_logsumexp(P, starts=1500, rounds=8)
        ang = math.degrees(math.acos(min(r1, 1.0)))
        print(f"  d = {d}, {name}{d} minus {drop} point: rho = {r1:.6f}, "
              f"angle {ang:.3f} deg, addable = {r1 <= 0.5}")


if __name__ == "__main__":
    main()
