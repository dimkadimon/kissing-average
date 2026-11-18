"""Can the 126 points of E_7 be rearranged to open a 60-degree hole?

This is the sharpest form of the question for d = 5, 6, 7.  Write

    rho(P) = min_{|x|=1} max_{p in P} <x, p>

(the "hole depth" of code/holes.py).  An (N+1)-point configuration exists **iff** some N-point
configuration P has rho(P) <= 1/2, because then x can be added.  So instead of searching over
N+1 points, we can search over N points and try to increase rho:

    maximise rho(P)   over 60-degree-separated P with |P| = N.

The record configuration has rho = 0.5774 (d=5), 0.6124 (d=6), 0.5774 (d=7), all far above
1/2.  If a rearrangement could push rho down to 1/2 we would have a new lower bound; if rho
stays above 1/2 for every P we meet, that is evidence (not proof) that the record is optimal.

Method: alternating minimax.  Given P, find the deepest hole x (log-sum-exp descent, exact
refinement); then repel the points that are closest to x, i.e. propose p <- normalise(p - eta*x)
for the points with <p,x> within a factor of the maximum, accepting a move only if P stays
60-degree separated and the hole gets deeper.  This is a local attack on the maximin, and its
outcome is reported as the best rho reached together with the number of accepted moves.

usage:
    python code/widen.py --dim 7 --budget 300
"""
import argparse
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from holes import hole_logsumexp, hole_active_set, record_configuration   # noqa: E402


def min_angle_inner(P):
    S = P @ P.T
    np.fill_diagonal(S, -9.0)
    return float(S.max())


def rho_of(P, starts=300, rounds=7):
    r, x = hole_logsumexp(P, starts=starts, rounds=rounds)
    r2, _ = hole_active_set(P, x)
    return min(r, r2), x


def repair(Q, d, tol=1e-10, rounds=10, maxiter=400):
    """Relax an almost-feasible configuration back to <= 1/2 by minimising the smooth max.

    Returns the repaired configuration, or None if it cannot be brought within tol."""
    from attack567 import smooth_max, max_inner
    from scipy.optimize import minimize
    Y = Q.ravel().copy()
    beta = 40.0
    for _ in range(rounds):
        r = minimize(smooth_max, Y, args=(beta, len(Q), d), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, maxfun=2 * maxiter))
        Y = np.asarray(r.x)
        U = Y.reshape(len(Q), d)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        Y = U.ravel()
        beta = min(beta * 2.0, 20000.0)
    if max_inner(Y, len(Q), d) <= 0.5 + tol:
        return Y.reshape(len(Q), d)
    return None


def push(P, d, budget, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best_rho, best_x = rho_of(P)
    bestP = P.copy()
    moves = 0
    while time.time() - t0 < budget:
        rho, x = best_x, None
        rho, x = rho_of(bestP, starts=200, rounds=6)
        if rho > 0.5 + 1e-9:
            # already open enough?  (means an (N+1)-point configuration exists)
            pass
        # points closest to the hole are the ones to move
        proj = bestP @ x
        order = np.argsort(-proj)
        cand = None
        for k in (2, 3, 5, 8, 14, 24):
            for eta in (0.25, 0.1, 0.04, 0.015, 0.006):
                Q = bestP.copy()
                for i in order[:k]:
                    v = Q[i] - eta * x
                    n = np.linalg.norm(v)
                    if n > 1e-9:
                        Q[i] = v / n
                if min_angle_inner(Q) <= 0.5 + 1e-9:
                    cand = Q
                    break
                if min_angle_inner(Q) <= 0.56:
                    # almost feasible: let everything move to restore separation
                    R = repair(Q, d)
                    if R is not None:
                        cand = R
                        break
            if cand is not None:
                break
        if cand is None:
            Q = bestP + 0.02 * rng.normal(size=bestP.shape)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            if min_angle_inner(Q) > 0.5 + 1e-9:
                r = repair(Q, d)
                if r is None:
                    continue
                Q = r
            cand = Q
        rho2, x2 = rho_of(cand, starts=200, rounds=6)
        if rho2 < rho - 1e-7:
            bestP, best_rho, best_x = cand, rho2, x2
            moves += 1
            if verbose:
                print(f"  move {moves}: rho = {rho2:.6f}  "
                      f"(min angle {math.degrees(math.acos(min(1.0, min_angle_inner(cand)))):.3f} deg, "
                      f"{time.time() - t0:.0f}s)", flush=True)
        else:
            attempts_since = locals().get("attempts_since", 0) + 1
            if rng.random() < 0.3:
                bestP = cand
            if attempts_since % 25 == 0:
                # restart from the incumbent with a shake rather than giving up
                Q2 = bestP + 0.01 * rng.normal(size=bestP.shape)
                Q2 /= np.linalg.norm(Q2, axis=1, keepdims=True)
                if min_angle_inner(Q2) <= 0.5 + 1e-9:
                    bestP = Q2
    return best_rho, bestP, moves


def diagnostic(P, d, seed=0, tries=40):
    """Is the record configuration a local maximum of rho?

    Perturb the configuration slightly (staying feasible), and see which way rho moves.  If rho
    can only go up, the configuration is locally *rigid* with respect to opening a hole, which
    is the sharpest numerical statement one can make without a proof."""
    rng = np.random.default_rng(seed)
    r0, _ = rho_of(P, starts=300, rounds=7)
    print(f"  rho at the record configuration: {r0:.6f}")
    for noise in (0.005, 0.02, 0.05, 0.1, 0.2):
        vals = []
        for _ in range(tries):
            Q = P + noise * rng.normal(size=P.shape)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            if min_angle_inner(Q) > 0.5 + 1e-9:
                R = repair(Q, d)
                if R is None:
                    continue
                Q = R
            r, _ = rho_of(Q, starts=120, rounds=5)
            vals.append(r)
        if vals:
            a = np.array(vals)
            print(f"  noise {noise:>5.3f}: rho in [{a.min():.6f}, {a.max():.6f}], "
                  f"{int((a < r0 - 1e-6).sum())}/{len(a)} perturbations reduce rho")
    print("  (a reduction to <= 0.5 would mean an extra point fits)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diagnostic", action="store_true")
    args = ap.parse_args()
    d = args.dim
    name = {5: "D", 6: "E", 7: "E"}[d]
    P = record_configuration(name, d)
    r0, _ = rho_of(P, starts=400, rounds=8)
    print(f"d = {d}: {name}{d} with N = {len(P)};  rho = {r0:.6f} "
          f"(min angle {math.degrees(math.acos(min(1.0, r0))):.3f} deg)")
    print("using rho <= 1/2 as the criterion: an (N+1)-point configuration exists iff we")
    print("can push rho down to 1/2 and keep the points 60-degree separated.")
    print()
    if args.diagnostic:
        diagnostic(P, d, seed=args.seed)
        return
    best, Q, moves = push(P, d, args.budget, seed=args.seed)
    ang = math.degrees(math.acos(min(1.0, min_angle_inner(Q))))
    print()
    print(f"best rho reached: {best:.6f} after {moves} accepted moves")
    print(f"  that is a hole of angle {math.degrees(math.acos(min(1.0, best))):.3f} deg")
    print(f"  the configuration itself has minimum angle {ang:.3f} deg")
    if best <= 0.5:
        print("  rho <= 1/2: a configuration with one more point EXISTS -- save it and check it")
    else:
        print(f"  rho stays above 1/2 by {best - 0.5:.6f} "
              f"({60.0 - math.degrees(math.acos(min(1.0, best))):.3f} deg of slack):")
        print("  no extra point fits into any configuration this search reached.")


if __name__ == "__main__":
    main()
