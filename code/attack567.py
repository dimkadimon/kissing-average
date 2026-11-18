"""Attack on dimensions 5, 6, 7: can 41, 73 or 127 points be placed at mutual angle >= 60 deg?

The problem, in unit-vector form: find N unit vectors in R^d with <y_i,y_j> <= 1/2 for all
i != j.  Equivalently find N points on S^{d-1} with minimum angle >= 60 degrees.  The record
is 40 (d=5), 72 (d=6), 126 (d=7) and the best known upper bounds are 43, 76, 132.

code/holes.py settles the *insertion* question: the record configurations are locally jammed by
5.3, 7.8 and 9.2 degrees, so no 41st/73rd/127th point fits into them.  This module attacks the
harder question -- can everything move?

Method: minimise the smooth maximum
    F_beta(Y) = (1/beta) log sum_{i<j} exp(beta <y_i,y_j>)
over the product of spheres, with L-BFGS, then anneal beta upward.  F_beta -> max inner
product.  Two things make the search credible:
  * it is *validated on the cases where the answer is known* -- it must find 12 in d=3, 24 in
    d=4, 40 in d=5, 72 in d=6, 126 in d=7 from random starts;
  * the interesting case is N = record + 1, and a run that fails is reported as a failure,
    with the best (largest) minimum angle it reached, so the size of the failure is visible.

A configuration with all inner products <= 1/2 *is* a valid packing, so a success in this search
would be a genuine improvement.  Failures prove nothing by themselves -- they are evidence, and
they are labelled as such.

usage:
    python code/attack567.py --dims 3 4 5 --budgets 20 60
    python code/attack567.py --dims 5 6 7 --budgets 60 240 --extra 1
"""
import argparse
import math
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, ".")
from holes import hole_logsumexp, hole_active_set, record_configuration   # noqa: E402


def smooth_max(Y, beta, N, d):
    U = Y.reshape(N, d)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    S = U @ U.T
    iu = np.triu_indices(N, 1)
    s = S[iu]
    m = s.max()
    w = np.exp(beta * (s - m))
    Z = w.sum()
    val = m + math.log(Z) / beta
    W = np.zeros((N, N))
    W[iu] = w / Z
    W = W + W.T
    G = (W @ U) * 2.0
    return val, G.ravel()


def max_inner(Y, N, d):
    U = Y.reshape(N, d)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    S = U @ U.T
    np.fill_diagonal(S, -9.0)
    return float(S.max())


def anneal(Y0, N, d, beta0=20.0, rounds=10, growth=1.8, maxiter=400):
    Y = Y0.ravel().copy()
    beta = beta0
    best = max_inner(Y, N, d)
    bestY = Y.copy()
    for _ in range(rounds):
        r = minimize(smooth_max, Y, args=(beta, N, d), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, maxfun=2 * maxiter))
        Y = np.asarray(r.x)
        # renormalise, then measure the honest objective
        U = Y.reshape(N, d)
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        Y = U.ravel()
        cur = max_inner(Y, N, d)
        if cur < best:
            best, bestY = cur, Y.copy()
        beta = min(beta * growth, 20000.0)
    return best, bestY


def starts_for(N, d, rng, seeds):
    """Random starts plus structured ones (record configuration, record + hole points)."""
    out = []
    for _ in range(3):
        Y = rng.normal(size=(N, d))
        out.append((Y / np.linalg.norm(Y, axis=1, keepdims=True)).ravel())
    for P in seeds:
        if P.shape[0] == N:
            out.append(P.ravel())
        elif P.shape[0] == N - 1:
            # add a point at the deepest hole of the seed
            _, x = hole_logsumexp(P, starts=400, rounds=6)
            out.append(np.vstack([P, x]).ravel())
        elif P.shape[0] > N:
            keep = rng.choice(P.shape[0], N, replace=False)
            out.append(P[keep].ravel())
    return out


def run(d, N, budget, seed, seeds):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = np.inf
    bestY = None
    tries = 0
    while time.time() - t0 < budget:
        for Y0 in starts_for(N, d, rng, seeds):
            if time.time() - t0 > budget:
                break
            tries += 1
            val, Y = anneal(Y0, N, d)
            # polish with a few more rounds at high beta from the best point seen
            if val < best:
                val2, Y2 = anneal(Y, N, d, beta0=200.0, rounds=6, maxiter=600)
                if val2 < val:
                    val, Y = val2, Y2
                best, bestY = val, Y
    return dict(d=d, N=N, best=best, tries=tries, seconds=time.time() - t0,
                angle=math.degrees(math.acos(min(best, 1.0))), Y=bestY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="*", default=[5, 6, 7])
    ap.add_argument("--budgets", type=float, nargs="*", default=[20, 60],
                    help="seconds: first for N = record, then for N = record + extra")
    ap.add_argument("--extra", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("search for N points on S^{d-1} at mutual angle >= 60 deg (max inner product <= 1/2)")
    print(f"budgets: {args.budgets} s;  'best angle' is the largest minimum angle reached")
    print()
    print(f"{'d':>2} {'N':>5} {'target':>8} {'best max<.,.>':>14} {'best angle':>11} "
          f"{'tries':>6}  verdict")
    for d in args.dims:
        name = {3: "A", 4: "D", 5: "D", 6: "E", 7: "E"}.get(d)
        try:
            P = record_configuration(name, d)
        except Exception:
            P = None
        seeds = [] if P is None else [P]
        rec = 0 if P is None else P.shape[0]
        for tag, N in [("record", rec)] + [(f"+{k}", rec + k) for k in range(1, args.extra + 1)]:
            budget = args.budgets[0] if tag == "record" else args.budgets[-1]
            r = run(d, N, budget, args.seed + N, seeds)
            ok = r["best"] <= 0.5 + 1e-7
            if ok:
                verdict = "FEASIBLE -- improvement!" if N > rec else "recovered the record"
            else:
                verdict = (f"infeasible in {r['tries']} tries, short by "
                           f"{r['angle'] - 60.0:+.2f} deg")
            print(f"{d:>2} {N:>5} {tag:>8} {r['best']:>14.6f} {r['angle']:>10.3f}d "
                  f"{r['tries']:>6}  {verdict}", flush=True)
    print()
    print("a feasible row would be a new lower bound and would need the explicit configuration")
    print("saved; every infeasible row is evidence, not proof. The local statement in")
    print("code/holes.py is the rigorous half of the argument.")


if __name__ == "__main__":
    main()
