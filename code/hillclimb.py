"""The cut-and-hill-climb experiment of README section 4a, kept as a record.

Hill-climbing 2 f1/f0 by deleting vertices of the 600-cell climbs past the published
record (13.15 against 12.61).  The climb is worthless -- every hull it visits fails the
tangency test in code/polytopes.py -- and this script exists so that the mistake cannot
be made again without being noticed.  Do not quote its output as a bound.

usage: python code/hillclimb.py
"""
import itertools, numpy as np, sys
from scipy.spatial import ConvexHull
PHI=(1+5**0.5)/2
def par(p): return sum(1 for i in range(len(p)) for j in range(i+1,len(p)) if p[i]>p[j])%2
def cell600():
    pts=set()
    for i in range(4):
        for s in (1.0,-1.0):
            v=[0.0]*4; v[i]=s; pts.add(tuple(v))
    for sg in itertools.product((0.5,-0.5),repeat=4): pts.add(tuple(sg))
    base=[0.0, 1/(2*PHI), 0.5, PHI/2]
    for perm in itertools.permutations(range(4)):
        if par(perm): continue
        v=[base[p] for p in perm]
        for sg in itertools.product((1.0,-1.0),repeat=4):
            pts.add(tuple(round(sg[i]*v[i],12) for i in range(4)))
    return np.array(sorted(pts))
def ratio(P):
    try:
        h=ConvexHull(P)
    except Exception:
        return None
    used=np.unique(h.simplices); f0=len(used)
    facets=set(frozenset(s) for s in h.simplices)
    edges=set()
    for f in facets:
        for pr in itertools.combinations(sorted(f),2): edges.add(pr)
    return 2*len(edges)/f0, f0, len(edges), len(facets)
P=cell600(); best=ratio(P); print("start", best, flush=True)
keep=list(range(len(P)))
for step in range(1,41):
    cand=None
    for idx in range(len(keep)):
        Q=P[[keep[j] for j in range(len(keep)) if j!=idx]]
        r=ratio(Q)
        if r and (cand is None or r[0]>cand[0][0]): cand=(r,idx)
    if cand is None: break
    r,idx=cand
    if r[0]<=best[0]+1e-9:
        print("no further improvement at step",step,"best kappa",round(best[0],5)); break
    best=r; keep.pop(idx)
    print(f"step {step}: kappa={r[0]:.5f} f0={r[1]} f1={r[2]} f3={r[3]}", flush=True)
print("FINAL", round(best[0],5), "vs record 12.61285 -- NOT a bound:")
print("  every hull on this climb has facet distances from the origin spread over 12-16%,")
print("  so it is not tangent to any sphere and kappa = 2 f1/f0 does not apply.")
