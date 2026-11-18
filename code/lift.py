"""Can the P48p block lift produce average-kissing lower bounds?  An exact answer.

Our kissing-number paper lifts a 52,416,000-point shell by t isometric images of a class
and a tail with t antipodal pairs.  The natural hope is that the same configuration is
worth something for the *average* contact number: it has 52 million points, so its average
degree is what a finite packing of that size can achieve.

This module computes the contact graph of the lift exactly, and the answer is a clean
negative.  Writing a lifted point in the three shapes of Theorem 2.5 (Yang et al.),

    shell   (x, 0)                    x a kept minimal direction
    block   ((sqrt3/2) y, eps/2 u_i)  y in a class block B_i, eps = +-1
    tail    (0, w)                    w in the tail configuration

every cross inner product is *strictly below 1/2*, so the contact graph of the lifted
configuration is a disjoint union of three pieces with no edges between them:

    shell vs block : (sqrt3/2)<x,y>            <= (sqrt3/2)(1/2)      = 0.4330 < 1/2
    block vs block : (3/4)<y,y'> + (1/4)<u_i,u_j> <= 3/8              = 0.3750 < 1/2
    shell vs tail  : 0
    block vs tail  : 0

(using <y,y'> <= 1/3 inside a class and <u_i,u_j> <= 1/2 between distinct antipodal pairs
of a kissing configuration -- both are the hypotheses of the lift).  So the configuration is
a kissing configuration but a *poor* packing for average degree: all the potential contacts
between its three parts are wasted.

The average degree of a disjoint union is the weighted average of the parts' average
degrees, so it is at most the largest of the three.  For the shell part that is the average
contact number of a large chunk of the minimal shell of P48p, which is the same order as the
kissing number of the lattice -- far above anything that matters in dimension 48, but far
*below* what the p48 lift is good for, and it has nothing to do with the class search that
made the paper work.  Conclusion: the lift is an engine for *point counts*, not for
*average contact numbers*.

usage:
    python code/lift.py            # exact inequalities and the decomposition
    python code/lift.py --class    # also measure the class contact graph, if available
"""
from fractions import Fraction as F
import sys


def cross_bounds():
    """The exact upper bounds on cross inner products, as Fractions."""
    shell_class = F(3, 4) * F(1, 2)          # (sqrt3/2)^2 = 3/4, <x,y> <= 1/2 in the shell
    block_block = F(3, 4) * F(1, 3) + F(1, 4) * F(1, 2)
    return {
        "shell vs block  <= (3/4)(1/2)": shell_class,
        "block vs block  <= (3/4)(1/3)+(1/4)(1/2)": block_block,
        "shell vs tail   <= 0": F(0),
        "block vs tail   <= 0": F(0),
    }


def contact_threshold():
    """Contacts need inner product exactly 1/2 (unit vectors)."""
    return F(1, 2)


def check():
    out = []
    for name, v in cross_bounds().items():
        ok = v < contact_threshold()
        out.append((name, str(v), ok))
        assert ok, name
    # within a block, <y,y'> = 1/3 is exactly a contact: (3/4)(1/3) + (1/4)(1) = 1/2
    within = F(3, 4) * F(1, 3) + F(1, 4)
    assert within == F(1, 2), within
    out.append(("within block, <y,y'> = 1/3 : (3/4)(1/3)+1/4", str(within), within == F(1, 2)))
    # and <y,y'> = 0 is not
    no = F(3, 4) * F(0) + F(1, 4)
    out.append(("within block, <y,y'> = 0", str(no), no < F(1, 2)))
    return out


def class_contacts(path="data/class_p48_7193.npy"):
    """The contact graph of a stored class: pairs with inner product exactly 1/3.

    This is the only part of the lift whose contact structure the class search controls,
    and it is worth measuring: a class whose contact graph is dense contributes more to
    average degree, but the lift's count formula does not use that at all.
    """
    import numpy as np
    try:
        X = np.load(path)
    except OSError:
        return None
    G = None
    for cand in ("data/external/p48p_gram.npy", "../data/external/p48p_gram.npy"):
        try:
            G = np.load(cand).astype(np.int64)
            break
        except OSError:
            continue
    if G is None:
        return None
    deg = np.zeros(len(X), dtype=np.int64)
    for s in range(0, len(X), 2000):
        blk = X[s:s + 2000].astype(np.int64)
        M = blk @ (X @ G).T          # <c_i, c_j> * 6 since norms are 6
        # norms are 6, so <c,c'> = cGc'^T / 6 and a contact at 60 degrees is cGc'^T = 2
        deg[s:s + 2000] = (M == 2).sum(axis=1)
    deg = deg - 0
    return dict(n=int(len(X)), edges=int(deg.sum() // 2),
                min_deg=int(deg.min()), max_deg=int(deg.max()),
                mean_deg=float(deg.mean()))


if __name__ == "__main__":
    for name, val, ok in check():
        print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {val}")
    print()
    print("  -> the contact graph of the lift splits into shell | blocks | tail,")
    print("     with no edges between the parts.")
    if "--class" in sys.argv:
        c = class_contacts()
        print()
        if c is None:
            print("  class file not present; run from a checkout with data/ populated")
        else:
            print(f"  class contact graph: {c['n']} lines, {c['edges']} contacts,"
                  f" degree {c['min_deg']}..{c['max_deg']}, mean {c['mean_deg']:.2f}")
