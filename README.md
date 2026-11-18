# Average kissing numbers: what our methods give, and what they cannot

**Short answer.** They do not transfer. The class-and-coverage machinery of
[arXiv:2611.01906](https://arxiv.org/abs/2611.01906) is an engine for *counting points* in a
lifted configuration; the average kissing number of that configuration is stuck at the level
of a single piece of it, because the lift's three parts make no contacts between them. The
average kissing number problem is a statement about **packings**, and the only route to a
lower bound is a *finite packing with a high average degree* — a different object from
anything in our paper. This repository records that conclusion with exact arithmetic rather
than prose, and states plainly which parts of it are proofs and which are measurements.

Two positive findings survive, and they are the reason this repository exists:

1. **A new exact mechanism, `code/lift.py`.** Every cross inner product in the P48p lift is
   *strictly* below 1/2 — at most 3/8 between shell and blocks, at most 3/8 between blocks,
   exactly 0 against the tail. Hence the contact graph of the lift is a disjoint union of
   three parts, and the lift cannot give an average-degree bound better than the best of its
   parts. This also answers a question the p48 construction literature does not seem to have
   asked, and it explains *why* the lift is so good at point counts: it maximises the number
   of points while guaranteeing that they do not touch each other more than necessary.
2. **The class is a 60° code, and we can measure its contact structure.** The 7,193-line
   class, rescaled to unit vectors, is a spherical 2-distance set: inner products are 1/3
   (which *is* a contact), 0 and -1/2. Measured exactly: 1,047,889 contacts, degrees 213 to
   470, mean degree 291.36. It is a genuine 60-degree code of size 7,193 in S^47, far above
   the 712 lines that a Caro-Wei-type bound guarantees, and it is a natural object for the
   *kissing-number* problem in dimension 48 — but for the average problem it is not usable,
   because a class is not a packing.

## 1. What the literature says (as of 19 November 2026)

| what | where | key content |
|---|---|---|
| the problem, and $\kappa_3\ge 666/53 = 12.566$ | Kuperberg and Schramm, *Average kissing numbers for non-congruent sphere packings*, Math. Res. Lett. **1** (1994) 339–344 | first non-trivial bounds; the shell/cap argument giving $\kappa_3\le 8+4\sqrt3 = 14.928$ |
| $\kappa_3 \ge 7656/607 = 12.6129$ | Eppstein, Kuperberg and Ziegler (2003), via arXiv:math/0204007 | a compound of cut 600-cells; still the dimension-3 record |
| $\kappa_n \le 2\tau_n$ always; $\kappa_4<34.681$, $\kappa_5<77.757$ | Glazyrin, *Contact graphs of ball packings*, arXiv:1707.02526 | the shell argument in all dimensions with $\mathrm{dens}\le 1$ |
| $\kappa_3\le13.606$, …, $\kappa_9\le722.629$ | Dostert, Kolpakov and de Oliveira Filho, Israel J. Math. **247** (2022) 635–659; arXiv:2003.11832 | semidefinite bounds on cap-packing density; first improvements on $2\tau_n$ in $6\le n\le9$ |
| **$\kappa_3\le13.327$ … $\kappa_{30}\le1{,}654{,}023.6$; $\kappa_8=240$, $\kappa_{24}=196560$** | Bockhaus and Glazyrin, arXiv:2610.36675 (27 Oct 2026) | current record upper bounds in $3\le n\le30$; the problem is **solved in dimensions 8 and 24** |

`code/records.py` holds the record table with sources and checks it for internal consistency
(`results/records.txt`). Two features of that table matter for anyone planning to work here:
the current upper bounds are roughly **half** of the old $2\tau_n$ in every dimension, and
the lower bounds are mostly *lattice kissing numbers* — for $4\le n\le30$ nobody has a lower
bound above the lattice one except where a finite construction exists ($n=3$). So the gap
narrows from below only by building finite packings, and the interesting dimensions are the
ones where the ratio upper/lower is worst:

    n = 13: 2233.68 / 918  = 2.43      n = 15: 5431.32 / 2340 = 2.32
    n = 17: 12219.4 / 5346 = 2.29      n = 19: 25902.5 / 10668 = 2.43

## 2. Why the P48p lift cannot help (exact)

A lifted point has one of three shapes; the cross inner products are bounded by the lift's own
hypotheses ($\langle x,y\rangle\le1/2$ inside the shell, $\le1/3$ inside a class,
$\le1/2$ between distinct tail pairs):

| pair | bound | value |
|---|---|---|
| shell vs block | $(3/4)\cdot(1/2)$ | $3/8 = 0.375$ |
| block vs block | $(3/4)\cdot(1/3) + (1/4)\cdot(1/2)$ | $3/8 = 0.375$ |
| shell vs tail | $0$ | $0$ |
| block vs tail | $0$ | $0$ |
| **within a block, at inner product exactly $1/3$** | $(3/4)(1/3) + 1/4$ | $1/2$ — a contact |

`code/lift.py` checks these as rational identities (`results/lift.txt`). The consequence is
immediate: the average degree of a disjoint union is a weighted average of the parts, so

$$\bar d(\text{lift}) \le \max\{\bar d(\text{shell}), \bar d(\text{blocks}), \bar d(\text{tail})\},$$

and no choice of class, images or tail changes that. The 52-million-point shell is what
carries the number, and the shell's average contact number is a property of $P_{48p}$, not of
anything we search. **Conclusion: the lift's advantage is anti-correlated with what the
average problem rewards.** It maximises $52{,}416{,}000 + 2S + 2t$ by making the images
*overlap as little as possible* in the line set; the average problem wants them to overlap as
much as possible, i.e. touching.

## 3. The one bridge that does exist, and how far it goes

The shell bound is

$$\kappa_n \;\le\; \frac{q(g)}{A_{n,\rho}(1)},\qquad q(g)=\frac{2\,(G-2)}{G-2-L},$$

with $G$ the contact multiplicity of the chord $g$ (a Delsarte/Levenshtein answer),
$L$ the forced number of common neighbours, and $A$ the cap fraction on the shell of radius
$\rho$. At $n=3$, $\rho=\sqrt3$, $G=3$, $L=0$ this gives $q=6$ and

$$\kappa_3 \le \frac{6}{(2-\sqrt3)/4} = 8+4\sqrt3 = 14.9282\ldots,$$

the Kuperberg–Schramm bound. `code/shell.py` verifies every ingredient as an exact identity
(`results/shell_selftest.txt`): $\cos^2\theta = 3/4$, $A = (2-\sqrt3)/4$, and
$A(r)+A(1/r)$ is minimised at $r=1$ — checked at $r=\tfrac13,\tfrac12,2,3$. It also reproduces
Glazyrin's dimension 4 and 5 numbers, $34.681$ and $77.757$, by the same code, which is the
test that the cap arithmetic is right.

So the bridge is: *one exact, verified implementation of the classical shell bound*, in exact
arithmetic, in arbitrary dimension, ready to be fed a better $q$. What it is **not** is a new
bound. Any improvement has to come from one of three places, and this repository says
honestly what each costs:

- **A better $q$ (local argument).** Needs the contact graph of a configuration of *unequal*
  balls to have a forced structure — an $L$-intersecting or face-size argument sharper than
  $G=3$. This is where the recent progress has been, and it is a proof, not a search.
- **A better cap-packing density $den_n(\rho)$.** Dostert–Kolpakov–de Oliveira Filho and
  Bockhaus–Glazyrin do this with semidefinite programming; it needs an SDP solver and exact
  certificates, which is a project in itself.
- **A better lower bound (the only search-shaped route).** Build a finite packing whose
  contact graph has a high average degree: junk balls of many radii, all touching as much as
  possible. The record in $n=3$ is $7656/607$, from a compound of cut 600-cells.

## 4. A tower of regular contact graphs (a small, real observation)

`code/graphs.py` computes contact graphs of root systems exactly. The 60-degree contact graph
of the kissing configuration is **regular** in the dimensions where the configuration is a
root system, with degree $2(d-1)$:

| configuration | points | contacts | degree | regular? |
|---|---:|---:|---:|---|
| $A_3$ (cuboctahedron) | 12 | 24 | 4 | yes |
| $D_4$ (24-cell) | 24 | 96 | 8 | yes |
| $D_5$ | 40 | 240 | 12 | yes |
| $E_7$ | 126 | 2,016 | 32 | yes |
| $E_8$ | 240 | 6,720 | 56 | yes |

Regularity is what makes a contact identity close: in the $F_4\to E_6$ step ($12\to24$) it is
the single hypothesis with which $24 = \frac{12\cdot4}{2}$, i.e. one contact per point, turns
the "deleted points" of a lift into the "new points". The same computation gives the coverage
number of the two systems (largest number of directions within 60° of one point): at most 5 of
12 for $A_3$, at most 9 of 24 for $D_4$ — measured by search, upper bounds not proved, and
recorded here as an open target rather than a result.

## 5. What is certified, and what is only measured

This matters more than usual here, because an average-kissing *lower* bound is a statement
about one explicit packing, and every decimal in it has to be rounded the right way.

**Exact, checked in this repository**

- the cross-inner-product bounds of the P48p lift, and hence its decomposition (§2): rational
  identities, `code/lift.py`;
- the cap identity and the dimension-3 Kuperberg–Schramm value $8+4\sqrt3$ (§3):
  `code/shell.py`;
- contact graphs and degrees of $A_3,D_4,D_5,E_7,E_8$ (§4): exact integer arithmetic;
- the class contact graph, 1,047,889 contacts, degrees 213–470: exact integer arithmetic.

**Measured only (no proof here)**

- the coverage numbers 5/12 and 9/24 (§4): lower bounds from a search over lattice
  directions, not optima;
- the record table's lower bounds for $n\ge9$: taken from the citations, not re-derived.

**Not attempted**

- any new upper bound, and therefore any improvement on
  $\kappa_3\le13.327,\ldots,\kappa_{30}\le1{,}654{,}023.6$ (Bockhaus–Glazyrin);
- any new lower bound. Nothing in this repository beats $7656/607$, and the honest reading of
  §2 is that our own machinery cannot get close.

## 6. Where a real attack would start

If this project is to produce a bound, the order of work that seems right to us:

1. **Reproduce the $n=3$ record exactly.** Take the Eppstein–Kuperberg–Ziegler compound of
   $n$ cut 600-cells, build its contact graph, and certify $7656/607 = 12.6129$ in exact
   arithmetic. That is a finite, checkable computation, and it gives this repository a lower
   bound of its own to stand on before any attempt at improvement.
2. **Search the same compound family.** The record is a formula in $n$,
   $\frac{2(666n+54)}{106n+14}\to\frac{666}{53}$; the question is whether a different
   compound — a corona, a chain, a tree of cut cells with unequal radii — has a higher average
   degree. This is a small, well-defined optimisation over finite packings.
3. **Only then** consider $n\ge4$, where the lower bounds are lattice kissing numbers and the
   first construction to beat one is publishable. The $D_4$ cell is the natural candidate: 24
   points, contact graph 8-regular, and the corresponding cut-cell compound has not, as far as
   the record table shows, been evaluated.

## Files

```
code/shell.py    cap fractions, the shell bound, exact selftest of 8+4sqrt3
code/lift.py     exact cross inner products of the P48p lift; its contact graph splits
code/graphs.py   exact contact graphs of root systems, degrees, coverage measures
code/records.py  the dated record table and its consistency checks
results/         the outputs quoted above
```

Licence: MIT (see `LICENSE`). Third-party data used in `data/` keeps its own terms.

## References

1. G. Kuperberg and O. Schramm, *Average kissing numbers for non-congruent sphere packings*,
   Math. Res. Lett. **1** (1994) 339–344.
2. D. Eppstein, G. Kuperberg and G. M. Ziegler, *Fat 4-polytopes and fatter 3-spheres*,
   arXiv:math/0204007 (2002/2003).
3. A. Glazyrin, *Contact graphs of ball packings*, arXiv:1707.02526; J. Combin. Theory Ser. A.
4. M. Dostert, A. Kolpakov and F. M. de Oliveira Filho, *Semidefinite programming bounds for
   the average kissing number*, Israel J. Math. **247** (2022) 635–659; arXiv:2003.11832.
5. M. Bockhaus and A. Glazyrin, *New upper bounds for average kissing numbers*,
   arXiv:2610.36675 (2026).
6. H. Cohn, *Table of kissing number bounds*, <https://cohn.mit.edu/kissing-numbers/>.
7. D. Kamenetsky, *Improved lower bounds for the kissing number in dimensions 49 to 61*,
   arXiv:2611.01906 (2026) — the paper whose machinery is examined in §2.
