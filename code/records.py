"""The published record table this repository compares against, with sources.

Hard-coded on purpose: a bound is only "improved" relative to a dated, citable value, and
the numbers move -- the upper bounds were all replaced in the last twelve months by
Bockhaus-Glazyrin (arXiv:2610.36675, 27 Oct 2026), and several lower bounds come from
2026 repositories rather than from print.

Columns:
    n, lower, upper, source of the lower bound
The lower bound is max(lattice kissing number, best finite packing construction) as
recorded by Bockhaus-Glazyrin, Table 1.  Values are floats where the record is a limit of
finite configurations and therefore not attained.

Cross-checks performed by `python code/records.py`:
  * for n = 8 and n = 24 the lower bound equals the kissing number (attained exactly);
  * every upper bound is >= the lower bound;
  * the upper bounds beat the trivial 2 * tau_n bound, which is recorded alongside.
"""
rows = [
    # n, lower bound, new upper bound (Bockhaus-Glazyrin 2026), previous upper bound
    # (2*tau_n from Cohn's table), source of the lower bound
    (3,  12.612,   13.327,       13.606,   "Eppstein-Kuperberg-Ziegler (7656/607)"),
    (4,  24.0,     25.786,       27.439,   "lattice: D4"),
    (5,  40.0,     46.660,       64.022,   "lattice: D5"),
    (6,  72.0,     82.646,       121.105,  "lattice: E6"),
    (7,  126.0,    140.384,      223.144,  "lattice: E7"),
    (8,  240.0,    240.0,        408.386,  "lattice: E8 (solved)"),
    (9,  272.0,    380.244,      722.629,  "lattice (272)"),
    (10, 336.0,    596.184,      1106.0,   "lattice (336)"),
    (11, 438.0,    916.228,      1736.0,   "lattice (438)"),
    (12, 756.0,    1418.098,     2710.0,   "lattice: K12 (756)"),
    (13, 918.0,    2233.684,     4128.0,   "lattice (918)"),
    (14, 1422.0,   3492.338,     6348.0,   "lattice (1422)"),
    (15, 2340.0,   5431.321,     9706.0,   "lattice (2340)"),
    (16, 4320.0,   8314.473,     14640.0,  "lattice: Barnes-Wall/Leech (4320)"),
    (17, 5346.0,   12219.420,    21956.0,  "lattice (5346)"),
    (18, 7398.0,   17878.342,    32812.0,  "lattice (7398)"),
    (19, 10668.0,  25902.460,    48834.0,  "lattice (10668)"),
    (20, 17400.0,  37976.0,      72390.0,  "lattice (17400)"),
    (21, 27720.0,  55535.0,      None,     "lattice (27720)"),
    (22, 49896.0,  87592.0,      None,     "lattice (49896)"),
    (23, 93150.0,  128104.059,   244702.0, "lattice (93150)"),
    (24, 196560.0, 196560.0,     393120.0, "Leech lattice (solved)"),
    (25, 197586.0, 278379.872,   530012.0, "lattice + Kravatskiy (2026)"),
    (26, 201192.0, 396999.509,   735550.0, "Kravatskiy/Takhanov (2026)"),
    (27, 210108.0, 560721.861,   1044424.0, "Takhanov (2026)"),
    (28, 223496.0, 790758.685,   1504584.0, "Takhanov (2026)"),
    (29, 232153.0, 1130937.482,  2151982.0, "Takhanov (2026)"),
    (30, 248212.0, 1654023.607,  3075414.0, "Kravatskiy (2026)"),
]


def check():
    out = []
    for n, lo, hi, prev, src in rows:
        assert hi >= lo, (n, lo, hi)
        if prev is not None:
            assert hi < prev, ("new bound should improve on 2 tau", n, hi, prev)
        out.append((n, lo, hi, prev, (hi / prev) if prev else None))
    for n in (8, 24):
        r = [x for x in rows if x[0] == n][0]
        assert r[1] == r[2], ("solved but not tight", n)
    return out


if __name__ == "__main__":
    print(" n   lower        upper      previous     upper/previous")
    ratios = [r[4] for r in check() if r[4] is not None]
    for n, lo, hi, prev, ratio in check():
        p = f"{prev:10.1f}" if prev else "         -"
        q = f"{ratio:8.3f}" if ratio else "       -"
        print(f"{n:2d}  {lo:10.3f}  {hi:10.3f}  {p}   {q}")
    print()
    print("upper/previous ranges from %.3f to %.3f (1.000 = the problem is solved)."
          % (min(ratios), max(ratios)))
    print("uncertain records are the high-dimensional ones, where the lower bounds are")
    print("recent repository constructions rather than print.")
