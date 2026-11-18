"""Figure for the repository: what the objects look like, and where the numbers sit.

Panel (a): the kissing configuration in R^3.  Twelve unit balls tangent to a central unit
ball, centres at distance 2 on the vertices of a cuboctahedron.  Among the twelve there are
24 tangencies -- the contact graph is 4-regular -- and that regularity is the property that
makes contact identities close in higher dimensions.  The average contact number of these 13
balls is 2*36/13 = 5.54: a kissing configuration is a *local* optimum, not a packing with a
high average.

Panel (b): one octahedral hole of fcc, to scale.  Six fcc balls of radius 1/sqrt2 surround a
ball of radius 1 - 1/sqrt2, which touches exactly those six and nothing else.  code/packings.py
shows exactly that filling every octahedral hole of fcc leaves the average contact number at
12, unchanged: the interstitial trick is neutral, which is why the dimension-3 record of
7656/607 = 12.6129 needs the cut-600-cell compound of Eppstein-Kuperberg-Ziegler.

Panel (c): the numbers.  fcc, fcc with the holes filled, the record, the trivial upper bound
2*tau_3 = 24 and the current upper bound 13.327 of Bockhaus-Glazyrin.

usage:
    python code/figure3d.py            # writes figures/fig_3d.png
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = "figures/fig_3d.png"
BLUE, RED, GREY, DARK = "#2b5aa6", "#b22222", "#9aa0a6", "#1c1c1c"


def unit_sphere(ax, c, r, color, alpha=0.45, res=26, lw=0.0):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=lw, shade=True,
                    antialiased=True)


def kissing_3d():
    """Unit vectors of the 12 contact points: (+-e_i +-e_j)/sqrt2, i<j."""
    P = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(3)
                    v[i], v[j] = si, sj
                    P.append(v / math.sqrt(2))
    return np.array(P)


def panel_kissing(ax):
    P = kissing_3d()
    C = 2.0 * P                      # centres at distance 2: tangent to the unit ball at 0
    unit_sphere(ax, np.zeros(3), 1.0, GREY, alpha=0.30, res=30)
    for c in C:
        unit_sphere(ax, c, 1.0, BLUE, alpha=0.42)
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            if abs(np.linalg.norm(C[i] - C[j]) - 2.0) < 1e-9:
                ax.plot(*zip(C[i], C[j]), color=DARK, lw=1.1, alpha=0.85)
    ax.set_title("(a) kissing configuration in $\\mathbb{R}^3$\n"
                 "12 unit balls on the central one, 24 mutual tangencies (4-regular)",
                 fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    lim = 3.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=18, azim=32)
    ax.set_axis_off()


def panel_hole(ax):
    """One octahedral hole, to scale: 6 fcc balls of radius 1/sqrt2 at distance 1 around a
    ball of radius 1-1/sqrt2."""
    R = 1 / math.sqrt(2)
    rh = 1 - R
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    C = dirs * (R + rh)
    assert abs(np.linalg.norm(C[0]) - (R + rh)) < 1e-12
    unit_sphere(ax, np.zeros(3), rh, RED, alpha=0.95, res=22)
    for c in C:
        unit_sphere(ax, c, R, BLUE, alpha=0.35)
        ax.plot(*zip(np.zeros(3), c), color=DARK, lw=1.2, alpha=0.9)
    ax.set_title("(b) one octahedral hole of fcc, to scale\n"
                 "the red ball touches exactly 6 blue ones, and no other red ball",
                 fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=20, azim=40)
    ax.set_axis_off()


def panel_numbers(ax):
    labels = ["fcc", "fcc +\nholes", "record\n(7656/607)", "upper bound\n(2026)", "trivial\n$2\\tau_3$"]
    vals = [12.0, 12.0, 7656 / 607, 13.327, 24.0]
    colors = [BLUE, RED, "#2e7d32", GREY, "#cccccc"]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for i, (b, v) in enumerate(zip(bars, vals)):
        txt = f"{v:.4f}" if i in (2, 3) else f"{v:g}"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, txt, ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("average contact number")
    ax.set_ylim(0, 26.5)
    ax.axhline(12.6129, ls=":", lw=1, color="#2e7d32")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("(c) where the numbers sit in $\\mathbb{R}^3$\n"
                 "filling the holes of fcc changes nothing: both give exactly 12",
                 fontsize=10)


def main():
    fig = plt.figure(figsize=(15, 5.4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133)
    panel_kissing(ax1)
    panel_hole(ax2)
    panel_numbers(ax3)
    fig.suptitle("Average kissing number in three dimensions: the object, the obstruction, "
                 "and the gap", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT, dpi=190)
    fig.savefig(OUT.replace(".png", ".pdf"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
