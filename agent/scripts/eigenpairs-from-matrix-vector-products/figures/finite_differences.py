"""finite-differences.svg: the discretization of u -> -u'' of Section 2.2.

Left: a function u on [0, 1] with u(0) = u(1) = 0, sampled at the interior grid
points t_i = ih, which gives the vector x; the stencil at i = 3 combines x_2,
x_3 and x_4 with the weights -1, 2, -1. Right: the same combination is row 3 of
the tridiagonal matrix A, applied to x.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .common import BLUE, GREY, INK, ORANGE, save

NAME = "finite-differences"
D = 7
ROW = 3  # the stencil drawn in both panels
HIGHLIGHT = "#e9f0fe"


def u(t):
    return np.sin(np.pi * t) * (1.0 + 0.7 * t)


def function_panel(axis):
    h = 1.0 / (D + 1)
    t = np.linspace(0, 1, 400)
    grid = h * np.arange(D + 2)
    axis.plot(t, u(t), color=GREY, linewidth=1.6, zorder=1)
    axis.axhline(0, color="#56615d", linewidth=0.9, zorder=0)
    for i, ti in enumerate(grid):
        axis.plot([ti, ti], [0, u(ti)], color="#c9d0cd", linewidth=0.8, zorder=0)
    interior = grid[1:-1]
    axis.plot(interior, u(interior), "o", color=BLUE, markersize=6.5, zorder=3)
    axis.plot(grid[[0, -1]], [0, 0], "o", markerfacecolor="white", markeredgecolor=ORANGE,
              markeredgewidth=1.6, markersize=6.5, zorder=3)
    for i in range(1, D + 1):
        axis.text(grid[i], -0.13, rf"$t_{i}$", ha="center", va="top", color=INK, fontsize=10)
    axis.text(grid[0], -0.13, r"$t_0$", ha="center", va="top", color=INK, fontsize=10)
    axis.text(grid[-1], -0.13, rf"$t_{{{D + 1}}}$", ha="center", va="top", color=INK, fontsize=10)
    axis.text(grid[0] + 0.01, 0.1, r"$x_0=0$", color=ORANGE, fontsize=10, ha="left", va="bottom")
    axis.text(grid[-1] - 0.01, 0.1, rf"$x_{{{D + 1}}}=0$", color=ORANGE, fontsize=10, ha="right",
              va="bottom")
    for i in (1, 6):
        axis.text(grid[i] + 0.015, u(grid[i]) + 0.05, rf"$x_{i}=u(t_{i})$", color=BLUE, fontsize=10,
                  ha="left" if i == 1 else "left", va="bottom")
    # The stencil at t_3: weights -1, 2, -1, divided by h^2.
    for offset, weight in ((-1, "$-1$"), (0, "$+2$"), (1, "$-1$")):
        ti = grid[ROW + offset]
        axis.plot([ti], [u(ti)], "o", markerfacecolor="none", markeredgecolor=INK, markersize=12,
                  markeredgewidth=1.3, zorder=4)
        axis.text(ti, u(ti) + 0.12, weight, ha="center", va="bottom", color=INK, fontsize=10.5,
                  fontweight="bold")
    axis.text(0.5, 1.93, rf"$-u''(t_{ROW})\approx\dfrac{{-x_{ROW - 1}+2x_{ROW}-x_{ROW + 1}}}{{h^2}}=(\mathbf{{A}}\mathbf{{x}})_{ROW}$",
              ha="center", va="center", color=INK, fontsize=11.5)
    # The step h between two grid points.
    y_h = -0.42
    axis.annotate("", xy=(grid[5], y_h), xytext=(grid[4], y_h),
                  arrowprops=dict(arrowstyle="<|-|>", color=INK, linewidth=0.9, mutation_scale=8,
                                  shrinkA=0, shrinkB=0))
    axis.text((grid[4] + grid[5]) / 2, y_h - 0.05, "$h$", ha="center", va="top", color=INK, fontsize=10.5)
    axis.text(0.98, 1.38, r"$u$", color=GREY, fontsize=12, ha="right")
    axis.set_xlim(-0.06, 1.06)
    axis.set_ylim(-0.62, 2.15)
    axis.axis("off")


def matrix_panel(axis):
    cell = 0.28
    x0, y0 = 0.0, 0.5  # lower-left corner of A
    top = y0 + D * cell
    axis.text(x0 - 0.1, y0 + D * cell / 2, r"$\mathbf{A}=\dfrac{1}{h^2}$", ha="right", va="center",
              color=INK, fontsize=11.5)
    # Highlight row 3 of A and the entries of x and Ax that it involves.
    axis.add_patch(Rectangle((x0, top - ROW * cell), D * cell, cell, facecolor=HIGHLIGHT, edgecolor="none",
                             zorder=0))
    for i in range(D):
        for j in range(D):
            value = 2 if i == j else (-1 if abs(i - j) == 1 else 0)
            axis.add_patch(Rectangle((x0 + j * cell, top - (i + 1) * cell), cell, cell, facecolor="none",
                                     edgecolor="#d6ddda", linewidth=0.6, zorder=1))
            if value:
                axis.text(x0 + (j + 0.5) * cell, top - (i + 0.5) * cell, f"{value:d}".replace("-", "−"),
                          ha="center", va="center", color=INK, fontsize=9.5, zorder=2)
    axis.add_patch(Rectangle((x0, y0), D * cell, D * cell, facecolor="none", edgecolor="#56615d",
                             linewidth=1.0, zorder=2))
    # The vector x, then the result Ax.
    xv = x0 + D * cell + 0.2
    for name, xpos, rows in (("x", xv, "x"), ("Ax", xv + cell + 0.5, "(Ax)")):
        for i in range(D):
            involved = (rows == "x" and abs(i + 1 - ROW) <= 1) or (rows != "x" and i + 1 == ROW)
            axis.add_patch(Rectangle((xpos, top - (i + 1) * cell), cell, cell,
                                     facecolor=HIGHLIGHT if involved else "white",
                                     edgecolor="#56615d" if involved else "#d6ddda", linewidth=0.8, zorder=1))
            text = rf"$x_{i + 1}$" if rows == "x" else rf"$\cdot$"
            axis.text(xpos + cell / 2, top - (i + 0.5) * cell, text if rows == "x" else "",
                      ha="center", va="center", color=BLUE if rows == "x" else INK, fontsize=9.5, zorder=2)
        axis.text(xpos + cell / 2, top + 0.08,
                  r"$\mathbf{x}$" if rows == "x" else r"$\mathbf{A}\mathbf{x}$", ha="center", va="bottom",
                  color=INK, fontsize=11)
    axis.text(xv + cell + 0.25, y0 + D * cell / 2, "=", ha="center", va="center", color=INK, fontsize=13)
    ax_col = xv + cell + 0.5
    axis.text(ax_col + cell + 0.08, top - (ROW - 0.5) * cell, rf"$(\mathbf{{A}}\mathbf{{x}})_{ROW}$",
              ha="left", va="center", color=INK, fontsize=10)
    axis.text(x0 + D * cell / 2, top + 0.08, r"row $3$: weights $-1,\,2,\,-1$", ha="center", va="bottom",
              color=INK, fontsize=9.5)
    axis.text(x0 + D * cell / 2, y0 - 0.12, r"rows $1$ and $d$: no column for $x_0=x_{d+1}=0$",
              ha="center", va="top", color=ORANGE, fontsize=9.2)
    axis.set_xlim(*PANEL_X)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")


PANEL_X = (-0.95, 3.95)  # inches covered by the matrix panel
HEIGHT = 3.2
LEFT_WIDTH = 4.1


def make(out_dir, preview=None) -> dict:
    width = LEFT_WIDTH + PANEL_X[1] - PANEL_X[0] + 0.1
    fig = plt.figure(figsize=(width, HEIGHT))
    fig.patch.set_facecolor("white")
    left = fig.add_axes([0.01, 0.03, LEFT_WIDTH / width, 0.94])
    right = fig.add_axes([(LEFT_WIDTH + 0.1) / width, 0.0, (PANEL_X[1] - PANEL_X[0]) / width, 1.0])
    function_panel(left)
    matrix_panel(right)
    save(fig, NAME, out_dir, preview)
    return {"d": D, "row": ROW}
