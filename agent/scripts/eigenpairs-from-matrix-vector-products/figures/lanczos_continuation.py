"""lanczos-continuation.svg: the structure of the basis and of T_d in Proposition 4.

An example with d = 9 and three runs of lengths 4, 3 and 2. Left: the columns of
Q_9, coloured by run; each run breaks down (beta = 0) and the next one starts
from a vector orthogonal to all the previous ones. Right: T_9, whose tridiagonal
blocks are the matrices of the runs, joined by the zero couplings beta_4 = beta_7 = 0.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from .common import BLUE, GREEN, GREY, INK, PURPLE, save

NAME = "lanczos-continuation"
RUNS = (4, 3, 2)
COLORS = (BLUE, GREEN, PURPLE)
FILLS = ("#e9f0fe", "#e8f5ec", "#f3effd")
STARTS = (r"$\mathbf{q}_1$", r"$\mathbf{P}_4\mathbf{g}$", r"$\mathbf{P}_7\mathbf{g}$")


def basis_panel(axis):
    width, gap, height = 0.3, 0.07, 1.65
    x, y0 = 0.15, 0.85
    index = 0
    for run, (length, color, fill, start) in enumerate(zip(RUNS, COLORS, FILLS, STARTS)):
        x_run = x
        for _ in range(length):
            index += 1
            axis.add_patch(FancyBboxPatch((x, y0), width, height, boxstyle="round,pad=0,rounding_size=0.05",
                                          facecolor=fill, edgecolor=color, linewidth=1.2))
            axis.text(x + width / 2, y0 + height / 2, rf"$\mathbf{{q}}_{index}$", ha="center", va="center",
                      color=INK, fontsize=10, rotation=0)
            x += width + gap
        x_end = x - gap
        axis.plot([x_run, x_end], [y0 - 0.12, y0 - 0.12], color=color, linewidth=1.6)
        axis.text((x_run + x_end) / 2, y0 - 0.2, f"run {run + 1}\nfrom " + start, ha="center", va="top",
                  color=color, fontsize=9.5)
        axis.text(x_end + gap / 2, y0 + height + 0.1, rf"$\beta_{index}=0$", ha="center", va="bottom",
                  color=GREY, fontsize=9.5)
        x += 0.22
    axis.text(0.15, y0 + height + 0.5, r"$\mathbf{Q}_9$", ha="left", va="bottom", color=INK, fontsize=12)
    axis.text(0.62, y0 + height + 0.54, "each run starts orthogonally to all the earlier vectors",
              ha="left", va="bottom", color=INK, fontsize=9.5)
    axis.set_xlim(0, 4.3)
    axis.set_ylim(0, 3.6)
    axis.set_aspect("equal")
    axis.axis("off")


def tridiagonal_panel(axis):
    d = sum(RUNS)
    cell = 0.3
    x0, y0 = 0.35, 0.3
    top = y0 + d * cell
    run_of = [run for run, length in enumerate(RUNS) for _ in range(length)]
    boundaries = {sum(RUNS[:r + 1]) for r in range(len(RUNS) - 1)}  # last index of each run but the last
    for i in range(1, d + 1):
        for j in range(1, d + 1):
            xc, yc = x0 + (j - 0.5) * cell, top - (i - 0.5) * cell
            same_run = run_of[i - 1] == run_of[j - 1]
            text, fill, color = "", "white", INK
            if i == j:
                text, fill = rf"$\alpha_{i}$", FILLS[run_of[i - 1]]
            elif abs(i - j) == 1 and same_run:
                text, fill = rf"$\beta_{min(i, j)}$", FILLS[run_of[i - 1]]
            elif abs(i - j) == 1 and min(i, j) in boundaries:
                text, fill, color = "$0$", "#eef0ef", GREY
            axis.add_patch(Rectangle((xc - cell / 2, yc - cell / 2), cell, cell, facecolor=fill,
                                     edgecolor="#d6ddda", linewidth=0.6))
            if text:
                axis.text(xc, yc, text, ha="center", va="center", color=color, fontsize=8.6)
    start = 0
    for length, color in zip(RUNS, COLORS):
        axis.add_patch(Rectangle((x0 + start * cell, top - (start + length) * cell), length * cell, length * cell,
                                 facecolor="none", edgecolor=color, linewidth=1.5))
        start += length
    axis.text(x0 + d * cell / 2, top + 0.1, r"$\mathbf{T}_9$, with $\mathbf{A}\mathbf{Q}_9=\mathbf{Q}_9\mathbf{T}_9$",
              ha="center", va="bottom", color=INK, fontsize=11)
    axis.text(x0 + d * cell + 0.15, top - 4 * cell, r"$\beta_4=0$", ha="left", va="center", color=GREY,
              fontsize=9.5)
    axis.text(x0 + d * cell + 0.15, top - 7 * cell, r"$\beta_7=0$", ha="left", va="center", color=GREY,
              fontsize=9.5)
    axis.set_xlim(0, 3.8)
    axis.set_ylim(0, 3.6)
    axis.set_aspect("equal")
    axis.axis("off")


def make(out_dir, preview=None) -> dict:
    fig = plt.figure(figsize=(8.2, 3.6))
    fig.patch.set_facecolor("white")
    basis_panel(fig.add_axes([0.0, 0.0, 4.3 / 8.2, 1.0]))
    tridiagonal_panel(fig.add_axes([4.4 / 8.2, 0.0, 3.8 / 8.2, 1.0]))
    save(fig, NAME, out_dir, preview)
    return {"runs": list(RUNS)}
