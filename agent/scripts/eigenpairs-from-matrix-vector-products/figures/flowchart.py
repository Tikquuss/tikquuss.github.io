"""Drawing helpers for the execution diagrams of the algorithms.

The axes are measured in inches (one data unit is one inch of the figure), so
that font sizes and box sizes can be chosen together.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

from .common import BLUE, GREEN, GREY, INK, ORANGE, PURPLE

PROCESS_FILL = "#f4f6f5"
PROCESS_EDGE = "#56615d"
DECISION_FILL = "#f3effd"
DECISION_EDGE = PURPLE
SUCCESS_FILL = "#e8f5ec"
FAILURE_FILL = "#fdeee6"
GROUP_EDGE = "#9aa5a1"
PRODUCT_FILL = "#e9f0fe"

__all__ = ["BLUE", "GREEN", "GREY", "INK", "ORANGE", "PURPLE", "PRODUCT_FILL", "canvas", "box",
           "diamond", "terminal", "arrow", "group", "label"]


def canvas(width: float, height: float):
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("white")
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    return fig, axis


def box(axis, x, y, width, height, title, lines=(), fill=PROCESS_FILL, edge=PROCESS_EDGE,
        size=9.6, title_size=10.2):
    """A process box centred at (x, y): a bold title, then lines of text or mathtext."""
    axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                  boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.1,
                                  facecolor=fill, edgecolor=edge, zorder=2))
    rows = ([title] if title else []) + list(lines)
    step = 0.235 if lines else 0.0
    top = y + step * (len(rows) - 1) / 2
    for index, row in enumerate(rows):
        bold = title and index == 0
        axis.text(x, top - index * step, row, ha="center", va="center", color=INK,
                  fontsize=title_size if bold else size, fontweight="bold" if bold else "normal",
                  zorder=3)
    return (x, y, width, height)


def diamond(axis, x, y, width, height, text, size=9.6):
    """A decision, centred at (x, y)."""
    axis.add_patch(Polygon([(x, y + height / 2), (x + width / 2, y), (x, y - height / 2),
                            (x - width / 2, y)], closed=True, linewidth=1.1,
                           facecolor=DECISION_FILL, edgecolor=DECISION_EDGE, zorder=2))
    axis.text(x, y, text, ha="center", va="center", color=INK, fontsize=size, zorder=3)
    return (x, y, width, height)


def terminal(axis, x, y, width, height, text, success=True, size=9.6):
    """An end of the algorithm: green for success, orange for failure."""
    axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                  boxstyle=f"round,pad=0,rounding_size={height / 2}", linewidth=1.3,
                                  facecolor=SUCCESS_FILL if success else FAILURE_FILL,
                                  edgecolor=GREEN if success else ORANGE, zorder=2))
    axis.text(x, y, text, ha="center", va="center", color=INK, fontsize=size, zorder=3)
    return (x, y, width, height)


def arrow(axis, points, color=PROCESS_EDGE, width=1.2, text=None, at=0, offset=(0.08, 0.08),
          text_color=None, ha="left", va="bottom", size=9.0):
    """A polyline through the given points, with an arrowhead at the last one.

    ``text`` is written next to the start of segment ``at``.
    """
    xs, ys = zip(*points)
    axis.plot(xs[:-1], ys[:-1], color=color, linewidth=width, solid_capstyle="butt", zorder=1)
    axis.annotate("", xy=points[-1], xytext=points[-2],
                  arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, shrinkA=0,
                                  shrinkB=0, mutation_scale=11), zorder=1)
    if text:
        axis.text(points[at][0] + offset[0], points[at][1] + offset[1], text, color=text_color or color,
                  fontsize=size, ha=ha, va=va, zorder=3)


def group(axis, x0, y0, x1, y1, text, color=GROUP_EDGE, size=9.6):
    """A dashed frame around a group of steps, with its label in the top left corner."""
    axis.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                  boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.0,
                                  linestyle=(0, (4, 3)), facecolor="none", edgecolor=color, zorder=0))
    axis.text(x0 + 0.12, y1 - 0.1, text, ha="left", va="top", color=color, fontsize=size,
              fontstyle="italic", zorder=3)


def label(axis, x, y, text, color=INK, size=9.0, **kwargs):
    axis.text(x, y, text, color=color, fontsize=size, zorder=3, **kwargs)


def top(node):
    x, y, _, height = node
    return (x, y + height / 2)


def bottom(node):
    x, y, _, height = node
    return (x, y - height / 2)


def left(node):
    x, y, width, _ = node
    return (x - width / 2, y)


def right(node):
    x, y, width, _ = node
    return (x + width / 2, y)
