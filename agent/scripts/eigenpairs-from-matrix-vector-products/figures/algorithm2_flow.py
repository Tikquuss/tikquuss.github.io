"""algorithm-2-flow.svg: the execution of Algorithm 2 (Lanczos with reorthogonalization and residual checks).

Steps 1 and 2 run once. Steps 3 to 6 form iteration j, and the blue arrow is
the passage to iteration j + 1. Boxes with a blue fill cost products with A.
``draw`` also serves Algorithm 3 (Arnoldi), whose step 6 has a restart case.
"""

from .common import save
from .flowchart import (BLUE, GREY, PRODUCT_FILL, arrow, bottom, box, canvas, diamond, group, label,
                        left, right, terminal, top)

NAME = "algorithm-2-flow"
EDGE = "#56615d"


def draw(arnoldi: bool = False):
    restart_rows = 0.9 if arnoldi else 0.0
    width_in = 7.8 if arnoldi else 7.6
    fig, axis = canvas(width_in, 10.5 + restart_rows)
    xc, width = 3.35, 4.3
    skip_lane, loop_lane, restart_lane = 0.85, 0.4, 7.5
    lift = restart_rows  # everything above step 6 moves up to make room for the restart case
    frame_right = 7.65 if arnoldi else 7.45

    group(axis, 0.15, 9.3 + lift, frame_right, 10.42 + lift, "Setup, once")
    setup_text = (r"fix the routine $\mathbf{x}\mapsto\mathbf{A}\mathbf{x}$, draw $\mathbf{q}_1=\mathbf{g}/\Vert\mathbf{g}\Vert$,"
                  + (r" set $j=1$" if arnoldi else r" set $\beta_0=0$, $j=1$"))
    setup = box(axis, xc, 9.8 + lift, width, 0.72, "1–2. Set up", [setup_text])
    group(axis, 0.15, 0.05, frame_right, 9.12 + lift, r"Iteration $j\geq1$, steps 3 to 6")

    if arnoldi:
        extend_lines = [r"$\mathbf{w}=\mathbf{A}\mathbf{q}_j-\mathbf{Q}_j\mathbf{h}$ with $\mathbf{h}=\mathbf{Q}_j^\top\mathbf{A}\mathbf{q}_j$,"
                        r" projected twice",
                        r"$h_{j+1,j}=\Vert\mathbf{w}\Vert$; if $h_{j+1,j}\leq\tau$, breakdown: $h_{j+1,j}=0$"]
        small, residual, pairs = r"\mathbf{H}_j", r"h_{j+1,j}", "wanted"
        check_title = r"5. Check with $\mathbf{A}$ ($k$ to $2k$ products)"
        quotient = r"$\widehat{\lambda}_i=\mathbf{x}_i^*\mathbf{A}\mathbf{x}_i$"
        next_vector = r"$\mathbf{q}_{j+1}=\mathbf{w}/h_{j+1,j}$"
    else:
        extend_lines = [r"$\mathbf{r}_j=\mathbf{A}\mathbf{q}_j-\beta_{j-1}\mathbf{q}_{j-1}-\alpha_j\mathbf{q}_j$, reorthogonalized",
                        r"$\beta_j=\Vert\mathbf{r}_j\Vert$; if $\beta_j\leq\tau$, breakdown: $\beta_j=0$"]
        small, residual, pairs = r"\mathbf{T}_j", r"\beta_j", "requested"
        check_title = r"5. Check with $\mathbf{A}$ ($k$ products)"
        quotient = r"$\widehat{\lambda}_i=\mathbf{x}_i^\top\mathbf{A}\mathbf{x}_i$"
        next_vector = r"$\mathbf{q}_{j+1}=\mathbf{r}_j/\beta_j$"

    extend = box(axis, xc, 8.3 + lift, width, 1.0, "3. Extend the basis (one product)", extend_lines,
                 fill=PRODUCT_FILL)
    check_now = diamond(axis, xc, 7.28 + lift, 3.1, 0.78, "$j$ multiple of $p$,\nor breakdown?")
    ritz = box(axis, xc, 6.27 + lift, width, 0.98, rf"4. Ritz pairs of ${small}$ (no product)",
               [rf"the $\min(j,k)$ {pairs} pairs $(\vartheta_i,\mathbf{{y}}_i)$; candidates:",
                rf"${residual}\,|(\mathbf{{y}}_i)_j|\leq\varepsilon_{{\mathrm{{abs}}}}+\varepsilon_{{\mathrm{{rel}}}}|\vartheta_i|$"])
    enough = diamond(axis, xc, 5.27 + lift, 2.6, 0.7, "$k$ candidates?")
    explicit = box(axis, xc, 4.3 + lift, width, 0.98, check_title,
                   [r"$\mathbf{x}_i=\mathbf{Q}_j\mathbf{y}_i/\Vert\mathbf{Q}_j\mathbf{y}_i\Vert$, " + quotient + ", accept if",
                    r"$\Vert\mathbf{A}\mathbf{x}_i-\widehat{\lambda}_i\mathbf{x}_i\Vert\leq\varepsilon_{\mathrm{abs}}"
                    r"+\varepsilon_{\mathrm{rel}}\Vert\mathbf{A}\mathbf{x}_i\Vert$"], fill=PRODUCT_FILL)
    success = diamond(axis, xc, 3.3 + lift, 2.7, 0.7, "all $k$ accepted?")
    failure = diamond(axis, xc, 2.4 + lift, 2.7, 0.7, "$j$ = maximum?")
    label(axis, 1.42, success[1], "6. Stop or\ncontinue", size=9.8, fontweight="bold", ha="center", va="center")
    terminal_x, terminal_width = (6.45, 1.7) if arnoldi else (6.5, 1.8)
    done = terminal(axis, terminal_x, success[1], terminal_width, 0.6, "stop: return\nthe $k$ pairs")
    fail = terminal(axis, terminal_x, failure[1], terminal_width, 0.6, "stop: report\nnonconvergence",
                    success=False)
    if arnoldi:
        costly = diamond(axis, xc, 1.5 + lift, 2.9, 0.7, r"$\mathbf{Q}_j$ too costly?")
        restart = box(axis, terminal_x, costly[1], terminal_width, 0.62, None,
                      ["restart from the", "wanted Ritz vectors"], size=9.2)
    broke = diamond(axis, xc, 1.5, 2.4, 0.66, "breakdown?")
    same_run = box(axis, 1.85, 0.88, 1.9, 0.42, None, [next_vector])
    new_run = box(axis, 4.95, 0.88, 2.9, 0.42, None,
                  [r"new run: $\mathbf{q}_{j+1}=\mathbf{P}_j\mathbf{g}/\Vert\mathbf{P}_j\mathbf{g}\Vert$"])
    advance = box(axis, xc, 0.33, 3.5, 0.4, None, [r"$\mathbf{Q}_{j+1}=[\mathbf{Q}_j,\mathbf{q}_{j+1}]$, $j\leftarrow j+1$"])

    yes = dict(text="yes", text_color=GREY)
    no = dict(text="no", text_color=GREY)
    arrow(axis, [bottom(setup), top(extend)])
    arrow(axis, [bottom(extend), top(check_now)])
    arrow(axis, [bottom(check_now), top(ritz)], **yes, offset=(0.1, -0.2))
    arrow(axis, [bottom(ritz), top(enough)])
    arrow(axis, [bottom(enough), top(explicit)], **yes, offset=(0.1, -0.2))
    arrow(axis, [bottom(explicit), top(success)])
    arrow(axis, [right(success), left(done)], **yes, offset=(0.12, 0.05))
    arrow(axis, [bottom(success), top(failure)], **no, offset=(0.1, -0.16))
    arrow(axis, [right(failure), left(fail)], **yes, offset=(0.12, 0.05))
    if arnoldi:
        arrow(axis, [bottom(failure), top(costly)], **no, offset=(0.1, -0.16))
        arrow(axis, [right(costly), left(restart)], **yes, offset=(0.12, 0.05))
        arrow(axis, [bottom(costly), top(broke)], **no, offset=(0.1, -0.16))
        arrow(axis, [right(restart), (restart_lane, restart[1]), (restart_lane, setup[1]), right(setup)],
              color=BLUE, width=1.6)
        label(axis, restart_lane + 0.12, (restart[1] + setup[1]) / 2, "restart", color=BLUE, size=9.6,
              rotation=90, ha="center", va="center")
    else:
        arrow(axis, [bottom(failure), top(broke)], **no, offset=(0.1, -0.16))
    arrow(axis, [left(broke), (same_run[0], broke[1]), top(same_run)], **no, offset=(-0.3, 0.05))
    arrow(axis, [right(broke), (new_run[0], broke[1]), top(new_run)], **yes, offset=(0.1, 0.05))
    arrow(axis, [bottom(same_run), (same_run[0], top(advance)[1])])
    arrow(axis, [bottom(new_run), (new_run[0], top(advance)[1])])

    # Steps 4 and 5 are skipped when there is nothing to check, or too little to stop.
    arrow(axis, [left(check_now), (skip_lane, check_now[1]), (skip_lane, failure[1]), left(failure)],
          **no, offset=(-0.32, 0.05))
    axis.plot([left(enough)[0], skip_lane], [enough[1], enough[1]], color=EDGE, linewidth=1.2, zorder=1)
    axis.plot([skip_lane], [enough[1]], marker="o", markersize=3.5, color=EDGE, zorder=2)
    label(axis, left(enough)[0] - 0.32, enough[1] + 0.05, "no", color=GREY, va="bottom")

    # The passage to the next iteration.
    arrow(axis, [left(advance), (loop_lane, advance[1]), (loop_lane, extend[1]), left(extend)], color=BLUE,
          width=1.6)
    label(axis, loop_lane - 0.12, (advance[1] + extend[1]) / 2, "next iteration", color=BLUE, size=9.6,
          rotation=90, ha="center", va="center")
    return fig


def make(out_dir, preview=None) -> dict:
    save(draw(arnoldi=False), NAME, out_dir, preview)
    return {}
