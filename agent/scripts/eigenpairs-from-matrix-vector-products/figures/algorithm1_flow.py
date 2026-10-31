"""algorithm-1-flow.svg: the execution of Algorithm 1 (deflated squared power iteration with sign separation).

Two loops: the inner loop of steps 1 to 4 iterates on the deflated matrix of the
stage (blue arrow), and each accepted pair starts the next stage (green arrow).
The two products of an inner step are tinted with the colours of h_1 and h_2 in
the post.
"""

from .common import save
from .flowchart import (BLUE, GREEN, GREY, PRODUCT_FILL, arrow, bottom, box, canvas, diamond, group,
                        label, left, right, terminal, top)

NAME = "algorithm-1-flow"
SECOND_PRODUCT_FILL = "#fff0e6"


def make(out_dir, preview=None) -> dict:
    fig, axis = canvas(7.6, 10.9)
    xc, width = 3.35, 4.3
    inner_lane, stage_lane, side_x = 0.88, 0.4, 6.45

    group(axis, 0.15, 9.92, 7.45, 10.82, "Setup, once")
    setup = box(axis, xc, 10.33, width, 0.62, None,
                [r"tolerances $\varepsilon_{\mathrm{abs}},\varepsilon_{\mathrm{rel}}$; $\mathbf{Q}$ with no column, "
                 r"so $\mathbf{P}=\mathbb{I}_d$"])
    group(axis, 0.15, 0.05, 7.45, 9.75, "Stage: one eigenpair")
    start = box(axis, xc, 9.15, width, 0.5, None,
                [r"start: $\mathbf{w}=\mathbf{P}\mathbf{z}/\Vert\mathbf{P}\mathbf{z}\Vert$, "
                 r"$\mathbf{z}\sim\mathcal{N}(0,\mathbb{I}_d)$"])
    group(axis, 0.62, 2.72, 7.37, 8.72, r"Inner loop on $\mathbf{B}=\mathbf{P}\mathbf{A}\mathbf{P}$, steps 1 to 5")

    first = box(axis, xc, 7.98, width, 0.72, "1. First product",
                [r"$\mathbf{h}_1=\mathbf{A}\mathbf{w}$, $\mathbf{a}=\mathbf{P}\mathbf{h}_1$, $\rho=\Vert\mathbf{a}\Vert$"],
                fill=PRODUCT_FILL)
    zero = diamond(axis, xc, 7.06, 2.0, 0.62, r"$\rho=0$?")
    second = box(axis, xc, 6.15, width, 0.72, "2. Second product",
                 [r"$\mathbf{y}=\mathbf{a}/\rho$, $\mathbf{h}_2=\mathbf{A}\mathbf{y}$, $\mathbf{b}=\mathbf{P}\mathbf{h}_2$"],
                 fill=SECOND_PRODUCT_FILL)
    candidate = box(axis, xc, 5.1, width, 0.95, "3. Candidate (no product)",
                    [r"$s=+1$ if $\mathbf{w}^\top\mathbf{y}\geq0$, else $s=-1$; $n=\Vert\mathbf{w}+s\mathbf{y}\Vert$",
                     r"$\mathbf{q}=(\mathbf{w}+s\mathbf{y})/n$, $\mathbf{A}\mathbf{q}=(\mathbf{h}_1+s\mathbf{h}_2)/n$"])
    check = box(axis, xc, 4.08, width, 0.7, "4. Check or update (no product)",
                [r"$\widehat{\lambda}=\mathbf{q}^\top\mathbf{A}\mathbf{q}$, $\mathbf{r}=\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}$"])
    test = diamond(axis, xc, 3.25, 3.6, 0.78,
                   r"$\Vert\mathbf{r}\Vert\leq\varepsilon_{\mathrm{abs}}+\varepsilon_{\mathrm{rel}}\Vert\mathbf{A}\mathbf{q}\Vert$?")
    step5 = box(axis, side_x, zero[1], 1.7, 0.82, "5. Zero case", [r"$\mathbf{q}=\mathbf{w}$, $\mathbf{A}\mathbf{q}=\mathbf{h}_1$",
                                                                  "test of step 4"], size=9.2, title_size=9.8)
    accept = box(axis, xc, 2.2, 3.7, 0.5, None,
                 [r"accept $(\widehat{\lambda},\mathbf{q})$, append $\mathbf{q}$ to $\mathbf{Q}$"])
    enough = diamond(axis, xc, 1.22, 3.0, 0.72, "all desired\npairs found?")
    done = terminal(axis, side_x, enough[1], 1.7, 0.56, "stop")

    yes = dict(text="yes", text_color=GREY)
    arrow(axis, [bottom(setup), top(start)])
    arrow(axis, [bottom(start), top(first)])
    arrow(axis, [bottom(first), top(zero)])
    arrow(axis, [right(zero), left(step5)], **yes, offset=(0.12, 0.05))
    arrow(axis, [bottom(zero), top(second)], text="no", text_color=GREY, offset=(0.1, -0.2))
    arrow(axis, [bottom(second), top(candidate)])
    arrow(axis, [bottom(candidate), top(check)])
    arrow(axis, [bottom(check), top(test)])
    arrow(axis, [bottom(test), top(accept)], **yes, offset=(0.1, -0.22))
    arrow(axis, [bottom(step5), (side_x, accept[1]), right(accept)], text="passes", text_color=GREY,
          at=1, offset=(0.08, 0.1))
    arrow(axis, [bottom(accept), top(enough)])
    arrow(axis, [right(enough), left(done)], **yes, offset=(0.12, 0.05))

    # The inner loop: a failed test updates w and returns to step 1.
    arrow(axis, [left(test), (inner_lane, test[1]), (inner_lane, first[1]), left(first)], color=BLUE, width=1.6)
    label(axis, inner_lane - 0.13, (test[1] + first[1]) / 2,
          r"no: $\mathbf{w}\leftarrow\mathbf{b}/\Vert\mathbf{b}\Vert$, next inner step", color=BLUE, size=9.6,
          rotation=90, ha="center", va="center")
    # The stage loop: an accepted pair deflates A further and starts a new stage.
    arrow(axis, [left(enough), (stage_lane, enough[1]), (stage_lane, start[1]), left(start)], color=GREEN,
          width=1.6)
    label(axis, stage_lane - 0.13, (enough[1] + start[1]) / 2, "no: next stage, with the enlarged Q",
          color=GREEN, size=9.6, rotation=90, ha="center", va="center")

    save(fig, NAME, out_dir, preview)
    return {}
