from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "public"
    / "images"
    / "blog"
    / "grokking"
    / "parameter-norm-reparameterization.svg"
)

L_VALUES = np.arange(1, 11)
SCENARIOS = (
    (0.5, 8.0, r"$a<b$", r"$a=0.5,\ b=8$"),
    (1.0, 1.0, r"$a=b$", r"$a=b=1$"),
    (7.0, 0.25, r"$a>b$", r"$a=7,\ b=0.25$"),
)


def w(lam: np.ndarray | float, a: float, b: float, degree: int):
    return a * np.asarray(lam) ** 2 + b / np.asarray(lam) ** (2 * degree)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.edgecolor": "#56615d",
            "axes.labelcolor": "#27322f",
            "axes.titlecolor": "#27322f",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "xtick.color": "#44504c",
            "ytick.color": "#44504c",
            "grid.color": "#d6ddda",
            "grid.linewidth": 0.7,
            "svg.hashsalt": "parameter-norm-reparameterization",
        }
    )

    lam = np.geomspace(0.03, 30.0, 1_600)
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=L_VALUES.min(), vmax=L_VALUES.max())

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.9),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.patch.set_facecolor("#fbfaf7")

    for axis, (a, b, relation, values) in zip(axes, SCENARIOS):
        axis.set_facecolor("#fbfaf7")

        for degree in L_VALUES:
            color = cmap(norm(degree))
            axis.plot(lam, w(lam, a, b, degree), color=color, linewidth=1.65)

            lam_min = (degree * b / a) ** (1 / (2 * degree + 2))
            axis.scatter(
                [lam_min],
                [w(lam_min, a, b, degree)],
                s=20,
                color=color,
                edgecolor="#fbfaf7",
                linewidth=0.8,
                zorder=4,
            )

        axis.axvline(1, color="#7d8783", linewidth=1, linestyle=(0, (3, 3)))
        axis.set_title(f"{relation}\n{values}", pad=9)
        axis.set_xlabel(r"rescaling $\lambda$ (log scale)")
        axis.set_xscale("log")
        axis.set_xlim(0.03, 30.0)
        axis.set_yscale("log")
        axis.set_ylim(0.5, 100_000)
        axis.xaxis.set_major_locator(
            mpl.ticker.FixedLocator([0.03, 0.1, 0.3, 1, 3, 10, 30])
        )
        axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        axis.grid(True, which="major", alpha=0.8)
        axis.grid(True, which="minor", alpha=0.25)
        axis.text(
            0.025,
            0.965,
            r"$\lambda\to0^+$: $w(\lambda)\to\infty$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#44504c",
            fontsize=9,
        )
        axis.text(
            0.975,
            0.965,
            r"$\lambda\to\infty$: $w(\lambda)\to\infty$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#44504c",
            fontsize=9,
        )

    axes[0].set_ylabel(r"squared parameter norm $w(\lambda)$")
    fig.suptitle(
        r"$w(\lambda)=a\lambda^2+b/\lambda^{2L}$ under an equivalent reparameterization",
        fontsize=15,
        fontweight="semibold",
        color="#27322f",
    )

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        ticks=[1, 2, 4, 6, 8, 10],
        fraction=0.025,
        pad=0.025,
    )
    colorbar.set_label(r"homogeneity degree $L$", color="#27322f")
    colorbar.ax.tick_params(colors="#44504c")
    colorbar.outline.set_edgecolor("#56615d")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        format="svg",
        facecolor=fig.get_facecolor(),
        metadata={"Date": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
