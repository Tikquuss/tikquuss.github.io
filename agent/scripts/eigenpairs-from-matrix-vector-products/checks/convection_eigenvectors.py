"""Eigenvectors of the convection-diffusion matrix of Section 5.1: where they live and how far from orthogonal.

For d = 40 and gamma = 0.2 (the matrix of Figure 13), the eigenvector of A for the
k-th eigenvalue has the entries s^(i-1) sin(k pi t_i). This script reports s^(d-1),
which is kappa(V) for V = S V~, the smallest share of the squared norm of a unit
eigenvector that lies in the last quarter of the grid, the largest |cosine| between
two unit eigenvectors, and checks the eigenpairs against a dense solver.

    python -m checks.convection_eigenvectors      # writes summaries/convection-eigenvectors.json
"""

from pathlib import Path

import numpy as np

from figures.common import write_summary

NAME = "convection-eigenvectors"
D, GAMMA = 40, 0.2


def main() -> None:
    h = 1.0 / (D + 1)
    t = h * np.arange(1, D + 1)
    a = (2 * np.eye(D) - (1 + GAMMA) * np.eye(D, k=-1) - (1 - GAMMA) * np.eye(D, k=1)) / h**2
    s = np.sqrt((1 + GAMMA) / (1 - GAMMA))
    scale = s ** np.arange(D)
    k = np.arange(1, D + 1)
    values = (2 - 2 * np.sqrt(1 - GAMMA**2) * np.cos(k * np.pi * h)) / h**2
    vectors = scale[:, None] * np.sin(np.outer(t, k * np.pi))  # column k: entries s^(i-1) sin(k pi t_i)
    residual = np.abs(a @ vectors - vectors * values).max() / np.abs(a @ vectors).max()
    unit = vectors / np.linalg.norm(vectors, axis=0)
    last_quarter = (unit[3 * D // 4:] ** 2).sum(axis=0)
    cosines = np.abs(unit.T @ unit)[np.triu_indices(D, 1)]
    v_tilde = np.sin(np.outer(t, k * np.pi))
    v_tilde /= np.linalg.norm(v_tilde, axis=0)
    summary = {
        "d": D,
        "gamma": GAMMA,
        "s": s,
        "s_power_d_minus_1": s ** (D - 1),
        "kappa_S_V_tilde": np.linalg.cond(scale[:, None] * v_tilde),
        "relative_eigen_residual": residual,
        "min_share_last_quarter": last_quarter.min(),
        "max_abs_cosine": cosines.max(),
        "median_abs_cosine": np.median(cosines),
    }
    write_summary(NAME, Path(__file__).resolve().parents[1] / "summaries", summary)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
