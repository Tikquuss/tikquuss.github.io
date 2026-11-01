"""The residual bound without a basis of eigenvectors (Remark 20 of Section 5.6).

For A = V J V^{-1} in Jordan form, with p the size of the largest Jordan block, the remark states

    min_lambda |z - lambda| <= max(eta, eta^(1/p)),    eta = p kappa(V) ||A x - z x||,

for every unit vector x and every z. This script checks it on random matrices with blocks of size up to
p = 4, at points z between 1e-3 and 10 from the spectrum, using for x the unit vector with the smallest
residual (the last right singular vector of A - z I), and reports the largest ratio of the distance to the
bound, which must not exceed 1. It also reports, for the block J = [[0, 1], [0, 0]], z = eps and
x = [1, eps] / sqrt(1 + eps^2), the norm of the residual, the distance eps, and the bound sqrt(2 * residual).

    python -m checks.jordan_residual_bound      # writes summaries/jordan-residual-bound.json
"""

from pathlib import Path

import numpy as np

from figures.common import write_summary

NAME = "jordan-residual-bound"


def jordan_matrix(sizes, eigenvalues):
    """Block diagonal J with the given block sizes and eigenvalues, ones just above the diagonal."""
    d = sum(sizes)
    j = np.zeros((d, d), dtype=complex)
    start = 0
    for size, value in zip(sizes, eigenvalues):
        for i in range(size):
            j[start + i, start + i] = value
            if i + 1 < size:
                j[start + i, start + i + 1] = 1.0
        start += size
    return j


def main() -> None:
    rng = np.random.default_rng(0)

    two_by_two = []
    j2 = np.array([[0.0, 1.0], [0.0, 0.0]])
    for eps in (1e-1, 1e-2, 1e-3):
        x = np.array([1.0, eps]) / np.hypot(1.0, eps)
        residual = np.linalg.norm(j2 @ x - eps * x)
        two_by_two.append({
            "eps": eps,
            "residual": residual,
            "residual_over_eps_squared": residual / eps**2,
            "distance_over_residual": eps / residual,
            "bound_sqrt_2_residual": np.sqrt(2 * residual),
        })

    worst = {}
    for p in (1, 2, 3, 4):
        ratios = []
        for _ in range(200):
            sizes = [p] + list(rng.integers(1, p + 1, size=rng.integers(0, 4)))
            eigenvalues = rng.normal(size=len(sizes)) + 1j * rng.normal(size=len(sizes))
            d = sum(sizes)
            v = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            a = v @ jordan_matrix(sizes, eigenvalues) @ np.linalg.inv(v)
            kappa = np.linalg.cond(v)
            for _ in range(20):
                # A point z at distance between 1e-3 and 10 from a chosen eigenvalue.
                angle = rng.uniform(0, 2 * np.pi)
                z = rng.choice(eigenvalues) + 10 ** rng.uniform(-3, 1) * np.exp(1j * angle)
                distance = np.abs(eigenvalues - z).min()
                smallest_residual = np.linalg.svd(a - z * np.eye(d), compute_uv=False)[-1]
                eta = p * kappa * smallest_residual
                ratios.append(distance / max(eta, eta ** (1 / p)))
        worst[f"p={p}"] = max(ratios)

    summary = {
        "two_by_two_jordan_block": two_by_two,
        "largest_distance_over_bound": worst,
    }
    write_summary(NAME, Path(__file__).resolve().parents[1] / "summaries", summary)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
