"""Numbers quoted in Section 4.3 of the post, where the Lanczos method is compared with
textbook Gram-Schmidt on the powers q_1, A q_1, A^2 q_1, ...

    python -m checks.gram_schmidt_on_powers    # from the folder of the post's code

For a diagonal matrix of size 1000 with eigenvalues drawn uniformly from [0, 1] and a random
unit starting vector, it reports
- the length of the new direction of A^j q_1 and of A q_j, relative to the length of the vector;
- the loss of orthogonality ||Q_m^T Q_m - I|| of the basis built from the powers, and of the
  basis built by the Lanczos recurrence, without reorthogonalization and with the full
  reorthogonalization of mvp.lanczos (two projections per step, as in Algorithm 2).
"""

from pathlib import Path

import numpy as np

from figures.common import unit_gaussian, write_summary
from mvp.lanczos import lanczos

NAME = "gram-schmidt-on-powers"
SUMMARIES = Path(__file__).resolve().parents[1] / "summaries"


def gram_schmidt_on_powers(apply_a, q1: np.ndarray, steps: int):
    """Modified Gram-Schmidt on q_1, A q_1, ..., A^(steps-1) q_1, each power formed from the
    previous one and rescaled, which does not change its direction. Returns the basis and, for
    each power A^j q_1, the length of its new direction relative to its own length."""
    basis = [q1]
    power = q1.copy()
    new_fraction = []
    for _ in range(steps - 1):
        power = apply_a(power)
        power /= np.linalg.norm(power)
        remainder = power.copy()
        for q in basis:
            remainder -= (q @ remainder) * q
        new_fraction.append(np.linalg.norm(remainder))
        basis.append(remainder / np.linalg.norm(remainder))
    return np.array(basis).T, np.array(new_fraction)


def orthogonality_loss(basis: np.ndarray) -> float:
    return float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), 2))


def main() -> None:
    rng = np.random.default_rng(0)
    eigenvalues = rng.uniform(0.0, 1.0, 1000)

    def apply_a(x):
        return eigenvalues * x

    q1 = unit_gaussian(eigenvalues.size, rng)

    powers_basis, power_fraction = gram_schmidt_on_powers(apply_a, q1, 31)
    alphas, betas, lanczos_basis = lanczos(apply_a, q1, 201, reorthogonalize=False)
    _, _, reorthogonalized_basis = lanczos(apply_a, q1, 201, reorthogonalize=True)
    lanczos_fraction = [betas[j - 1] / np.linalg.norm(apply_a(lanczos_basis[:, j - 1])) for j in range(1, 31)]

    summary = {
        "new direction of A^j q_1 relative to its length": {j: power_fraction[j - 1] for j in (1, 5, 10, 20)},
        "new direction of A q_j relative to its length": {j: lanczos_fraction[j - 1] for j in (1, 5, 10, 20)},
        "orthogonality loss of the basis from the powers": {
            m: orthogonality_loss(powers_basis[:, :m]) for m in (10, 20, 25, 30)},
        "orthogonality loss of the Lanczos basis": {
            m: orthogonality_loss(lanczos_basis[:, :m]) for m in (10, 20, 25, 30, 100, 150, 200)},
        "orthogonality loss of the Lanczos basis with full reorthogonalization": {
            m: orthogonality_loss(reorthogonalized_basis[:, :m]) for m in (30, 100, 150, 200)},
    }
    write_summary(NAME, SUMMARIES, summary)
    for key, values in summary.items():
        print(key)
        for index, value in values.items():
            print(f"  {index:4d}  {value:.1e}")


if __name__ == "__main__":
    main()
