"""How the pair lambda_2, conj(lambda_2) moves the walk of Figure 12 from one group to the next (Section 5.1).

The walker starts uniformly on group 1. In the basis of eigenvectors of P^T, the coefficient of
pi is 1 and that of conj(w) is conj(c), so p_k = pi + 2 Re(c lambda_2^k w) + r_k. This script
reports whether the eigenvalues are distinct, the group sums s_g = c sum_{i in g} w_i (modulus and
phase), the largest modulus among the other 57 eigenvalues, and the deviations of the group
probabilities from their stationary values after one and two steps, with the part of the pair.

    python -m checks.walk_circulation      # writes summaries/walk-circulation.json
"""

from pathlib import Path

import numpy as np

from figures.common import write_summary
from figures.directed_graph import walk

NAME = "walk-circulation"


def main() -> None:
    _, dense_p, group, _ = walk()
    pt = dense_p.T
    values, vectors = np.linalg.eig(pt)
    d = len(values)
    gaps = np.where(np.eye(d, dtype=bool), np.inf, np.abs(values[:, None] - values[None, :]))
    start = (group == 0).astype(float)
    start /= start.sum()
    coefficients = np.linalg.solve(vectors, start)
    one = int(np.argmin(np.abs(values - 1)))
    others = np.where((np.arange(d) != one) & (values.imag > 0), np.abs(values), -1.0)
    pair = int(np.argmax(others))
    pair_conjugate = int(np.argmin(np.abs(values - values[pair].conjugate())))
    lam, cw = values[pair], coefficients[pair] * vectors[:, pair]
    stationary = (coefficients[one] * vectors[:, one]).real
    s = np.array([cw[group == g].sum() for g in range(3)])
    rest = [j for j in range(d) if j not in (one, pair, pair_conjugate)]

    deviations, pair_parts, p = [], [], start.copy()
    for k in (1, 2):
        p = pt @ p
        deviations.append([(p - stationary)[group == g].sum() for g in range(3)])
        term = 2 * (lam**k * cw).real
        pair_parts.append([term[group == g].sum() for g in range(3)])

    summary = {
        "smallest_gap_between_eigenvalues": gaps.min(),
        "coefficient_of_pi": (coefficients[one] * vectors[:, one].sum()).real,
        "conjugate_coefficient_is_conjugate": bool(
            np.allclose(coefficients[pair_conjugate] * vectors[:, pair_conjugate], np.conj(cw))
        ),
        "lambda_2_modulus": abs(lam),
        "lambda_2_angle_degrees": np.degrees(np.angle(lam)),
        "max_modulus_other_57": np.abs(values[rest]).max(),
        "group_sum_moduli": np.abs(s),
        "group_sum_phases_degrees": np.degrees(np.angle(s)),
        "group_deviations_steps_1_2": deviations,
        "pair_part_steps_1_2": pair_parts,
    }
    write_summary(NAME, Path(__file__).resolve().parents[1] / "summaries", summary)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
