"""algorithm-3-flow.svg: the execution of Algorithm 3 (Arnoldi with reorthogonalization and residual checks).

The same diagram as for Algorithm 2, with the Arnoldi step and the restart
case of step 6, whose blue arrow returns to the setup with a new starting vector.
"""

from .algorithm2_flow import draw
from .common import save

NAME = "algorithm-3-flow"


def make(out_dir, preview=None) -> dict:
    save(draw(arnoldi=True), NAME, out_dir, preview)
    return {}
