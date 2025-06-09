"""pylucid bindings for Python."""

import os

if os.name == "nt" and os.environ.get("GUROBI_HOME", "") != "":
    # Windows
    os.add_dll_directory(os.path.join(os.environ.get("GUROBI_HOME", ""), "bin"))

from ._pylucid import *
from ._pylucid import __version__ as __pylucid_version__, __doc__ as __pylucid_doc__
from .ext import *
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Set
    import numpy as np

__version__ = __pylucid_version__
__doc__ = __pylucid_doc__


@dataclass(frozen=True)
class ScenarioConfig:
    x_samples: "np.typing.NDArray[np.float64]"
    xp_samples: "np.typing.NDArray[np.float64]"
    X_bounds: "Set"
    X_init: "Set"
    X_unsafe: "Set"
    T: int = 5
    gamma: float = 1.0
    f_det: "Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]] | None" = None
    estimator: "Estimator | None" = None
    num_freq_per_dim: int = -1
    feature_map: "FeatureMap | None" = None
    sigma_f: float = 1.0
    verify: bool = True
    plot: bool = True
    problem_log_file: str = ""
    iis_log_file: str = ""

    def keys(self) -> "list[str]":
        """Returns a list of keys for the configuration attributes."""
        return [
            "x_samples",
            "xp_samples",
            "X_bounds",
            "X_init",
            "X_unsafe",
            "T",
            "gamma",
            "f_det",
            "estimator",
            "num_freq_per_dim",
            "feature_map",
            "sigma_f",
            "verify",
            "plot",
            "problem_log_file",
            "iis_log_file",
        ]

    def __getitem__(self, key) -> "np.typing.NDArray[np.float64] | Set | int | float | str | None":
        return getattr(self, key)


# Initial verbosity level. Can be changed later.
set_verbosity(LOG_INFO)
