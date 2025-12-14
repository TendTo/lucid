import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io
import yaml

from ._pylucid import Estimator, Parameter, Stats
from .cli import Configuration
from .pipeline import OptimiserResult
from .util import raise_error

if TYPE_CHECKING:
    from typing import Callable

    from ._pylucid import NMatrix


class ModelEstimator(Estimator):
    """Estimator for the system dynamics.

    It can be used when you have a model of the system dynamics that you want to use directly instead of learning it from data.
    Useful to debug the pipeline, as the predictions will be exactly the same as the model function.

    Args:
        f: A callable that takes a numpy array as input and returns a numpy array as output.
        params: A dictionary of parameters that can be used to configure the model.
            This is not used in this case, but may be required to match the interface of the expected Estimator class.
    """

    def __init__(self, f: "Callable[[NMatrix], NMatrix]", params: "dict[str, int | float | NVector] | None" = None):
        super().__init__()
        self._f = f
        self._params = params or {}

    def predict(self, x: "NMatrix") -> "NMatrix":
        """Predict the next state given the current state by applying the model function."""
        return self._f(x)

    def consolidate(
        self,
        training_inputs: "NMatrix",
        training_outputs: "NMatrix",
        requests: "int",
    ) -> "ModelEstimator":
        """Consolidate the model with the training data.

        Since we are using the model directly, we do not need to change anything.
        """
        return self

    def score(self, evaluation_inputs: "NMatrix", evaluation_outputs: "NMatrix") -> float:
        """Score the model based on the evaluation data.

        Since we are using the model directly, we can return a fixed score.
        """
        return 1.0

    def get(self, param: Parameter) -> "dict[str, int | float | NVector]":
        """Get the parameters of the model."""
        return self._params[param]

    def clone(self) -> "ModelEstimator":
        """Clone the estimator."""
        return ModelEstimator(self._f)

    def __str__(self) -> str:
        return f"ModelEstimator( f( {self._f.__name__} ) )"


def save_result(output_file: str, config: Configuration, stats: Stats, result: OptimiserResult) -> None:
    """Save the final results to a file in YAML, JSON, MAT or CSV format.

    Args:
        output_file: The path to the output file.
        config: The configuration used for the scenario.
        stats: The statistics of the scenario.
    """
    merged_dict = {**config.to_safe_dict(), **stats.to_dict()}
    merged_dict["run_success"] = result["success"]
    merged_dict["run_c"] = result["c"]
    merged_dict["run_eta"] = result["eta"]
    merged_dict["run_time_horizon"] = result["T"]
    merged_dict["run_safety"] = 1 - result["obj_val"]
    merged_dict["run_norm"] = result["norm"]
    merged_dict["run_sol"] = result["sol"].tolist()

    output_path = Path(output_file)
    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(merged_dict, f)
    elif output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_dict, f, indent=4)
    elif output_path.suffix == ".mat":
        for key, value in merged_dict.items():
            if value is None:
                merged_dict[key] = np.nan
        scipy.io.savemat(output_path, merged_dict)
    elif output_path.suffix == ".npz":
        np.savez(output_path, **merged_dict)
    elif output_path.suffix == ".csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, merged_dict.keys())
            w.writeheader()
            w.writerow(merged_dict)
    else:
        raise_error(f"Unsupported output file format: {output_path.suffix}")
