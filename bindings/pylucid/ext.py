from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import Estimator

if TYPE_CHECKING:
    from typing import Callable


class ModelEstimator(Estimator):
    """Estimator for the system dynamics.

    It can be used when you have a model of the system dynamics that you want to use directly instead of learning it from data.
    Useful to debug the pipeline, as the predictions will be exactly the same as the model function.

    Args:
        f: A callable that takes a numpy array as input and returns a numpy array as output.
    """

    def __init__(self, f: "Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]"):
        super().__init__()
        self._f = f

    def predict(self, x: "np.typing.NDArray[np.float64]") -> "np.typing.NDArray[np.float64]":
        """Predict the next state given the current state by applying the model function."""
        return self._f(x)

    def consolidate(
        self,
        training_inputs: "np.typing.NDArray[np.float64]",
        training_outputs: "np.typing.NDArray[np.float64]",
        requests: "int",
    ) -> "ModelEstimator":
        """Consolidate the model with the training data.

        Since we are using the model directly, we do not need to change anything.
        """
        return self

    def score(
        self, evaluation_inputs: "np.typing.NDArray[np.float64]", evaluation_outputs: "np.typing.NDArray[np.float64]"
    ) -> float:
        """Score the model based on the evaluation data.

        Since we are using the model directly, we can return a fixed score.
        """
        return 1.0

    def clone(self) -> "ModelEstimator":
        """Clone the estimator."""
        return ModelEstimator(self._f)

    def __str__(self) -> str:
        return f"ModelEstimator( f( {self._f.__name__} ) )"
