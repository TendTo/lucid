import pytest
import numpy as np
from pylucid import (
    MedianHeuristicTuner,
    GridSearchTuner,
    ParameterValues,
    ParameterValue,
    GaussianKernel,
    KernelRidgeRegressor,
    Parameter,
)
from itertools import product


class TestTuner:
    class TestMedianHeuristicTuner:
        def test_init(self):
            tuner = MedianHeuristicTuner()
            assert isinstance(tuner, MedianHeuristicTuner)

        def test_with_estimator(self):
            kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, tuner=MedianHeuristicTuner())
            assert estimator.tuner is not None

        def test_basic_tuning(self):
            # Create an estimator with fixed initial sigma_l values
            kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            # Create data where dimensions have very different scales
            X = np.random.uniform(size=(20, 2))
            X[:, 0] *= 0.1  # First dimension has small scale
            X[:, 1] *= 10.0  # Second dimension has large scale
            y = np.sin(X[:, 0]) + np.cos(X[:, 1])

            # Save original sigma_l
            original_sigma_l = np.copy(estimator.get(Parameter.SIGMA_L))

            # Apply tuning
            tuner = MedianHeuristicTuner()
            tuner.tune(estimator, X, y)

            # Check that sigma_l has been updated and reflects the different scales
            new_sigma_l = estimator.get(Parameter.SIGMA_L)
            assert not np.allclose(original_sigma_l, new_sigma_l)
            assert new_sigma_l[1] > new_sigma_l[0]  # Second dimension should have larger sigma_l

        def test_different_scales(self):
            kernel = GaussianKernel(dimension=3, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            # Create data with different scales for each dimension
            X = np.random.uniform(size=(30, 3))
            X[:, 0] *= 0.1  # Small scale
            X[:, 1] *= 1.0  # Medium scale
            X[:, 2] *= 10.0  # Large scale
            y = np.sum(np.sin(X), axis=1)

            # Apply tuning
            tuner = MedianHeuristicTuner()
            tuner.tune(estimator, X, y)

            # Check that sigma_l reflects the different scales
            sigma_l = estimator.get(Parameter.SIGMA_L)
            assert sigma_l[0] < sigma_l[1] < sigma_l[2]

        def test_uniform_data(self):
            kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            # Create uniform grid data
            x = np.linspace(0, 1, 5)
            X = np.array([(i, j) for i in x for j in x])
            y = np.sin(X[:, 0] * X[:, 1])

            # Apply tuning
            tuner = MedianHeuristicTuner()
            tuner.tune(estimator, X, y)

            # For uniform grid, sigma_l should be similar for both dimensions
            sigma_l = estimator.get(Parameter.SIGMA_L)
            assert np.allclose(sigma_l[0], sigma_l[1], rtol=0.1)

        def test_single_sample_raises_exception(self):
            kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            # Single sample
            X = np.array([[1.0, 2.0]])
            y = np.array([3.0])

            # Should raise exception
            tuner = MedianHeuristicTuner()
            with pytest.raises(Exception):
                tuner.tune(estimator, X, y)

        def test_mismatched_inputs_outputs(self):
            kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            X = np.random.uniform(size=(10, 2))
            y = np.random.uniform(size=(5, 1))  # Mismatched sample count

            tuner = MedianHeuristicTuner()
            with pytest.raises(Exception):
                tuner.tune(estimator, X, y)

    class TestGridSearchTuner:
        def test_init(self):
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([0.1, 0.1]), np.array([1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [0.1, 1.0]),
            ]

            tuner = GridSearchTuner(params)
            assert isinstance(tuner, GridSearchTuner)
            assert len(tuner.parameters) == 2

            tuner = GridSearchTuner(params, n_jobs=2)
            assert tuner.n_jobs == 2

        def test_init_args(self):
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([0.1, 0.1]), np.array([1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [0.1, 1.0]),
            ]

            base_tuner = GridSearchTuner(params)

            tuner = GridSearchTuner(*params)
            assert isinstance(tuner, GridSearchTuner)
            assert len(tuner.parameters) == 2
            assert tuner.parameters == base_tuner.parameters

            tuner = GridSearchTuner(*params, n_jobs=2)
            assert tuner.n_jobs == 2
            assert tuner.parameters == base_tuner.parameters

        def test_init_dict(self):
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([0.1, 0.1]), np.array([1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [0.1, 1.0]),
            ]

            base_tuner = GridSearchTuner(params)

            params_dict = {p.parameter: p.values for p in params}

            tuner = GridSearchTuner(params_dict)
            assert isinstance(tuner, GridSearchTuner)
            assert len(tuner.parameters) == 2
            assert tuner.parameters == base_tuner.parameters

            tuner = GridSearchTuner(params_dict, n_jobs=2)
            assert tuner.n_jobs == 2
            assert tuner.parameters == base_tuner.parameters

        def test_parameter_values_access(self):
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([0.1, 0.1]), np.array([1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [0.1, 1.0]),
            ]

            tuner = GridSearchTuner(params)

            # Check parameters are accessible
            assert tuner.parameters[0].parameter == Parameter.SIGMA_L
            assert tuner.parameters[1].parameter == Parameter.SIGMA_F
            assert tuner.parameters[0].size == 2
            assert tuner.parameters[1].size == 2

        def test_basic_tuning(self):
            X = np.random.uniform(size=(20, 2))

            # Define parameter grid including the target values
            sigma_l_values = [
                np.array([0.1, 0.1]),  # Too small for both
                np.array([0.5, 2.0]),  # Target values
                np.array([5.0, 5.0]),  # Too large for both
            ]
            sigma_f_values = [0.1, 1.0, 10.0]  # Target is 1.0

            X = np.random.uniform(size=(20, 2))
            y = np.sin(X[:, 0] * np.pi) + np.cos(X[:, 1] * np.pi)

            best_score = -np.inf
            best_sigma_l = None
            best_sigma_f = None
            for sigma_l, sigma_f in product(sigma_l_values, sigma_f_values):
                estimator = KernelRidgeRegressor(kernel=GaussianKernel(sigma_l=sigma_l, sigma_f=sigma_f))
                estimator.consolidate(X, y)
                score = estimator.score(X, y)

                if score > best_score:
                    best_score = score
                    best_sigma_l = sigma_l
                    best_sigma_f = sigma_f

            params = [
                ParameterValues(Parameter.SIGMA_L, sigma_l_values),
                ParameterValues(Parameter.SIGMA_F, sigma_f_values),
            ]

            estimator = KernelRidgeRegressor(kernel=GaussianKernel(dimension=2), tuner=GridSearchTuner(params))

            estimator.fit(X, y)

            assert np.allclose(estimator.get(Parameter.SIGMA_L), best_sigma_l)
            assert estimator.get(Parameter.SIGMA_F) == best_sigma_f
            assert estimator.score(X, y) == best_score

        def test_multiprocessing(self):
            X = np.random.uniform(size=(20, 2))
            y = np.sin(X[:, 0] * np.pi) + np.cos(X[:, 1] * np.pi)

            # Define a simple grid
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([0.1, 0.1]), np.array([1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [0.1, 1.0]),
            ]

            # Test with different job counts
            for n_jobs in [1, 2, 4]:
                tuner = GridSearchTuner(params, n_jobs=n_jobs)
                kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
                estimator = KernelRidgeRegressor(kernel=kernel)

                # Should work without errors
                tuner.tune(estimator, X, y)

                # Results might vary, but at least we check it runs
                assert tuner.n_jobs == n_jobs
                assert estimator.get(Parameter.SIGMA_L) is not None
                assert estimator.get(Parameter.SIGMA_F) is not None

        def test_regularization_parameter(self):
            X = np.random.uniform(size=(20, 2))
            y = np.sin(X[:, 0] * np.pi) + np.cos(X[:, 1] * np.pi)

            # Define grid including regularization parameter
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [1.0]),
                ParameterValues(Parameter.REGULARIZATION_CONSTANT, [0.001, 0.01, 0.1, 1.0]),
            ]

            tuner = GridSearchTuner(params)
            kernel = GaussianKernel(dimension=2, sigma_l=1.0, sigma_f=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=1.0)

            tuner.tune(estimator, X, y)

            # It should have selected one of our test values
            assert estimator.get(Parameter.REGULARIZATION_CONSTANT) in [0.001, 0.01, 0.1, 1.0]
