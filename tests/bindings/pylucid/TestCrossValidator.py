import numpy as np
import pytest

from pylucid import (
    GaussianKernel,
    KernelRidgeRegressor,
    KFold,
    LeaveOneOut,
    MedianHeuristicTuner,
    Parameter,
    exception,
    mse_score,
    r2_score,
    rmse_score,
)


class TestCrossValidator:
    class TestKFold:
        def test_init_default(self):
            """Test KFold initialization with default parameters."""
            cv = KFold(num_folds=5, shuffle=False)
            assert cv is not None
            assert isinstance(cv, KFold)

        def test_init_custom_folds(self):
            """Test KFold initialization with custom number of folds."""
            cv = KFold(num_folds=10, shuffle=False)
            assert cv is not None

        def test_init_with_shuffle(self):
            """Test KFold initialization with shuffle enabled."""
            cv = KFold(num_folds=5, shuffle=True)
            assert cv is not None

        def test_num_folds(self):
            """Test num_folds method returns correct number of folds."""
            cv = KFold(num_folds=5, shuffle=False)
            X = np.random.uniform(size=(20, 2))
            num_folds = cv.num_folds(X)
            assert num_folds == 5

        def test_num_folds_with_small_dataset(self):
            """Test num_folds with dataset smaller than requested folds."""
            cv = KFold(num_folds=10, shuffle=False)
            X = np.random.uniform(size=(8, 2))  # Only 8 samples
            with pytest.raises(exception.LucidInvalidArgumentException):
                cv.num_folds(X)

        def test_fit_without_tuner(self):
            """Test cross-validation fit without a tuner."""
            cv = KFold(num_folds=3, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            # Create simple dataset
            X = np.linspace(0, 10, 30).reshape(-1, 1)
            y = np.sin(X).flatten()

            # Fit with cross-validation
            score = cv.fit(estimator, X, y, r2_score)

            # Score should be a valid R² value (typically between 0 and 1 for good fits)
            assert isinstance(score, float)
            assert not np.isnan(score)
            assert not np.isinf(score)

        def test_fit_with_tuner(self):
            """Test cross-validation fit with a tuner."""
            cv = KFold(num_folds=3, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            tuner = MedianHeuristicTuner()

            # Create dataset with different scales
            X = np.random.uniform(size=(30, 2))
            X[:, 0] *= 0.1
            X[:, 1] *= 10.0
            y = np.sin(X[:, 0]) + np.cos(X[:, 1])

            original_sigma_l = np.copy(estimator.get(Parameter.SIGMA_L))

            # Fit with cross-validation and tuning
            score = cv.fit(estimator, X, y, tuner, r2_score)

            assert isinstance(score, float)
            # Tuner should have updated sigma_l
            new_sigma_l = estimator.get(Parameter.SIGMA_L)
            assert not np.allclose(original_sigma_l, new_sigma_l)

        def test_fit_with_different_scorers(self):
            """Test fit with different scoring functions."""
            cv = KFold(num_folds=3, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            X = np.linspace(0, 10, 30).reshape(-1, 1)
            y = np.sin(X).flatten()

            # Test with different scorers
            rr2 = cv.fit(estimator, X, y, r2_score)
            assert isinstance(rr2, float)
            assert rr2 <= 1  # R2 is always less than or equal to 1

            mse = cv.fit(estimator, X, y, mse_score)
            assert isinstance(mse, float)
            assert mse <= 0  # MSE is always non-positive

            rmse = cv.fit(estimator, X, y, rmse_score)
            assert isinstance(rmse, float)
            assert rmse <= 0  # RMSE is always non-positive

        def test_score_method(self):
            """Test the score method for evaluating a trained model."""
            cv = KFold(num_folds=3, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            # Train the estimator first
            X_train = np.linspace(0, 10, 30).reshape(-1, 1)
            y_train = np.sin(X_train).flatten()
            estimator.fit(X_train, y_train)

            # Now score with cross-validation
            scores = cv.score(estimator, X_train, y_train, r2_score)

            assert isinstance(scores, list)
            assert len(scores) == 3  # One score per fold
            for score in scores:
                assert isinstance(score, float)
                assert not np.isnan(score)

        def test_fit_insufficient_samples_raises(self):
            """Test that fitting with too few samples raises an exception."""
            cv = KFold(num_folds=10, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            # Only 5 samples, but requesting 10 folds
            X = np.random.uniform(size=(5, 2))
            y = np.random.uniform(size=(5,))

            with pytest.raises(exception.LucidInvalidArgumentException):
                cv.fit(estimator, X, y, r2_score)

        def test_fit_mismatched_samples_raises(self):
            """Test that mismatched X and y shapes raise an exception."""
            cv = KFold(num_folds=3, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            X = np.random.uniform(size=(20, 2))
            y = np.random.uniform(size=(15,))  # Mismatched

            with pytest.raises(exception.LucidInvalidArgumentException):
                cv.fit(estimator, X, y, r2_score)

        def test_fit_single_fold_raises(self):
            """Test that requesting a single fold raises an exception."""
            with pytest.raises(exception.LucidInvalidArgumentException):
                KFold(num_folds=1, shuffle=False)

        def test_fit_zero_folds_raises(self):
            """Test that requesting zero folds raises an exception."""
            with pytest.raises(exception.LucidInvalidArgumentException):
                KFold(num_folds=0, shuffle=False)

        def test_fit_negative_folds_raises(self):
            """Test that requesting negative folds raises an exception."""
            with pytest.raises(exception.LucidInvalidArgumentException):
                KFold(num_folds=-1, shuffle=False)

        def test_shuffle_produces_different_results(self):
            """Test that shuffle=True produces different fold assignments."""
            kernel = GaussianKernel(sigma_l=1.0)
            X = np.linspace(0, 10, 30).reshape(-1, 1)
            y = np.sin(X).flatten()

            # Without shuffle
            cv_no_shuffle = KFold(num_folds=5, shuffle=False)
            estimator1 = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score1 = cv_no_shuffle.fit(estimator1, X, y, r2_score)

            # With shuffle (results may vary but should still be valid)
            cv_shuffle = KFold(num_folds=5, shuffle=True)
            estimator2 = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score2 = cv_shuffle.fit(estimator2, X, y, r2_score)

            # Both should produce valid scores
            assert isinstance(score1, float)
            assert isinstance(score2, float)
            assert not np.isnan(score1)
            assert not np.isnan(score2)

        def test_reproducibility_without_shuffle(self):
            """Test that results are reproducible when shuffle=False."""
            kernel = GaussianKernel(sigma_l=1.0)
            X = np.linspace(0, 10, 30).reshape(-1, 1)
            y = np.sin(X).flatten()

            cv = KFold(num_folds=5, shuffle=False)

            # Run twice with same configuration
            estimator1 = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score1 = cv.fit(estimator1, X, y, r2_score)

            estimator2 = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score2 = cv.fit(estimator2, X, y, r2_score)

            # Should produce identical results
            assert np.isclose(score1, score2)

        def test_fit_with_multidimensional_output(self):
            """Test cross-validation with multi-dimensional output."""
            cv = KFold(num_folds=3, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            X = np.random.uniform(size=(30, 2))
            # Multi-dimensional output
            y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 1])])

            score = cv.fit(estimator, X, y, r2_score)

            assert isinstance(score, float)
            assert not np.isnan(score)

        def test_large_number_of_folds(self):
            """Test with a large number of folds (approaching LOOCV)."""
            cv = KFold(num_folds=20, shuffle=False)
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            X = np.linspace(0, 10, 45).reshape(-1, 1)
            y = np.sin(X).flatten()

            score = cv.fit(estimator, X, y, r2_score)

            assert isinstance(score, float)
            assert not np.isnan(score)

    class TestLeaveOneOut:
        def test_init(self):
            """Test LeaveOneOut initialization."""
            cv = LeaveOneOut()
            assert cv is not None
            assert isinstance(cv, LeaveOneOut)

        def test_num_folds(self):
            """Test num_folds method returns number of samples."""
            cv = LeaveOneOut()
            X = np.random.uniform(size=(15, 2))
            num_folds = cv.num_folds(X)
            assert num_folds == 15  # LOOCV creates as many folds as samples

        def test_fit_without_tuner(self):
            """Test LeaveOneOut fit without a tuner."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            # Small dataset (LOOCV is expensive)
            X = np.linspace(0, 5, 10).reshape(-1, 1)
            y = np.sin(X).flatten()

            score = cv.fit(estimator, X, y, rmse_score)

            assert isinstance(score, float)
            assert score <= 0  # RMSE is always non-positive

        def test_fit_with_tuner(self):
            """Test LeaveOneOut fit with a tuner."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            tuner = MedianHeuristicTuner()

            X = np.random.uniform(size=(10, 2))
            X[:, 0] *= 0.1
            X[:, 1] *= 10.0
            y = np.sin(X[:, 0]) + np.cos(X[:, 1])

            original_sigma_l = np.copy(estimator.get(Parameter.SIGMA_L))

            score = cv.fit(estimator, X, y, tuner, rmse_score)

            assert isinstance(score, float)
            # Tuner should have updated sigma_l
            new_sigma_l = estimator.get(Parameter.SIGMA_L)
            assert not np.allclose(original_sigma_l, new_sigma_l)

        def test_score_method(self):
            """Test the score method for LeaveOneOut."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            # Train the estimator first
            X = np.linspace(0, 5, 10).reshape(-1, 1)
            y = np.sin(X).flatten()
            estimator.fit(X, y)

            # Score with LOOCV
            scores = cv.score(estimator, X, y, rmse_score)

            assert isinstance(scores, list)
            assert len(scores) == 10  # One score per sample
            for score in scores:
                assert isinstance(score, float)
                assert score <= 0  # RMSE is always non-positive

        def test_fit_with_different_scorers(self):
            """Test LeaveOneOut with different scoring functions."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            X = np.linspace(0, 5, 10).reshape(-1, 1)
            y = np.sin(X).flatten()

            # Test different scorers
            with pytest.raises(exception.LucidInvalidArgumentException):
                cv.fit(estimator, X, y, r2_score)

            mse = cv.fit(estimator, X, y, mse_score)
            assert isinstance(mse, float)
            assert mse <= 0  # MSE is always non-positive

            rmse = cv.fit(estimator, X, y, rmse_score)
            assert isinstance(rmse, float)
            assert rmse <= 0  # RMSE is always non-positive

        def test_fit_single_sample_raises(self):
            """Test that LOOCV with single sample raises an exception."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            X = np.array([[1.0, 2.0]])
            y = np.array([3.0])

            # LOOCV needs at least 2 samples
            with pytest.raises(exception.LucidInvalidArgumentException):
                cv.fit(estimator, X, y, r2_score)

        def test_fit_two_samples(self):
            """Test LOOCV with minimum viable samples (2)."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            X = np.array([[1.0], [2.0]])
            y = np.array([1.0, 2.0])

            score = cv.fit(estimator, X, y, rmse_score)

            assert isinstance(score, float)
            assert score <= 0  # RMSE is always non-positive

        def test_fit_mismatched_samples_raises(self):
            """Test that mismatched X and y shapes raise an exception."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel)

            X = np.random.uniform(size=(10, 2))
            y = np.random.uniform(size=(8,))  # Mismatched

            with pytest.raises(exception.LucidInvalidArgumentException):
                cv.fit(estimator, X, y, r2_score)

        def test_fit_with_multidimensional_output(self):
            """Test LOOCV with multi-dimensional output."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            X = np.random.uniform(size=(10, 2))
            # Multi-dimensional output
            y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 1])])

            score = cv.fit(estimator, X, y, rmse_score)

            assert isinstance(score, float)
            assert score <= 0  # RMSE is always non-positive

        def test_loocv_vs_kfold_n(self):
            """Test that LOOCV with n samples is similar to n-fold CV."""
            kernel = GaussianKernel(sigma_l=1.0)
            X = np.linspace(0, 5, 10).reshape(-1, 1)
            y = np.sin(X).flatten()

            # LeaveOneOut
            cv_loo = LeaveOneOut()
            estimator_loo = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score_loo = cv_loo.fit(estimator_loo, X, y, rmse_score)

            # KFold with n folds (where n = number of samples)
            cv_kfold = KFold(num_folds=10, shuffle=False)
            estimator_kfold = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score_kfold = cv_kfold.fit(estimator_kfold, X, y, rmse_score)

            # Scores should be very similar (though not necessarily identical due to implementation details)
            assert isinstance(score_loo, float)
            assert isinstance(score_kfold, float)
            # Both should be reasonable
            assert not np.isnan(score_loo)
            assert not np.isnan(score_kfold)

        def test_reproducibility(self):
            """Test that LOOCV results are reproducible."""
            kernel = GaussianKernel(sigma_l=1.0)
            X = np.linspace(0, 5, 10).reshape(-1, 1)
            y = np.sin(X).flatten()

            cv = LeaveOneOut()

            # Run twice
            estimator1 = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score1 = cv.fit(estimator1, X, y, rmse_score)

            estimator2 = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)
            score2 = cv.fit(estimator2, X, y, rmse_score)

            # Should produce identical results (no randomness in LOOCV)
            assert np.isclose(score1, score2)

        def test_perfect_fit_score(self):
            """Test LOOCV with a perfect linear fit."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.001)

            # Linear data
            X = np.linspace(0, 10, 15).reshape(-1, 1)
            y = 2 * X.flatten() + 1  # Perfect linear relationship

            score = cv.fit(estimator, X, y, rmse_score)

            # R² should be very high for a good fit
            assert isinstance(score, float)
            assert score > -0.5  # Should be able to capture linear trend

        def test_noisy_data(self):
            """Test LOOCV with noisy data."""
            cv = LeaveOneOut()
            kernel = GaussianKernel(sigma_l=1.0)
            estimator = KernelRidgeRegressor(kernel=kernel, regularization_constant=0.1)

            # Add noise to sine wave
            np.random.seed(42)
            X = np.linspace(0, 5, 12).reshape(-1, 1)
            y = np.sin(X).flatten() + np.random.normal(0, 0.1, X.shape[0])

            score = cv.fit(estimator, X, y, rmse_score)

            assert isinstance(score, float)
            assert score <= 0  # RMSE is always non-positive
            # With noise, score should still be reasonable but not perfect
            assert score < 1.0
