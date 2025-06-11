import numpy as np
import pytest
from pylucid import (
    Estimator,
    GaussianKernel,
    KernelRidgeRegressor,
    LucidInvalidArgumentException,
    MedianHeuristicTuner,
    Parameter,
)

try:
    from sklearn.kernel_ridge import KernelRidge
except ImportError:
    KernelRidge = None


class TestRegression:
    class TestGaussianKernelRidgeRegressor:

        def test_init(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = KernelRidgeRegressor(kernel=k, regularization_constant=11)
            assert o is not None
            assert isinstance(o, Estimator)
            assert o.kernel is not k
            assert o.training_inputs.size == 0
            assert o.coefficients.size == 0
            assert o.get(Parameter.REGULARIZATION_CONSTANT) == o.regularization_constant == 11
            assert np.allclose(o.get(Parameter.SIGMA_L), o.kernel.sigma_l)
            assert o.get(Parameter.SIGMA_F) == o.kernel.sigma_f == 2

        def test_has(self):
            o = KernelRidgeRegressor(kernel=GaussianKernel(1))
            assert o.has(Parameter.SIGMA_F) and Parameter.SIGMA_F in o
            assert o.has(Parameter.SIGMA_L) and Parameter.SIGMA_L in o
            assert o.has(Parameter.REGULARIZATION_CONSTANT) and Parameter.REGULARIZATION_CONSTANT in o

        def test_set(self):
            k = GaussianKernel(sigma_l=np.ones((4,)))
            o = KernelRidgeRegressor(kernel=k)
            o.consolidate(
                x=np.array([[1.0, 2.0, 3.0, 5.0], [4.0, 5.0, 6.0, 1.0]]), y=np.array([[1.0, 2.0], [5.0, 6.0]])
            )
            o.set(Parameter.SIGMA_L, np.array([1.0, 2.0, 3.0, 4.0]))
            assert np.allclose(o.get(Parameter.SIGMA_L), [1.0, 2.0, 3.0, 4.0])
            o.set(Parameter.SIGMA_F, 2.0)
            assert o.get(Parameter.SIGMA_F) == o.kernel.sigma_f == 2.0
            o.set(Parameter.REGULARIZATION_CONSTANT, 51.0)
            assert o.get(Parameter.REGULARIZATION_CONSTANT) == o.regularization_constant == 51.0

        def test_data(self):
            k = GaussianKernel(sigma_f=2.0, sigma_l=[3.0, 4.0, 5.0])
            o = KernelRidgeRegressor(kernel=k)
            o.consolidate(
                x=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), y=np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 1.0]])
            )

            assert isinstance(o.training_inputs, np.ndarray)
            assert o.training_inputs.flags.c_contiguous
            assert not o.training_inputs.flags.writeable
            assert not o.training_inputs.flags.owndata

            assert isinstance(o.coefficients, np.ndarray)
            assert o.coefficients.flags.c_contiguous
            assert not o.coefficients.flags.writeable
            assert not o.coefficients.flags.owndata

            assert isinstance(o.get(Parameter.SIGMA_L), np.ndarray)
            assert o.get(Parameter.SIGMA_L).flags.c_contiguous
            assert not o.get(Parameter.SIGMA_L).flags.writeable
            assert not o.get(Parameter.SIGMA_L).flags.owndata

            assert o.kernel is not k

        def test_call(self):
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_f=2.0, sigma_l=[3.0, 4.0, 5.0]))
            o.consolidate(
                x=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), y=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )
            assert np.allclose(o(np.array([[5.0, 6.0, 1.0]])), [1.48168935, 1.89938798, 2.31708661])

        def test_call_baseline(self):
            if KernelRidge is None:
                return

            sigma_l = 2
            reg_coeff = 1

            # Solving (K + λnI) * x = y
            # K = sigma_f^2 * exp(-||x_i - x_j||^2 / (2 * sigma_l^2))
            r = KernelRidgeRegressor(
                kernel=GaussianKernel(sigma_l=np.full(3, fill_value=sigma_l)),
                regularization_constant=reg_coeff,
            )

            # Solving (K + λI) * x = y
            # K = sigma_f^2 * exp(-γ||x_i - x_j||^2)
            kr = KernelRidge(kernel="rbf", gamma=0.5 * sigma_l**-2, alpha=reg_coeff * 2)

            training_inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            training_outputs = np.array([[1.0, 2.0], [5.0, 3.0]])

            r.fit(training_inputs, training_outputs)
            kr.fit(training_inputs, training_outputs)

            x = np.array([[8.0, 9.0, 10.0]])
            assert np.allclose(r.predict(x), kr.predict(x))

        def test_call_before_fit(self):
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5]))
            with pytest.raises(LucidInvalidArgumentException):
                o(x=np.array([[1.0, 2.0, 3.0]]))

        def test_clone(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = KernelRidgeRegressor(kernel=k)
            oc = o.clone()
            assert o is not oc
            assert o.kernel is not oc.kernel
            assert o.get(Parameter.SIGMA_F) == oc.get(Parameter.SIGMA_F)
            assert o.get(Parameter.REGULARIZATION_CONSTANT) == oc.get(Parameter.REGULARIZATION_CONSTANT)
            assert np.allclose(o.get(Parameter.SIGMA_L), oc.get(Parameter.SIGMA_L))
            assert np.allclose(o.training_inputs, oc.training_inputs)
            assert np.allclose(o.coefficients, oc.coefficients)
            assert o.regularization_constant == oc.regularization_constant

        def test_tuner(self):
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5]))
            assert o.tuner is None

            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5]), tuner=MedianHeuristicTuner())
            assert isinstance(o.tuner, MedianHeuristicTuner)

        def test_str(self):
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5]))
            assert str(o).startswith("KernelRidgeRegressor")
