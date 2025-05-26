from pylucid import KernelRidgeRegressor, Estimator, GaussianKernel, Parameter, LucidInvalidArgumentException
import numpy as np
import pytest


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
            k = GaussianKernel(sigma_l=np.zeros((4,)))
            o = KernelRidgeRegressor(kernel=k)
            o.consolidate(x=np.array([[1, 2, 3, 5], [4, 5, 6, 1]]), y=np.array([[1, 2], [5, 6]]))
            o.set(Parameter.SIGMA_L, np.array([1, 2, 3, 4]))
            assert np.allclose(o.get(Parameter.SIGMA_L), [1, 2, 3, 4])
            o.set(Parameter.SIGMA_F, 2.0)
            assert o.get(Parameter.SIGMA_F) == o.kernel.sigma_f == 2.0
            o.set(Parameter.REGULARIZATION_CONSTANT, 51.0)
            assert o.get(Parameter.REGULARIZATION_CONSTANT) == o.regularization_constant == 51.0

        def test_data(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = KernelRidgeRegressor(kernel=k)
            o.consolidate(x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[1, 2, 3], [5, 6, 1]]))
            assert o.training_inputs.flags.c_contiguous
            assert not o.training_inputs.flags.writeable
            assert not o.training_inputs.flags.owndata
            assert o.coefficients.flags.c_contiguous
            assert not o.coefficients.flags.writeable
            assert not o.coefficients.flags.owndata
            assert o.get(Parameter.SIGMA_L).flags.c_contiguous
            assert not o.get(Parameter.SIGMA_L).flags.writeable
            assert not o.get(Parameter.SIGMA_L).flags.owndata
            assert o.kernel is not k

        def test_call(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = KernelRidgeRegressor(kernel=k)
            o.consolidate(x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[1, 2, 3], [5, 6, 1]]))
            assert np.allclose(o(x=np.array([[1, 2, 3]])), [1.0, 2.0, 3.0])

        def test_call_before_fit(self):
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5]))
            with pytest.raises(LucidInvalidArgumentException):
                o(x=np.array([[1, 2, 3]]))

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
