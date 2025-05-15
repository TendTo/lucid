from pylucid import KernelRidgeRegression, Estimator, GaussianKernel, Parameter
import numpy as np


class TestRegression:
    class TestGaussianKernelRidgeRegression:

        def test_init(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = KernelRidgeRegression(
                kernel=k,
                training_inputs=np.array([[1, 2, 3], [4, 5, 6]]),
                training_outputs=np.array([[1, 2, 3], [5, 6, 1]]),
            )
            assert o is not None
            assert isinstance(o, Estimator)
            assert o.kernel is not k
            assert o.training_inputs.shape == (2, 3)
            assert o.coefficients.shape == (2, 3)
            assert o.get(Parameter.REGULARIZATION_CONSTANT) == o.regularization_constant == 1e-6
            assert np.allclose(o.get(Parameter.SIGMA_L), o.kernel.sigma_l)
            assert o.get(Parameter.SIGMA_F) == o.kernel.sigma_f == 2

        def test_setter(self):
            k = GaussianKernel(sigma_l=np.zeros((4,)))
            o = KernelRidgeRegression(
                kernel=k,
                training_inputs=np.array([[1, 2, 3], [4, 5, 6]]),
                training_outputs=np.array([[1, 2, 3], [5, 6, 1]]),
            )
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

        def test_data(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = KernelRidgeRegression(
                kernel=k,
                training_inputs=np.array([[1, 2, 3], [4, 5, 6]]),
                training_outputs=np.array([[1, 2, 3], [5, 6, 1]]),
            )
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
            o = KernelRidgeRegression(
                kernel=k,
                training_inputs=np.array([[1, 2, 3], [4, 5, 6]]),
                training_outputs=np.array([[1, 2, 3], [5, 6, 1]]),
            )
            assert np.allclose(o(np.array([[1, 2, 3]])), [1.0, 2.0, 3.0])
