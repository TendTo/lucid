from pylucid import GaussianKernelRidgeRegression, Regression, GaussianKernel
import numpy as np


class TestRegression:
    class TestGaussianKernelRidgeRegression:

        def test_init(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = GaussianKernelRidgeRegression(
                kernel=k,
                training_inputs=np.array([[1, 2, 3], [4, 5, 6]]),
                training_outputs=np.array([[1, 2, 3], [5, 6, 1]]),
            )
            assert o is not None
            assert isinstance(o, Regression)
            assert o.kernel is not k
            assert o.training_inputs.shape == (2, 3)
            assert o.coefficients.shape == (2, 3)

        def test_call(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            o = GaussianKernelRidgeRegression(
                kernel=k,
                training_inputs=np.array([[1, 2, 3], [4, 5, 6]]),
                training_outputs=np.array([[1, 2, 3], [5, 6, 1]]),
            )
            assert np.allclose(o(np.array([[1, 2, 3]])), [1.0, 2.0, 3.0])
