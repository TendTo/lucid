/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * KernelRidgeRegression class.
 */
#pragma once

#include "lucid/math/Kernel.h"
#include "lucid/math/Regression.h"
#include "lucid/util/concept.h"

namespace lucid {

// Forward declaration
class GaussianKernel;

/**
 * Ridge regression with a kernel function.
 * This is a linear regression with @f$ L_2 @f$ regularization.
 * Given two vector spaces @f$ \mathcal{X}, \mathcal{Y} @f$
 * and the dataset @f$ \{ (x_i, y_i) \}_{i=1}^n @f$, where @f$ x_i \in \mathcal{X} @f$ and @f$ y_i \in \mathcal{Y} @f$,
 * the goal is to find the function @f$ f^*: \mathcal{X} \to \mathcal{Y} @f$ such that the sum of the squared errors is
 * minimized, i.e.
 * @f[
 *   f^* = \arg\min_{f \in \mathcal{H}} \left\{ \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda
 * \|f\|_{\mathcal{H}}^2 \right\}
 * @f]
 * where @f$ \mathcal{H} @f$ is a reproducing kernel Hilbert space (RKHS) with kernel @f$ k: \mathcal{X} \times
 * \mathcal{X} \to \mathbb{R} @f$.
 * Due to the reproducing property of the RKHS, @f$ f^* @f$ can be expressed as the linear combination
 * @f[
 *  f^*(x) = \sum_{i=1}^n w_i k(x_i, x)
 * @f]
 * where @f$ w \in \mathbb{R}^n @f$ are some coefficients we want to find.
 * We can rewrite the loss function in vector form as
 * @f[
 * \frac{1}{n} \|y - Kw\|^2 + \lambda w^T K w
 * @f]
 * where @f$ K @f$ is the Gram matrix with elements @f$ K_{ij} = k(x_i, x_j) @f$.
 * Looking for the minimum, the partial gradient of the loss function with respect to @f$ w @f$ is set to zero
 * @f[
 * \frac{\partial}{\partial w} \left\{ \frac{1}{n} \|y - Kw\|^2 + \lambda w^T K w \right\} = 0
 * @f]
 * leads us to the closed-form solution:
 * @f[
 * w = (K + \lambda I)^{-1} y .
 * @f]
 */
class KernelRidgeRegression final : public Regression {
 public:
  /**
   * @brief Construct a new Kernel Ridge Regression object with the given parameters.
   * Creating an instance of this regressor will immediately initialise the model to be used for prediction.
   * @tparam K type of kernel function
   * @param kernel kernel function
   * @param training_inputs input data used for training
   * @param training_outputs output data used for training
   * @param regularization_constant regularization constant
   */
  template <IsAnyOf<GaussianKernel> K>
  KernelRidgeRegression(K kernel, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                        Scalar regularization_constant);
};

}  // namespace lucid
