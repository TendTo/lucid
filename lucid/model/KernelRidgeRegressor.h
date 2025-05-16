/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * KernelRidgeRegressor class.
 */
#pragma once

#include <memory>

#include "lucid/model/Estimator.h"
#include "lucid/model/FeatureMap.h"
#include "lucid/model/Kernel.h"

namespace lucid {

/**
 * Ridge regressor with a kernel function.
 * This is a linear regressor with @f$ L_2 @f$ regularization.
 * Given two vector spaces @f$ \mathcal{X}, \mathcal{Y} @f$ and the training datasets @f$ \{ (x_i, y_i) \}_{i=1}^n @f$,
 * where @f$ x_i \in \mathcal{X} @f$ and @f$ y_i \in \mathcal{Y} @f$,
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
 * w = (K + \lambda n I)^{-1} y .
 * @f]
 * Finally, when we want to predict the output for a new input @f$ x @f$, we can just compute
 * @f[
 * f^*(x) = k(x, x_1) w_1 + k(x, x_2) w_2 + \cdots + k(x, x_n) w_n
 * @f]
 * of, in matrix form
 * @f[
 * f^*(x) = K(x, x_\text{traning}) w
 * @f]
 * where @f$ K(x,  x_\text{traning}) @f$ is the vector of kernel evaluations between @f$ x @f$
 * and the training inputs @f$ x_\text{traning} @f$.
 */
class KernelRidgeRegressor final : public Estimator {
 public:
  using Estimator::operator();
  using Estimator::fit;
  /**
   * Construct a new Kernel Ridge Regressor object with the given parameters.
   * @param kernel kernel function used to compute the Gram matrix
   * @param regularization_constant regularization constant. Avoids overfitting by penalizing large coefficients
   */
  explicit KernelRidgeRegressor(const Kernel& kernel, Scalar regularization_constant = 0);
  /**
   * Construct a new Kernel Ridge Regressor object with the given parameters.
   * @param kernel kernel function used to compute the Gram matrix
   * @param regularization_constant regularization constant. Avoids overfitting by penalizing large coefficients
   */
  explicit KernelRidgeRegressor(std::unique_ptr<Kernel>&& kernel, Scalar regularization_constant = 0);

  [[nodiscard]] Matrix predict(ConstMatrixRef x) const override;
  /**
   * A model is a function that takes a @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X}
   * @f$ and returns a @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$.
   * The `feature_map` is used to approximate the kernel vector @f$ k(x, x_i) @f$.
   * @warning Using this method introduces an approximation error.
   * We suggest using the @ref operator(ConstMatrixRef) method instead.
   * @param x @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
   * @param feature_map feature map used to approximate the kernel vector @f$ k(x, x_i) @f$
   * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
   */
  [[nodiscard]] Matrix operator()(ConstMatrixRef x, const FeatureMap& feature_map) const;
  /**
   * A model is a function that takes a @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X}
   * @f$ and returns a @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$.
   * The `feature_map` is used to approximate the kernel vector @f$ k(x, x_i) @f$.
   * @warning Using this method introduces an approximation error.
   * We suggest using the @ref operator(ConstMatrixRef) method instead.
   * @param x @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
   * @param feature_map feature map used to approximate the kernel vector @f$ k(x, x_i) @f$
   * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
   */
  [[nodiscard]] Matrix predict(ConstMatrixRef x, const FeatureMap& feature_map) const;

  /** @getter{kernel, regressor} */
  [[nodiscard]] const std::unique_ptr<Kernel>& kernel() const { return kernel_; }
  /** @getter{training inputs, regressor} */
  [[nodiscard]] const Matrix& training_inputs() const { return training_inputs_; }
  /** @getter{coefficients, regressor} */
  [[nodiscard]] const Matrix& coefficients() const { return coefficients_; }
  /** @getter{regularization constant, regressor} */
  [[nodiscard]] double regularization_constant() const { return regularization_constant_; }

  void set(Parameter parameter, int value) override;
  void set(Parameter parameter, double value) override;
  void set(Parameter parameter, const Vector& value) override;

  Estimator& consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) override;

  [[nodiscard]] double score(ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) const override;

  [[nodiscard]] std::unique_ptr<Estimator> clone() const override;

 private:
  [[nodiscard]] int get_i(Parameter parameter) const override;
  [[nodiscard]] double get_d(Parameter parameter) const override;
  [[nodiscard]] const Vector& get_v(Parameter parameter) const override;

  std::unique_ptr<Kernel> kernel_;  ///< Kernel function
  double regularization_constant_;  ///< Regularization constant
  Matrix training_inputs_;          ///< Training inputs
  Matrix coefficients_;             ///< Coefficients of the linear combination describing the regression model
};

}  // namespace lucid
