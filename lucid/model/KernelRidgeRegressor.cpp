/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/KernelRidgeRegressor.h"

#include <memory>
#include <numbers>  // NOLINT(build/include_order) fake positive C header
#include <numeric>
#include <utility>
#include <vector>

#include "lucid/model/GaussianKernel.h"
#include "lucid/model/GramMatrix.h"
#include "lucid/model/Scorer.h"
#include "lucid/model/Tuner.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

KernelRidgeRegressor::KernelRidgeRegressor(const Kernel& kernel, const Scalar regularization_constant,
                                           const std::shared_ptr<const Tuner>& tuner)
    : KernelRidgeRegressor{kernel.clone(), regularization_constant, tuner} {}
KernelRidgeRegressor::KernelRidgeRegressor(std::unique_ptr<Kernel>&& kernel, const Scalar regularization_constant,
                                           const std::shared_ptr<const Tuner>& tuner)
    : GradientOptimizable{kernel->parameters() | Parameter::REGULARIZATION_CONSTANT, tuner},
      kernel_{std::move(kernel)},
      regularization_constant_{regularization_constant},
      training_inputs_{},
      coefficients_{} {
  LUCID_TRACE_FMT("({}, {}, {})", *kernel_, regularization_constant, tuner.get() != nullptr);
  LUCID_CHECK_ARGUMENT_CMP(regularization_constant, >=, 0.0);
  LUCID_CHECK_ARGUMENT_EXPECTED(kernel_ != nullptr, "kernel", nullptr, "not nullptr");
  LUCID_CHECK_ARGUMENT(!kernel_->has(Parameter::REGULARIZATION_CONSTANT), "kernel",
                       "parameter 'regularization constant' is hidden by the regressor");
}

Matrix KernelRidgeRegressor::predict(ConstMatrixRef x) const {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(x));
  LUCID_CHECK_ARGUMENT(training_inputs_.size() > 0, "training_inputs", "the model is not fitted yet");
  LUCID_CHECK_ARGUMENT_EQ(x.cols(), training_inputs_.cols());
  return (*kernel_)(x, training_inputs_) * coefficients_;
}

Matrix KernelRidgeRegressor::operator()(ConstMatrixRef x, const FeatureMap& feature_map) const {
  return predict(x, feature_map);
}
Matrix KernelRidgeRegressor::predict(ConstMatrixRef x, const FeatureMap& feature_map) const {
  LUCID_TRACE_FMT("({}, {})", LUCID_FORMAT_MATRIX(x), feature_map);
  LUCID_WARN("Experts only. We do not know what will happen to your interpolation. And you may die. Sorry about that.");
  LUCID_CHECK_ARGUMENT(training_inputs_.size() > 0, "training_inputs", "the model is not fitted yet");
  LUCID_CHECK_ARGUMENT_EQ(x.cols(), training_inputs_.cols());
  Matrix kernel_input{Matrix::NullaryExpr(
      x.rows(), training_inputs_.rows(), [this, &x, &feature_map](const Index row, const Index col) {
        return (feature_map(x.row(row)) * feature_map(training_inputs_.row(col)).transpose()).value();
      })};
  LUCID_TRACE_FMT("kernel_input = {}", LUCID_FORMAT_MATRIX(kernel_input));
  return kernel_input * coefficients_;
}

/**
 * Compute the log marginal likelihood of the model given the Gram matrix K and the training outputs y.
 * The log marginal likelihood is given by:
 * @f[
 * \log P(y|x, \theta) = -\frac{1}{2}\ln|K| - \frac{1}{2}y^tK^{-1}y - \frac{N}{2}\ln{2\pi} .
 * @f]
 * where @f$ N @f$ is the number of training samples,
 * @f$ K @f$ is the Gram matrix,
 * @theta are the log hyperparameters of the kernel,
 * @f$ y @f$ are the training outputs.
 * @param K Gram matrix
 * @param y training outputs
 * @return log marginal likelihood
 */
double compute_log_marginal(const GramMatrix& K, ConstMatrixRef y) {
  LUCID_TRACE_FMT("({}, {})", LUCID_FORMAT_MATRIX(K), LUCID_FORMAT_MATRIX(y));
  LUCID_CHECK_ARGUMENT_EQ(K.rows(), K.cols());
  LUCID_CHECK_ARGUMENT_EQ(K.rows(), y.rows());
  // Compute the log marginal likelihood
  // log p(y | K, λ) = -0.5 . (y^T * w) - sum(log(diag(L))) - log(2*pi) * n / 2
  static const double log2pi = std::log(2 * std::numbers::pi);
  const Matrix w{K.inverse() * y};
  return (-0.5 * (y.array() * w.array()).matrix().colwise().sum().array()  //  -0.5 . (y^T * w)
          - K.L().diagonal().array().log().sum()                           //  -sum(log(diag(L)))
          - log2pi * static_cast<double>(y.rows()) / 2)                    // -log(2*pi) * n / 2
      .sum();
}

/**
 * Compute the gradient of the log marginal likelihood of the model given the Gram matrix K,
 * the training outputs y, the coefficients alpha and the gradient of the Gram matrix with respect to
 * the @ref GRADIENT_OPTIMIZABLE hyperparameters.
 * The gradient of the log marginal likelihood is given by:
 * @f[
 * \frac{\partial}{\partial\theta_i} \log P(y|x, \theta) =
 * \frac{1}{2}y^TK^{-1}\frac{\partial K}{\partial\theta_i}K^{-1}y^T
 * -\frac{1}{2}\mathrm{tr}\left(K^{-1}\frac{\partial K}{\partial\theta_i}\right) ,
 * @f]
 * where @f$ N @f$ is the number of training samples,
 * @f$ K @f$ is the Gram matrix,
 * @theta are the log hyperparameters of the kernel,
 * @f$ y @f$ are the training outputs.
 * For efficiency reasons, we compute the coefficients @f$ \alpha = K^{-1}y @f$ beforehand
 * and use them to rewrite the gradient as:
 * @f[
 * \frac{1}{2}\mathrm{tr}\left((\alpha\alpha^T - K^{-1})\frac{\partial K}{\partial\theta_i}\right) .
 * @f]
 * @note The gradient is computed with respect to the log of the actual hyperparameters.
 * @param K Gram matrix
 * @param alpha coefficients such that @f$ \alpha = K^{-1}y @f$
 * @param gradient gradient of the Gram matrix with respect to the @ref GRADIENT_OPTIMIZABLE hyperparameters
 * @return log marginal likelihood function gradient with respect to the @ref GRADIENT_OPTIMIZABLE hyperparameters
 */
Vector compute_log_marginal_gradient(const GramMatrix& K, const Matrix& alpha, const std::vector<Matrix>& gradient) {
  LUCID_TRACE_FMT("({}, {}, gradient)", LUCID_FORMAT_MATRIX(K), LUCID_FORMAT_MATRIX(alpha));
  LUCID_CHECK_ARGUMENT_EQ(K.rows(), K.cols());
  LUCID_CHECK_ARGUMENT_EQ(K.rows(), alpha.rows());
  // Compute the log marginal likelihood gradient
  const Matrix k_inv{K.inverse()};

  std::vector<Matrix> inner_term{};
  inner_term.reserve(alpha.cols());
  for (Index i = 0; i < alpha.cols(); ++i) {
    inner_term.emplace_back(alpha.col(i) * alpha.col(i).transpose() - k_inv);
  }

  Matrix log_likelihood_gradient_dims{Matrix::NullaryExpr(
      gradient.size(), inner_term.size(),
      [&](const Index row, const Index col) { return gradient[row].cwiseProduct(inner_term[col]).sum() * 0.5; })};
  Vector log_likelihood_gradient{log_likelihood_gradient_dims.rowwise().sum()};

  return log_likelihood_gradient;
}
Estimator& KernelRidgeRegressor::consolidate_impl(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                                                  const Requests requests) {
  LUCID_TRACE_FMT("({}, {}, {})", LUCID_FORMAT_MATRIX(training_inputs), LUCID_FORMAT_MATRIX(training_outputs),
                  requests);

  LUCID_CHECK_ARGUMENT_EQ(training_inputs.rows(), training_outputs.rows());
  training_inputs_ = training_inputs;
  std::vector<Matrix> kernel_gradient;
  // Compute gram matrix K (nxn) with elements K_{ij} = k(x_i, x_j)
  GramMatrix gram_matrix{*kernel_, training_inputs_, requests && Request::GRADIENT ? &kernel_gradient : nullptr};
  // Add the regularisation term to the diagonal K + λnI
  gram_matrix.add_diagonal_term(regularization_constant_ * static_cast<double>(training_inputs_.rows()));
  // Invert the gram matrix and compute the coefficients as (K + λnI)^-1 y
  coefficients_ = gram_matrix.inverse() * training_outputs;
  LUCID_TRACE_FMT("coefficients = {}", LUCID_FORMAT_MATRIX(coefficients_));
  if (requests && Request::OBJECTIVE_VALUE) {
    // Compute and update the log marginal likelihood
    objective_value_ = compute_log_marginal(gram_matrix, training_outputs);
    LUCID_TRACE_FMT("log_marginal_likelihood = [{}]", objective_value_);
  }
  if (requests && Request::GRADIENT) {
    // and its gradient
    LUCID_TRACE_FMT("kernel_gradient = [{}]", kernel_gradient);
    gradient_ = compute_log_marginal_gradient(gram_matrix, coefficients_, kernel_gradient);
    LUCID_TRACE_FMT("gradient.size() = {}", gradient_.size());
  }
  return *this;
}

double KernelRidgeRegressor::score(ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) const {
  LUCID_TRACE_FMT("({}, {})", LUCID_FORMAT_MATRIX(evaluation_inputs), LUCID_FORMAT_MATRIX(evaluation_outputs));
  return scorer::r2_score(*this, evaluation_inputs, evaluation_outputs);
}

std::unique_ptr<Estimator> KernelRidgeRegressor::clone() const {
  LUCID_TRACE("Cloning");
  std::unique_ptr<Estimator> out{std::make_unique<KernelRidgeRegressor>(kernel_->clone(), regularization_constant_)};
  static_cast<KernelRidgeRegressor*>(out.get())->training_inputs_ = training_inputs_;
  static_cast<KernelRidgeRegressor*>(out.get())->coefficients_ = coefficients_;
  return out;
}

void KernelRidgeRegressor::set(const Parameter parameter, int value) { kernel_->set(parameter, value); }
void KernelRidgeRegressor::set(const Parameter parameter, double value) {
  switch (parameter) {
    case Parameter::REGULARIZATION_CONSTANT:
      regularization_constant_ = value;
      break;
    default:
      kernel_->set(parameter, value);
  }
}
void KernelRidgeRegressor::set(const Parameter parameter, const Vector& value) { kernel_->set(parameter, value); }
int KernelRidgeRegressor::get_i(const Parameter parameter) const { return kernel_->get<int>(parameter); }
double KernelRidgeRegressor::get_d(const Parameter parameter) const {
  switch (parameter) {
    case Parameter::REGULARIZATION_CONSTANT:
      return regularization_constant_;
    default:
      return kernel_->get<double>(parameter);
  }
}
const Vector& KernelRidgeRegressor::get_v(const Parameter parameter) const {
  return kernel_->get<const Vector&>(parameter);
}
std::ostream& operator<<(std::ostream& os, const KernelRidgeRegressor& regressor) {
  return os << "KernelRidgeRegressor( "
            << "kernel( " << *regressor.kernel() << " ) "
            << "regularization_constant( " << regressor.regularization_constant() << " ) )";
}

}  // namespace lucid
