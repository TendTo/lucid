/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/KernelRidgeRegressor.h"

#include <memory>
#include <numbers>
#include <numeric>
#include <utility>

#include "Scorer.h"
#include "lucid/model/GaussianKernel.h"
#include "lucid/model/GramMatrix.h"
#include "lucid/model/Tuner.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

KernelRidgeRegressor::KernelRidgeRegressor(const Kernel& kernel, const Scalar regularization_constant,
                                           const std::shared_ptr<const Tuner>& tuner)
    : KernelRidgeRegressor{kernel.clone(), regularization_constant, tuner} {}
KernelRidgeRegressor::KernelRidgeRegressor(std::unique_ptr<Kernel>&& kernel, const Scalar regularization_constant,
                                           const std::shared_ptr<const Tuner>& tuner)
    : Estimator{kernel->parameters() | Parameter::REGULARIZATION_CONSTANT, tuner},
      kernel_{std::move(kernel)},
      regularization_constant_{regularization_constant},
      training_inputs_{},
      coefficients_{},
      log_marginal_likelihood_{-std::numeric_limits<double>::infinity()} {
  LUCID_CHECK_ARGUMENT_EXPECTED(regularization_constant >= 0.0, "regularization_constant", regularization_constant,
                                ">= 0.0");
  LUCID_CHECK_ARGUMENT_EXPECTED(kernel_ != nullptr, "kernel", nullptr, "not nullptr");
  LUCID_CHECK_ARGUMENT(!kernel_->has(Parameter::REGULARIZATION_CONSTANT), "kernel",
                       "parameter 'regularization constant' is hidden by the regressor");
}

Matrix KernelRidgeRegressor::predict(ConstMatrixRef x) const {
  LUCID_CHECK_ARGUMENT(training_inputs_.size() > 0, "training_inputs", "the model is not fitted yet");
  LUCID_CHECK_ARGUMENT_EXPECTED(x.cols() == training_inputs_.cols(), "input.cols()", x.cols(), training_inputs_.cols());
  return (*kernel_)(x, training_inputs_) * coefficients_;
}

Matrix KernelRidgeRegressor::operator()(ConstMatrixRef x, const FeatureMap& feature_map) const {
  return predict(x, feature_map);
}
Matrix KernelRidgeRegressor::predict(ConstMatrixRef x, const FeatureMap& feature_map) const {
  LUCID_WARN("Experts only. We do not know what will happen to your interpolation. And you may die. Sorry about that.");
  LUCID_CHECK_ARGUMENT(training_inputs_.size() > 0, "training_inputs", "the model is not fitted yet");
  LUCID_CHECK_ARGUMENT_EXPECTED(x.cols() == training_inputs_.cols(), "input.cols()", x.cols(), training_inputs_.cols());
  Matrix kernel_input{Matrix::NullaryExpr(
      x.rows(), training_inputs_.rows(), [this, &x, &feature_map](const Index row, const Index col) {
        return (feature_map(x.row(row)) * feature_map(training_inputs_.row(col)).transpose()).value();
      })};
  LUCID_DEBUG_FMT("Computed kernel input shape: [{} x {}]", kernel_input.rows(), kernel_input.cols());
  LUCID_TRACE_FMT("Computed kernel input: [{}]", kernel_input);
  return kernel_input * coefficients_;
}

double compute_log_marginal(const GramMatrix& K, ConstMatrixRef y) {
  LUCID_CHECK_ARGUMENT_EXPECTED(K.rows() == K.cols(), "K.rows() == K.cols()", K.rows(), K.cols());
  LUCID_CHECK_ARGUMENT_EXPECTED(K.rows() == y.rows(), "K.rows() == y.rows()", K.rows(), y.rows());
  // Compute the log marginal likelihood
  // log p(y | K, λ) = -0.5 . (y^T * w) - sum(log(diag(L))) - log(2*pi) * n / 2
  // where n is the number of training samples,
  // K is the Gram matrix,
  // λ is the regularisation constant,
  // L is the Cholesky decomposition of K + λnI
  // w are the coefficients computed as L^{-1} y
  const Matrix w{K.inverse() * y};
  double val = (-0.5 * (y.array() * w.array()).matrix().colwise().sum().array()  //  -0.5 . (y^T * w)
                - K.L().diagonal().array().log().sum()                           //  -sum(log(diag(L)))
                - std::log(2 * std::numbers::pi) * y.rows() / 2)                 // -log(2*pi) * n / 2
                   .sum();
  LUCID_CRITICAL_FMT("Log Marginal Likelihood: {}", val);
  return val;
}

Vector compute_log_marginal_gradient(const GramMatrix& K, ConstMatrixRef y, const Matrix& alpha,
                                     const std::vector<Matrix>& gradient) {
  LUCID_CHECK_ARGUMENT_EXPECTED(K.rows() == K.cols(), "K.rows() == K.cols()", K.rows(), K.cols());
  LUCID_CHECK_ARGUMENT_EXPECTED(K.rows() == y.rows(), "K.rows() == y.rows()", K.rows(), y.rows());
  // Compute the log marginal likelihood gradient
  const Matrix k_inv{K.inverse()};

  std::vector<Matrix> inner_term{};
  inner_term.reserve(y.cols());
  for (Index i = 0; i < y.cols(); ++i) {
    inner_term.emplace_back(alpha.col(i) * alpha.col(i).transpose() - k_inv);
    LUCID_CRITICAL_FMT("inner_term (1):\n{}", Matrix{alpha.col(i) * alpha.col(i).transpose()});
  }

  Matrix log_likelihood_gradient_dims{Matrix::NullaryExpr(
      gradient.size(), inner_term.size(),
      [&](const Index row, const Index col) { return gradient[row].cwiseProduct(inner_term[col]).sum() * 0.5; })};
  Vector log_likelihood_gradient{log_likelihood_gradient_dims.rowwise().sum()};

  LUCID_CRITICAL_FMT("gradient:\n{}", gradient);
  LUCID_CRITICAL_FMT("inner_term:\n{}", inner_term);
  LUCID_CRITICAL_FMT("k_inv:\n{}", k_inv);
  LUCID_CRITICAL_FMT("log_likelihood_gradient_dims:\n{}", log_likelihood_gradient_dims);
  LUCID_CRITICAL_FMT("log_likelihood_gradient:\n{}", log_likelihood_gradient);
  return log_likelihood_gradient;
}
Estimator& KernelRidgeRegressor::consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                                             const Requests requests) {
  LUCID_TRACE_FMT("KernelRidgeRegressor::consolidate({}, {}, {})", training_inputs, training_outputs, requests);

  LUCID_CHECK_ARGUMENT_EXPECTED(training_inputs.rows() == training_outputs.rows(), "training_inputs.rows()",
                                training_inputs.rows(), training_outputs.rows());
  training_inputs_ = training_inputs;
  std::vector<Matrix> kernel_gradient;
  // Compute gram matrix K (nxn) with elements K_{ij} = k(x_i, x_j)
  GramMatrix gram_matrix{*kernel_, training_inputs_, requests && Request::GRADIENT ? &kernel_gradient : nullptr};
  // Add the regularisation term to the diagonal K + λnI
  gram_matrix.add_diagonal_term(regularization_constant_ * static_cast<double>(training_inputs_.rows()));
  // Invert the gram matrix and compute the coefficients as (K + λnI)^-1 y
  coefficients_ = gram_matrix.inverse() * training_outputs;
  LUCID_TRACE_FMT("KernelRidgeRegressor::consolidate(): coefficients = [{}]", coefficients_);
  if (requests && Request::GRADIENT) {
    LUCID_TRACE_FMT("KernelRidgeRegressor::consolidate(): kernel_gradient = [{}]", kernel_gradient);
    // With the coefficients computed, we can now compute the log marginal likelihood, which is our objective value
    log_marginal_likelihood_ = compute_log_marginal(gram_matrix, training_outputs);
    // and its gradient
    compute_log_marginal_gradient(gram_matrix, training_outputs, coefficients_, kernel_gradient);
  }
  return *this;
}

double KernelRidgeRegressor::score([[maybe_unused]] ConstMatrixRef evaluation_inputs,
                                   [[maybe_unused]] ConstMatrixRef evaluation_outputs) const {
  return scorer::r2_score(*this, evaluation_inputs, evaluation_outputs);
}

std::unique_ptr<Estimator> KernelRidgeRegressor::clone() const {
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
