/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/KernelRidgeRegressor.h"

#include <memory>
#include <utility>

#include "Scorer.h"
#include "lucid/model/GaussianKernel.h"
#include "lucid/model/GramMatrix.h"
#include "lucid/model/Tuner.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

KernelRidgeRegressor::KernelRidgeRegressor(const Kernel& kernel, const Scalar regularization_constant,
                                           const std::shared_ptr<Tuner>& tuner)
    : KernelRidgeRegressor{kernel.clone(), regularization_constant, tuner} {}
KernelRidgeRegressor::KernelRidgeRegressor(std::unique_ptr<Kernel>&& kernel, const Scalar regularization_constant,
                                           const std::shared_ptr<Tuner>& tuner)
    : Estimator{tuner},
      kernel_{std::move(kernel)},
      regularization_constant_{regularization_constant},
      training_inputs_{},
      coefficients_{} {
  LUCID_CHECK_ARGUMENT_EXPECTED(regularization_constant >= 0.0, "regularization_constant", regularization_constant,
                                ">= 0.0");
  LUCID_CHECK_ARGUMENT_EXPECTED(kernel_ != nullptr, "kernel", nullptr, "not nullptr");
  LUCID_CHECK_ARGUMENT(!kernel_->has(Parameter::REGULARIZATION_CONSTANT), "kernel",
                       "parameter 'regularization constant' is hidden by the regressor");
}

Matrix KernelRidgeRegressor::predict(ConstMatrixRef x) const {
  LUCID_CHECK_ARGUMENT(training_inputs_.size() > 0, "training_inputs", "the model is not fitted yet");
  LUCID_CHECK_ARGUMENT_EXPECTED(x.cols() == training_inputs_.cols(), "input.cols()", x.cols(), training_inputs_.cols());
  Matrix kernel_input{Matrix::NullaryExpr(
      x.rows(), training_inputs_.rows(),
      [this, &x](const Index row, const Index col) { return (*kernel_)(x.row(row), training_inputs_.row(col)); })};
  LUCID_DEBUG_FMT("Computed kernel input shape: [{} x {}]", kernel_input.rows(), kernel_input.cols());
  LUCID_TRACE_FMT("Computed kernel input: \n[{}]", kernel_input);
  return kernel_input * coefficients_;
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
  LUCID_TRACE_FMT("Computed kernel input: \n[{}]", kernel_input);
  return kernel_input * coefficients_;
}

Estimator& KernelRidgeRegressor::consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) {
  LUCID_CHECK_ARGUMENT_EXPECTED(training_inputs.rows() == training_outputs.rows(), "training_inputs.rows()",
                                training_inputs.rows(), training_outputs.rows());
  training_inputs_ = training_inputs;
  // Compute gram matrix K (nxn) with elements K_{ij} = k(x_i, x_j)
  GramMatrix gram_matrix{*kernel_, training_inputs_};
  // Add the regularisation term to the diagonal K + λnI
  gram_matrix.add_diagonal_term(regularization_constant_ * static_cast<double>(training_inputs_.rows()));
  // Invert the gram matrix and compute the coefficients as (K + λnI)^-1 y
  coefficients_ = gram_matrix.inverse() * training_outputs;
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

bool KernelRidgeRegressor::has(const Parameter parameter) const {
  switch (parameter) {
    case Parameter::REGULARIZATION_CONSTANT:
      return true;
    default:
      return kernel_->has(parameter);
  }
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
