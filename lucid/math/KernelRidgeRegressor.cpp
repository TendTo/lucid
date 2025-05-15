/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include <utility>

#include "lucid/math/GaussianKernel.h"
#include "lucid/math/GramMatrix.h"
#include "lucid/math/KernelRidgeRegressor.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

template <IsAnyOf<GaussianKernel> K>
KernelRidgeRegressor<K>::KernelRidgeRegressor(K kernel, Matrix training_inputs, ConstMatrixRef training_outputs,
                                                const Scalar regularization_constant)
    : kernel_{std::move(kernel)}, training_inputs_{std::move(training_inputs)}, coefficients_{} {
  LUCID_CHECK_ARGUMENT_EXPECTED(training_inputs_.rows() == training_outputs.rows(), "training_inputs.rows()",
                                training_inputs_.rows(), training_outputs.rows());
  // Compute gram matrix K (nxn) with elements K_{ij} = k(x_i, x_j)
  GramMatrix gram_matrix{kernel_, training_inputs_};
  // Add the regularisation term to the diagonal K + λnI
  gram_matrix.add_diagonal_term(regularization_constant * static_cast<double>(training_inputs_.rows()));
  // Invert the gram matrix and compute the coefficients as (K + λnI)^-1 y
  coefficients_ = gram_matrix.inverse() * training_outputs;
}

template <IsAnyOf<GaussianKernel> K>
Matrix KernelRidgeRegressor<K>::operator()(ConstMatrixRef x) const {
  if (x.cols() != training_inputs_.cols())
    LUCID_INVALID_ARGUMENT_EXPECTED("input.cols()", x.cols(), training_inputs_.cols());
  Matrix kernel_input{Matrix::NullaryExpr(
      x.rows(), training_inputs_.rows(),
      [this, &x](const Index row, const Index col) { return kernel_(x.row(row), training_inputs_.row(col)); })};
  LUCID_DEBUG_FMT("Computed kernel input shape: [{} x {}]", kernel_input.rows(), kernel_input.cols());
  LUCID_TRACE_FMT("Computed kernel input: \n[{}]", kernel_input);
  return kernel_input * coefficients_;
}

template <IsAnyOf<GaussianKernel> K>
Matrix KernelRidgeRegressor<K>::operator()(ConstMatrixRef x, const FeatureMap& feature_map) const {
  LUCID_WARN("Experts only. We do not know what will happen to your interpolation. And you may die. Sorry about that.");
  if (x.cols() != training_inputs_.cols())
    LUCID_INVALID_ARGUMENT_EXPECTED("input.cols()", x.cols(), training_inputs_.cols());
  Matrix kernel_input{Matrix::NullaryExpr(
      x.rows(), training_inputs_.rows(), [this, &x, &feature_map](const Index row, const Index col) {
        return (feature_map(x.row(row)) * feature_map(training_inputs_.row(col)).transpose()).value();
      })};
  LUCID_DEBUG_FMT("Computed kernel input shape: [{} x {}]", kernel_input.rows(), kernel_input.cols());
  LUCID_TRACE_FMT("Computed kernel input: \n[{}]", kernel_input);
  return kernel_input * coefficients_;
}

template class KernelRidgeRegressor<GaussianKernel>;

}  // namespace lucid
