/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "KernelRidgeRegression.h"

#include "lucid/math/GaussianKernel.h"
#include "lucid/math/GramMatrix.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

template <IsAnyOf<GaussianKernel> K>
KernelRidgeRegression<K>::KernelRidgeRegression(K kernel, Matrix training_inputs, ConstMatrixRef training_outputs,
                                                const Scalar regularization_constant)
    : kernel_{std::move(kernel)}, training_inputs_{std::move(training_inputs)} {
  // Compute gram matrix K
  GramMatrix gram_matrix{kernel_, training_inputs_};
  // Add the regularization term to the diagonal K + λI
  gram_matrix.add_diagonal_term(regularization_constant * static_cast<double>(training_inputs_.rows()));
  // Invert the gram matrix and compute the coefficients as (K + λI)^-1 y
  coefficients_ = gram_matrix.inverse() * training_outputs;
}

template <IsAnyOf<GaussianKernel> K>
Matrix KernelRidgeRegression<K>::operator()(ConstMatrixRef x) const {
  if (x.cols() != training_inputs_.cols())
    LUCID_INVALID_ARGUMENT_EXPECTED("input.cols()", x.cols(), training_inputs_.cols());
  Matrix kernel_input{Matrix::NullaryExpr(
      x.rows(), training_inputs_.rows(),
      [this, &x](const Index row, const Index col) { return kernel_(x.row(row), training_inputs_.row(col)); })};
  LUCID_TRACE_FMT("Computed kernel state: \n[{}]", kernel_input);
  return kernel_input * coefficients_;
}

template class KernelRidgeRegression<GaussianKernel>;

}  // namespace lucid
