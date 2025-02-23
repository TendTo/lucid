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
KernelRidgeRegression::KernelRidgeRegression(K kernel, ConstMatrixRef training_inputs,
                                             ConstMatrixRef training_outputs, const Scalar regularization_constant) {
  // Compute gram matrix K
  GramMatrix gram_matrix{kernel, training_inputs};
  // Add the regularization term to the diagonal K + λI
  gram_matrix.add_diagonal_term(regularization_constant * static_cast<double>(training_inputs.rows()));
  // Invert the gram matrix and compute the coefficients as (K + λI)^-1 y
  Matrix coefficients{gram_matrix.inverse() * training_outputs};

  // Store the model as a function that computes the prediction for a given input
  model_ = [kernel_ = std::move(kernel), regression_inputs_ = std::move(training_inputs.eval()),
            coefficients_ = std::move(coefficients)](ConstMatrixRef inputs) {
    if (inputs.cols() != regression_inputs_.cols())
      LUCID_INVALID_ARGUMENT_EXPECTED("input.cols()", inputs.cols(), regression_inputs_.cols());
    Matrix kernel_input{Matrix::NullaryExpr(inputs.rows(), regression_inputs_.rows(),
                                            [&kernel_, &regression_inputs_, &inputs](const Index row, const Index col) {
                                              return kernel_(inputs.row(row), regression_inputs_.row(col));
                                            })};
    LUCID_TRACE_FMT("Computed kernel state: \n[{}]", kernel_input);
    return (kernel_input * coefficients_).eval();
  };
}

template KernelRidgeRegression::KernelRidgeRegression(GaussianKernel, ConstMatrixRef, ConstMatrixRef, Scalar);

}  // namespace lucid
