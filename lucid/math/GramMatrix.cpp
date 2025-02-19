/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/GramMatrix.h"

#include <utility>

#include "lucid/util/error.h"

namespace lucid {

GramMatrix::GramMatrix(const Kernel& kernel, Matrix initial_states, const double regularization_constant)
    : kernel_{kernel},
      initial_states_{std::move(initial_states)},
      gram_matrix_{initial_states_.rows(), initial_states_.rows()},
      coefficients_{initial_states_.rows(), initial_states_.cols()} {
  if (!coefficients_.rows() || !coefficients_.cols()) LUCID_INVALID_ARGUMENT("states", "empty");
  // TODO(tend): should this be kernel(initial_states_.col(i), initial_states_.col(i))?
  gram_matrix_.diagonal() = Vector::Constant(
      gram_matrix_.rows(), kernel(initial_states_.row(0), initial_states_.row(0)) + regularization_constant);
  for (Index row_idx = 0; row_idx < gram_matrix_.rows(); ++row_idx) {
    for (Index col_idx = 0; col_idx < row_idx; ++col_idx) {
      gram_matrix_(row_idx, col_idx) = kernel(initial_states_.row(row_idx), initial_states_.row(col_idx));
    }
  }
  LUCID_TRACE_FMT("Computed gram matrix: \n[{}]", gram_matrix_);
}
GramMatrix::GramMatrix(const Kernel& kernel, Matrix initial_states, const Matrix& transition_states,
                       const double regularization_constant)
    : GramMatrix(kernel, std::move(initial_states), regularization_constant) {
  compute_coefficients(transition_states);
}
void GramMatrix::compute_coefficients(const Matrix& transition_states) {
  if (transition_states.cols() != initial_states_.cols())
    LUCID_INVALID_ARGUMENT_EXPECTED("transition_states.cols()", transition_states.cols(), initial_states_.cols());
  if (transition_states.rows() != initial_states_.rows())
    LUCID_INVALID_ARGUMENT_EXPECTED("transition_states.rows()", transition_states.rows(), initial_states_.rows());
  coefficients_ = gram_matrix_.selfadjointView<Eigen::Lower>().ldlt().solve(transition_states);
  LUCID_TRACE_FMT("Computed coefficients: \n[{}]", coefficients_);
}
Vector GramMatrix::operator()(const Vector& state) const {
  if (coefficients_.size() == 0) LUCID_RUNTIME_ERROR("Coefficients have not been computed yet.");
  if (state.size() != initial_states_.rows())
    LUCID_INVALID_ARGUMENT_EXPECTED("state.size()", state.size(), initial_states_.rows());
  Vector kernel_state{Vector::NullaryExpr(
      gram_matrix_.rows(), [this, &state](const Index i) { return kernel_(initial_states_.row(i), state); })};
  LUCID_TRACE_FMT("Computed kernel state: \n[{}]", kernel_state);
  return kernel_state.transpose() * coefficients_;
}
Matrix GramMatrix::operator()(const Matrix& state) const {
  if (coefficients_.size() == 0) LUCID_RUNTIME_ERROR("Coefficients have not been computed yet.");
  if (state.rows() != initial_states_.rows())
    LUCID_INVALID_ARGUMENT_EXPECTED("state.cols()", state.cols(), initial_states_.cols());
  Matrix kernel_state{Matrix::NullaryExpr(
      gram_matrix_.rows(), state.rows(),
      [this, &state](const Index row, const Index col) { return kernel_(initial_states_.row(row), state.row(col)); })};
  LUCID_TRACE_FMT("Computed kernel state: \n[{}]", kernel_state);
  return kernel_state.transpose() * coefficients_;
}
const Matrix& GramMatrix::coefficients() const {
  if (coefficients_.size() == 0) LUCID_RUNTIME_ERROR("Coefficients have not been computed yet.");
  return coefficients_;
}
std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix) {
  os << "GramMatrix\n"
     << "Matrix:\n"
     << gram_matrix.gram_matrix();
  if (gram_matrix.is_computed()) os << "\nCoefficients: " << gram_matrix.coefficients();
  return os;
}

}  // namespace lucid
