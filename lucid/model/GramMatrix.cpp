/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/GramMatrix.h"

#include "lucid/model/InverseGramMatrix.h"
#include "lucid/util/error.h"

namespace lucid {

GramMatrix& GramMatrix::add_diagonal_term(const Scalar diagonal_term) {
  gram_matrix_.diagonal().array() += diagonal_term;
  return *this;
}
void GramMatrix::compute_decomposition() const {
  if (decomposition_.cols() == 0) decomposition_ = gram_matrix_.selfadjointView<Eigen::Lower>().llt();
  LUCID_ASSERT(decomposition_.matrixL().determinant() > 0.0, "The Gram matrix is not invertible");
  LUCID_ASSERT(decomposition_.rows() > 0, "The decomposition is not initialized");
}
std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix) {
  return os << "GramMatrix(\n" << gram_matrix.gram_matrix() << "\n)";
}

}  // namespace lucid
