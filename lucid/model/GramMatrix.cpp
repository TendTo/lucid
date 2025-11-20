/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/GramMatrix.h"

#include <string>

#include "lucid/model/InverseGramMatrix.h"
#include "lucid/util/error.h"

namespace lucid {

GramMatrix& GramMatrix::add_diagonal_term(const Scalar diagonal_term) {
  gram_matrix_.diagonal().array() += diagonal_term;
  return *this;
}
void GramMatrix::compute_decomposition() const {
  if (decomposition_.cols() == 0) {
    LUCID_CHECK_ARGUMENT_EQ(gram_matrix_.fullPivLu().isInvertible(), true);
    decomposition_ = gram_matrix_.selfadjointView<Eigen::Lower>().llt();
  }
  LUCID_ASSERT(decomposition_.rows() > 0, "The decomposition is not initialized");
}

std::string GramMatrix::to_string() const { return fmt::format("GramMatrix(\n{}\n)", gram_matrix_); }

std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix) { return os << gram_matrix.to_string(); }

}  // namespace lucid
