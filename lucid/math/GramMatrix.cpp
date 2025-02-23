/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/GramMatrix.h"

#include "lucid/math/InverseGramMatrix.h"

namespace lucid {

GramMatrix& GramMatrix::add_diagonal_term(const Scalar diagonal_term) {
  gram_matrix_.diagonal().array() += diagonal_term;
  return *this;
}
std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix) {
  return os << "GramMatrix\n" << gram_matrix.gram_matrix();
}

}  // namespace lucid
