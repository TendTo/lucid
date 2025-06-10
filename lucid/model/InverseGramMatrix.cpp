/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/InverseGramMatrix.h"

#include "lucid/model/GramMatrix.h"

namespace lucid {

InverseGramMatrix::operator Matrix() const {
  return gram_matrix_.inverse_mult(Matrix::Identity(gram_matrix_.rows(), gram_matrix_.cols()));
}

std::ostream& operator<<(std::ostream& os, const InverseGramMatrix& inverse_gram_matrix) {
  return os << inverse_gram_matrix.gram_matrix() << "^-1";
}

}  // namespace lucid
