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

std::string InverseGramMatrix::to_string() const { return fmt::format("{}^-1", gram_matrix_.to_string()); }

std::ostream& operator<<(std::ostream& os, const InverseGramMatrix& inverse_gram_matrix) {
  return os << inverse_gram_matrix.to_string();
}

}  // namespace lucid
