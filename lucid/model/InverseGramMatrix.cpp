/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/InverseGramMatrix.h"

#include "lucid/model/GramMatrix.h"

namespace lucid {
std::ostream& operator<<(std::ostream& os, const InverseGramMatrix& inverse_gram_matrix) {
  return os << "Inverse of " << inverse_gram_matrix.gram_matrix();
}
}  // namespace lucid
