/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * InverseGramMatrix class.
 */
#pragma once

#include <iosfwd>

#include "lucid/lib/eigen.h"

namespace lucid {

// Forward declaration
class GramMatrix;

/**
 * The inverse of a Gram matrix, @f$ K^{-1} @f$, allowing transparent multiplication with vectors and matrices.
 * Note that only right multiplication is supported.
 * The implementation is just a thin wrapper around @ref GramMatrix::inverse_mult.
 * No data is copied, making this class valid only as long as it maintains a valid reference to the Gram matrix.
 * @code
 * Vector x;
 * GramMatrix gram_matrix{...};
 * Vector result = gram_matrix.inverse() * x;
 * @endcode
 */
class InverseGramMatrix {
 public:
  /**
   * Construct an inverse Gram matrix from a Gram matrix.
   * @param gram_matrix Gram matrix to use for the inversion.
   */
  explicit InverseGramMatrix(const GramMatrix& gram_matrix) : gram_matrix_{gram_matrix} {}

  /** @getter{underlying Gram matrix, inverse Gram matrix} */
  [[nodiscard]] const GramMatrix& gram_matrix() const { return gram_matrix_; }

  /**
   * Multiply the inverse of the Gram matrix with another matrix, @f$ K^{-1}A @f$.
   * @param A matrix to multiply with the inverse of the Gram matrix
   * @return result of the multiplication
   */
  template <class Derived>
  Matrix operator*(const MatrixBase<Derived>& A) const;

  operator Matrix() const;

 private:
  const GramMatrix& gram_matrix_;  ///< Gram matrix to use for the inversion
};

std::ostream& operator<<(std::ostream& os, const InverseGramMatrix& inverse_gram_matrix);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::InverseGramMatrix)

#endif
