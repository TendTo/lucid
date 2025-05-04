/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GramMatrix class.
 */
#pragma once

#include <iosfwd>
#include <utility>
#include <vector>

#include "lucid/math/InverseGramMatrix.h"
#include "lucid/math/Kernel.h"
#include "lucid/util/exception.h"

namespace lucid {

/**
 * Gram Matrix obtained from a kernel function.
 * Given a vector space @X and a kernel function @f$ k: \mathcal{X} \times \mathcal{X} \to \mathbb{R} @f$,
 * the Gram matrix is defined as the matrix @f$ K \in \mathbb{R}^{s \times s} @f$ where @f$ K_{ij} = k(x_i, x_j) @f$ or,
 * equivalently,
 * @f[
 * K = \begin{bmatrix}
 *     k(x_1, x_1) & k(x_1, x_2) & \cdots & k(x_1, x_s) \\
 *     k(x_2, x_1) & k(x_2, x_2) & \cdots & k(x_2, x_s) \\
 *     \vdots      & \vdots      & \ddots & \vdots      \\
 *     k(x_s, x_1) & k(x_s, x_2) & \cdots & k(x_s, x_s)
 * \end{bmatrix} .
 * @f]
 */
class GramMatrix {
 public:
  /**
   * Compute the Gram matrix from the kernel and the initial states.
   * The initial states should be an @Nxn matrix where @n is the dimension of the vector space @X
   * and @N is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{initial_states} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_s \end{bmatrix} .
   * @f]
   * with @f$ x_i \in \mathcal{X} @f$ for @f$ 1 \le i \le s @f$.
   * @param kernel rkhs kernel used to compute the Gram matrix
   * @param initial_states @N initial states used to compute the Gram matrix
   * @param regularization_constant regularization constant added to the diagonal of the Gram matrix
   */
  template <class Derived>
  GramMatrix(const Kernel& kernel, const MatrixBase<Derived>& initial_states)
      : gram_matrix_{initial_states.rows(), initial_states.rows()} {
    gram_matrix_.diagonal() =
        Vector::Constant(gram_matrix_.rows(), kernel(initial_states.row(0), initial_states.row(0)));
    for (Index row_idx = 0; row_idx < gram_matrix_.rows(); ++row_idx) {
      for (Index col_idx = 0; col_idx < row_idx; ++col_idx) {
        gram_matrix_(row_idx, col_idx) = kernel(initial_states.row(row_idx), initial_states.row(col_idx));
      }
    }
  }

  /**
   * Compute the Gram matrix from the kernel and the initial states and immediately compute the coefficients too.
   * The `initial_states` should be an @Nxn matrix where @n is the dimension of the vector space @X
   * and @N is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{initial_states} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_s \end{bmatrix} .
   * @f]
   * with @f$ x_i \in \mathcal{X} @f$ for @f$ 1 \le i \le s @f$.
   * In a similar fashion, the `transition_states` should be an @Nxn matrix
   * @f[
   * \text{transition_states} = \begin{bmatrix} t(x_1) \\ t(x_2) \\ \vdots \\ t(x_s) \end{bmatrix}
   * @f]
   * where @f$ t: \mathcal{X} \to \mathcal{X} @f$ is some transition function.
   * @param kernel rhks kernel used to compute the Gram matrix
   * @param initial_states @N row vector initial states used to compute the Gram matrix
   * @param transition_states @N row vector states obtained after applying the transition function to each initial state
   * @param regularization_constant regularization constant added to the diagonal of the Gram matrix
   */
  // GramMatrix(const Kernel& kernel, Matrix initial_states);

  /**
   * Given the initial state @f$ x \in \mathcal{X} @f$, compute the coefficients @f$ \alpha \in \mathbb{R}^s @f$.
   * The coefficients will be stored to be used later to interpolate the transition function on an arbitrary state.
   * @param transition_states @N row vector states obtained after applying the transition function to each initial state
   */
  void compute_coefficients(const Matrix& transition_states);

  [[nodiscard]] InverseGramMatrix inverse() const { return InverseGramMatrix{*this}; }

  /**
   * Add a diagonal term to the Gram matrix.
   * Useful when dealing with ill-conditioned matrices or regularization.
   * @param diagonal_term term to add to the diagonal
   * @return reference to the object
   */
  GramMatrix& add_diagonal_term(Scalar diagonal_term);

  /**
   * Right multiply the Gram matrix with another matrix, @f$ KA @f$.
   * @tparam Derived type of the other matrix
   * @param A matrix to multiply with the Gram matrix
   * @return result of the multiplication
   */
  template <class Derived>
  Matrix operator*(const MatrixBase<Derived>& A) const {
    if (A.rows() != gram_matrix_.rows())
      throw exception::LucidInvalidArgumentException("A.rows() != gram_matrix.rows()");
    return gram_matrix_ * A;
  }

  /**
   * Right multiply the inverse of the Gram matrix with another matrix, @f$ K^{-1}A @f$.
   * @tparam Derived type of the other matrix
   * @param A matrix to multiply with the Gram matrix
   * @return result of the multiplication
   */
  template <class Derived>
  [[nodiscard]] Matrix inverse_mult(const MatrixBase<Derived>& A) const {
    if (A.rows() != gram_matrix_.rows())
      throw exception::LucidInvalidArgumentException("A.rows() != gram_matrix.rows()");
    return gram_matrix_.selfadjointView<Eigen::Lower>().ldlt().solve(A);
  }

  /** @getter{internal matrix structure, Gramm matrix} */
  [[nodiscard]] const Matrix& gram_matrix() const { return gram_matrix_; }
  /** @getter{number of rows, Gramm matrix} */
  [[nodiscard]] Dimension rows() const { return gram_matrix_.rows(); }
  /** @getter{number of columns, Gramm matrix} */
  [[nodiscard]] Dimension cols() const { return gram_matrix_.cols(); }

 private:
  Matrix gram_matrix_;  ///< Gram matrix
};

// Because of templates, we need to define the inverse Gram matrix multiplication in the header file.
template <class Derived>
Matrix InverseGramMatrix::operator*(const MatrixBase<Derived>& A) const {
  return gram_matrix_.inverse_mult(A);
}

/**
 * Left multiply the Gram matrix with another matrix, @f$ AK @f$.
 * @tparam Derived type of the other matrix
 * @param A matrix to multiply with the Gram matrix
 * @param gram_matrix Gram matrix to multiply with
 * @return result of the multiplication
 */
template <class Derived>
Matrix operator*(const MatrixBase<Derived>& A, const GramMatrix& gram_matrix) {
  return A * gram_matrix.gram_matrix();
}

std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GramMatrix)

#endif
