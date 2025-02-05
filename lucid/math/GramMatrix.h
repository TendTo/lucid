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

#include "lucid/math/Kernel.h"

namespace lucid {

/**
 * Gram Matrix obtained from a kernel function.
 * Given a vector space @X
 * and a kernel function @f$ k: \mathcal{X} \times \mathcal{X} \to \mathbb{R} @f$,
 * the Gram matrix is defined as the matrix @f$ K \in \mathbb{R}^{n \times n} @f$ where @f$ K_{ij} = k(x_i, x_j) @f$.
 */
class GramMatrix {
 public:
  /**
   * Compute the Gram matrix from the kernel and the initial states.
   * Initial states should be an `n` x `num_states` matrix where `n` is the dimension of the vector space @X
   * and `num_states` is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{initial_states} = \begin{bmatrix} x_1 & x_2 & \cdots & x_{\text{num_states}} \end{bmatrix}
   * @f]
   * with @f$ x_i \in \mathcal{X} @f$.
   * @param kernel rhks kernel used to compute the Gram matrix
   * @param initial_states `num_states` initial states used to compute the Gram matrix
   * @param regularization_constant regularization constant added to the diagonal of the Gram matrix
   */
  GramMatrix(const Kernel& kernel, Matrix initial_states, double regularization_constant = 0);
  /**
   * Compute the Gram matrix from the kernel and the initial states and immediately compute the coefficients too.
   * Initial states should be an `n` x `num_states` matrix where `n` is the dimension of the vector space @X
   * and `num_states` is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{initial_states} = \begin{bmatrix} x_1 & x_2 & \cdots & x_{\text{num_states}} \end{bmatrix}
   * @f]
   * with @f$ x_i \in \mathcal{X} @f$.
   * The `transition_states` should be an `n` x `num_states` matrix where `n` is the dimension of the vector space @X
   * and `num_states` is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{transition_states} = \begin{bmatrix} x_1^+ & x_2^+ & \cdots & x_{\text{num_states}}^+ \end{bmatrix}
   * @f]
   * where @f$ x_i^+ = t(x_i) @f$.
   * @param kernel rhks kernel used to compute the Gram matrix
   * @param initial_states `num_states` states used to compute the Gram matrix
   * @param transition_states `num_states` states obtained after applying the transition function to each initial state
   * @param regularization_constant regularization constant added to the diagonal of the Gram matrix
   */
  GramMatrix(const Kernel& kernel, Matrix initial_states, const Matrix& transition_states,
             double regularization_constant = 0);

  void compute_coefficients(const Matrix& transition_states);

  Vector operator()(const Vector& state) const;
  Matrix operator()(const Matrix& state) const;

  [[nodiscard]] bool is_computed() const { return coefficients_.size() != 0; }
  [[nodiscard]] const Matrix& gram_matrix() const { return gram_matrix_; }
  [[nodiscard]] const Matrix& coefficients() const;

 private:
  const Kernel& kernel_;
  Matrix initial_states_;
  Matrix gram_matrix_;
  Matrix coefficients_;
};

std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GramMatrix)

#endif
