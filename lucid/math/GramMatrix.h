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
 * Given a vector space @X and a kernel function @f$ k: \mathcal{X} \times \mathcal{X} \to \mathbb{R} @f$,
 * the Gram matrix is defined as the matrix @f$ K \in \mathbb{R}^{n \times n} @f$ where @f$ K_{ij} = k(x_i, x_j) @f$.
 */
class GramMatrix {
 public:
  /**
   * Compute the Gram matrix from the kernel and the initial states.
   * The initial states should be an @sxn matrix where @n is the dimension of the vector space @X
   * and @s is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{initial_states} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_s \end{bmatrix} .
   * @f]
   * with @f$ x_i \in \mathcal{X} @f$ for @f$ 1 \le i \le s @f$.
   * @param kernel rkhs kernel used to compute the Gram matrix
   * @param initial_states @s initial states used to compute the Gram matrix
   * @param regularization_constant regularization constant added to the diagonal of the Gram matrix
   */
  GramMatrix(const Kernel& kernel, Matrix initial_states, double regularization_constant = 0);
  /**
   * Compute the Gram matrix from the kernel and the initial states and immediately compute the coefficients too.
   * The `initial_states` should be an @sxn matrix where @n is the dimension of the vector space @X
   * and @s is the number of states used to compute the Gram matrix, i.e.
   * @f[
   * \text{initial_states} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_s \end{bmatrix} .
   * @f]
   * with @f$ x_i \in \mathcal{X} @f$ for @f$ 1 \le i \le s @f$.
   * In a similar fashion, the `transition_states` should be an @sxn matrix
   * @f[
   * \text{transition_states} = \begin{bmatrix} t(x_1) \\ t(x_2) \\ \vdots \\ t(x_s) \end{bmatrix}
   * @f]
   * where @f$ t: \mathcal{X} \to \mathcal{X} @f$ is some transition function.
   * @param kernel rhks kernel used to compute the Gram matrix
   * @param initial_states @s row vector initial states used to compute the Gram matrix
   * @param transition_states @s row vector states obtained after applying the transition function to each initial state
   * @param regularization_constant regularization constant added to the diagonal of the Gram matrix
   */
  GramMatrix(const Kernel& kernel, Matrix initial_states, const Matrix& transition_states,
             double regularization_constant = 0);

  /**
   * Given the initial state @f$ x \in \mathcal{X} @f$, compute the coefficients @f$ \alpha \in \mathbb{R}^s @f$.
   * The coefficients will be stored to be used later to interpolate the transition function on an arbitrary state.
   * @param transition_states @s row vector states obtained after applying the transition function to each initial state
   */
  void compute_coefficients(const Matrix& transition_states);

  Vector operator()(const Vector& state) const;
  Matrix operator()(const Matrix& state) const;

  /** @checker{ready to be used, i.e. the coefficients have been computed, Gramm matrix} */
  [[nodiscard]] bool is_computed() const { return coefficients_.size() != 0; }
  /** @getter{internal matrix structure, Gramm matrix} */
  [[nodiscard]] const Matrix& gram_matrix() const { return gram_matrix_; }
  /** @getter{coefficients used to interpolate the transition function, Gramm matrix} */
  [[nodiscard]] const Matrix& coefficients() const;

 private:
  const Kernel& kernel_;   ///< Kernel function
  Matrix initial_states_;  ///< Initial states
  Matrix gram_matrix_;     ///< Gram matrix
  Matrix coefficients_;    ///< Coefficients used to interpolate the transition function
};

std::ostream& operator<<(std::ostream& os, const GramMatrix& gram_matrix);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GramMatrix)

#endif
