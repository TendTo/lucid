/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FeatureMap class.
 */
#pragma once

#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/model/FeatureMap.h"
#include "lucid/model/RectSet.h"

namespace lucid {

/**
 * Truncated Fourier feature map.
 * It maps vectors from @XsubRd to a higher-dimensional space using the truncated Fourier series.
 * The number of frequencies, @f$ M > 0 @f$, determines the dimension of the output space.
 * All possible combinations of frequencies are computed for each input dimension.
 * Therefore, the output vector has a dimension of @f$ 2^d M + 1 @f$.
 * The feature map is computed as
 * @f[
 * \phi_{M}(x) = \sigma_{f}\begin{bmatrix}
 * w_{0}\\
 * \sqrt{2}w_{1}\cos(\omega_{1}^{\top}P(x))\\
 * \sqrt{2}w_{1}\sin(\omega_{1}^{\top}P(x))\\
 * \vdots\\
 * \sqrt{2}w_{M}\cos(\omega_{M}^{\top}P(x))\\
 * \sqrt{2}w_{M}\sin(\omega_{M}^{\top}P(x))
 * \end{bmatrix},
 * @f]
 * The way the weights and omegas are computed depends on the subclass.
 */
class TruncatedFourierFeatureMap : public FeatureMap {
 public:
  /**
   * Construct a truncated Fourier feature map.
   * @pre `num_frequencies` must be greater than 0.
   * @pre `sigma_f` must be greater than 0.
   * @param num_frequencies number of frequencies per dimension. Includes the zero frequency
   * @param prob_per_dim probability distribution of frequencies per dimension, @f$ \mathbb{P}(\zeta_j) @f$
   * @param omega_per_dim matrix omega where each row is a dimension and each column is a frequency coefficient
   * @param sigma_l length scales per dimension
   * @param sigma_f scaling factor
   * @param X_bounds domain of the input space, @XsubRd
   */
  TruncatedFourierFeatureMap(int num_frequencies, const Matrix& prob_per_dim, const Matrix& omega_per_dim,
                             ConstVectorRef sigma_l, Scalar sigma_f, const RectSet& X_bounds);
  /**
   * Construct a truncated Fourier feature map.
   * @pre `num_frequencies` must be greater than 0.
   * @pre `sigma_f` must be greater than 0.
   * @param num_frequencies number of frequencies per dimension. Includes the zero frequency
   * @param prob_per_dim probability distribution of frequencies per dimension, @f$ \mathbb{P}(\zeta_j) @f$
   * @param omega_per_dim matrix omega where each row is a dimension and each column is a frequency coefficient
   * @param sigma_l length scales per dimension
   * @param sigma_f scaling factor
   * @param X_bounds domain of the input space, @XsubRd
   */
  TruncatedFourierFeatureMap(int num_frequencies, const Matrix& prob_per_dim, const Matrix& omega_per_dim,
                             double sigma_l, Scalar sigma_f, const RectSet& X_bounds);
  /**
   * Construct a truncated Fourier feature map.
   * It will not compute the cross-frequencies of the basis, thus reducing the problem size significantly with
   * respect to the other constructor.
   * @pre `num_frequencies` must be greater than 0.
   * @pre `sigma_f` must be greater than 0.
   * @param num_frequencies number of frequencies per dimension. Includes the zero frequency
   * @param prob_per_dim probability distribution of frequencies per dimension, @f$ \mathbb{P}(\zeta_j) @f$
   * @param omega_per_dim matrix omega where each row is a dimension and each column is a frequency coefficient
   * @param sigma_f scaling factor
   * @param X_bounds domain of the input space, @XsubRd
   * @param unused unused parameter to differentiate the constructor from the other one
   */
  TruncatedFourierFeatureMap(int num_frequencies, const Matrix& prob_per_dim, const Matrix& omega_per_dim,
                             Scalar sigma_f, const RectSet& X_bounds, bool unused);

  /**
   * Given a @d dimensional vector @x, project it to the unit hypercube @f$ [0, 1]^d @f$, then compute the feature map.
   * @param x input vector
   * @return @f$ 2 M + 1 @f$ dimensional feature map
   */
  [[nodiscard]] Vector map_vector(ConstVectorRef x) const;
  /**
   * Given an @nxd dimensional matrix @x, project each row vector to the unit hypercube @f$ [0, 1]^d @f$,
   * then compute the feature map.
   * @param x input vector
   * @return @f$ n \times 2 M + 1 @f$ dimensional feature map
   */
  [[nodiscard]] Matrix map_matrix(ConstMatrixRef x) const;

  /**
   * Given an @nxd dimensional matrix @x, project each row vector to the unit hypercube @f$ [0, 1]^d @f$,
   * then compute the feature map.
   * @param x input vector
   * @return @f$ n \times 2 M + 1 @f$ dimensional feature map
   */
  [[nodiscard]] Matrix apply_impl(ConstMatrixRef x) const override;

  /** @getter{dimension, the feature map space} */
  [[nodiscard]] Dimension dimension() const { return weights_.size(); }
  /** @getter{frequency matrix, truncated Fourier feature map} */
  [[nodiscard]] const Matrix& omega() const { return omega_; }
  /** @getter{weights matrix, truncated Fourier feature map} */
  [[nodiscard]] const Vector& weights() const { return weights_; }
  /** @getter{number of frequencies per dimension, truncated Fourier feature map} */
  [[nodiscard]] int num_frequencies() const { return num_frequencies_per_dimension_; }
  /** @getter{probability captured by the Fourier expansion, truncated Fourier feature map, NaN if not computed} */
  [[nodiscard]] Scalar captured_probability() const { return captured_probability_; }
  /** @getter{limits, original input space} */
  [[nodiscard]] const RectSet& X_bounds() const { return X_bounds_; }
  /** @getter{sigma_f value, truncated Fourier feature map} */
  [[nodiscard]] double sigma_f() const { return sigma_f_; }
  /** @getter{periodic coefficients, truncated Fourier feature map} */
  [[nodiscard]] const Vector& periodic_coefficients() const { return periodic_coefficients_; }

  /**
   * Return the periodic input domain for this linear truncated Fourier map.
   * We want to find a space such that the smallest frequency is able to complete a full period.
   * Given that frequencies are defined as @f$ \omega P(x) @f$,
   * where @f$ P(x) @f$ is the projection to the unit hypercube of the original domain,
   * we want to find the upper bound of the periodic domain @f$ \bar{x} @f$ such that
   * @f[
   * \omega P(\bar{x}) = 2\pi
   * @f]
   * for the smallest @f$ \omega @f$.
   * Graphically,
   * @code{.unparsed}
   *                                New max (x̄)
   *   ┌───────────────────────────────●
   *   │                               │
   *   │                               │
   *   │                     Old max   │
   *   ├───────────────────────●       │
   *   │                       │       │
   *   │                       │       │
   *   │                       │       │
   *   │                       │       │
   *   │ Min                   │       │
   *   ●───────────────────────┴───────┘
   * @endcode
   * Notice how the lower bound remains fixed, while the upper bound is shifted to create the periodic domain.
   * @note The periodic domain could be smaller than the original domain, depending on the values of @f$ \sigma_l @f$.
   * @pre @sigmal must have the same dimension as the input space.
   * @pre All values in @sigmal must be greater than 0.
   * @return new RectSet representing the periodic input domain
   */
  [[nodiscard]] virtual RectSet get_periodic_set() const;

  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override;

 protected:
  int num_frequencies_per_dimension_;  ///< Number of frequencies per dimension
  Matrix omega_;                       ///< Frequencies matrix
  Vector weights_;                     ///< Weights matrix
  Scalar sigma_f_;                     ///< @sigmaf value
  Vector sigma_l_;                     ///< @sigmal vector
  RectSet X_bounds_;                   ///< Limits of the input space expressed as a matrix. The set is a rectangle
  Scalar captured_probability_;        ///< Probability captured by the Fourier expansion. NaN if not computed
  Vector periodic_coefficients_;       ///< Coefficient to convert from the truncated Fourier basis to the periodic one
};

}  // namespace lucid
