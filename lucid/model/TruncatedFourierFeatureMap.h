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
 * where @f$ \omega_j := 2\pi\zeta_{j}\ \in \mathbb{R}^d @f$, @f$ 1 \leq j \leq M @f$ with multi-index
 * @f$ \zeta_{j} \in \mathbb{N}_{\ge 0}^d @f$ and weights @f$ w_j \in \mathbb{R} @f$.
 * The way the weights are computed depends on the subclass.
 */
class TruncatedFourierFeatureMap : public FeatureMap {
 public:
  /**
   * Construct a truncated Fourier feature map.
   * @pre `num_frequencies` must be greater than 0.
   * @pre `sigma_f` must be greater than 0.
   * @param num_frequencies number of frequencies per dimension. Includes the zero frequency
   * @param prob_dim_wise probability distribution of frequencies per dimension, @f$ \mathbb{P}(\zeta_j) @f$
   * @param sigma_f scaling factor
   * @param x_limits domain of the input space, @XsubRd
   */
  TruncatedFourierFeatureMap(int num_frequencies, const Matrix& prob_dim_wise, Scalar sigma_f, const RectSet& x_limits);

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

  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override;

 protected:
  int num_frequencies_per_dimension_;  ///< Number of frequencies per dimension
  Matrix omega_;                       ///< Frequencies matrix
  Vector weights_;                     ///< Weights matrix
  Scalar sigma_f_;                     ///< Sigma_f value
  RectSet x_limits_;                   ///< Limits of the input space expressed as a matrix. The set is a rectangle
  Scalar captured_probability_;        ///< Probability captured by the Fourier expansion. NaN if not computed
};

}  // namespace lucid
