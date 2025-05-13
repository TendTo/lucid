/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FeatureMap class.
 */
#pragma once

#include "lucid/lib/eigen.h"
#include "lucid/math/FeatureMap.h"
#include "lucid/math/RectSet.h"

namespace lucid {

/**
 * Truncated Fourier feature map.
 * It maps vectors from @X to a higher-dimensional space using the truncated Fourier series.
 * The number of frequencies determines the dimension of the output space.
 * Furthermore, all possible combinations of frequencies are computed for each input dimension.
 * Therefore, the output space has a dimension of @f$ 2^\text{input_dimension} \times \text{num_frequencies} @f$.
 * The output of this feature map is computed as
 * @f[
 * \phi_{M}(x) = \sigma_{f}\begin{bmatrix}
 * w_{0}\\
 * \sqrt{2}w_{1}\cos(\omega_{1}^{\top}P(x))\\
 * \sqrt{2}w_{1}\sin(\omega_{1}^{\top}P(x))\\
 * \vdots\\
 * \sqrt{2}w_{M}\cos(\omega_{M}^{\top}P(x))\\
 * \sqrt{2}w_{M}\sin(\omega_{M}^{\top}P(x))
 * \end{bmatrix}
 * @f]
 * characterized by @f$ M \in \mathbb{N}_{>0} @f$ wavenumbers,
 * @f$ \omega_j \coloneqq 2\pi\zeta_{j}\ \in \mathbb{R}^n @f$, @f$ 1 \leq j \leq M @f$ with multi-index
 * @f$ \zeta_{j} \in \mathbb{N}_{\ge 0}^n @f$ and weights @f$ w_j \in \mathbb{R} @f$ computed from the n-dimensional
 * cumulative distribution function of the normal distribution with standard deviation @f$ \sigma_l @f$, i.e.
 * @f[
 *  w_j^2 \coloneqq \int_{(2\zeta_j-1)\pi}^{(2\zeta_j+1)\pi}\mathcal{N}(d\xi|0,\Sigma), 1 \le j \le M .
 * @f]
 */
class TruncatedFourierFeatureMap : public FeatureMap {
 public:
  TruncatedFourierFeatureMap(int num_frequencies, const Matrix& prob_dim_wise, Scalar sigma_f, const RectSet& x_limits);

  /**
   * Given an @d dimensional vector @x, project it to the unit hypercube @f$ [0, 1]^d @f$, then compute the feature map.
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

  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const override;

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

 protected:
  int num_frequencies_per_dimension_;  ///< Number of frequencies per dimension
  Matrix omega_;                       ///< Frequencies matrix
  Vector weights_;                     ///< Weights matrix
  Scalar sigma_f_;                     ///< Sigma_f value
  RectSet x_limits_;                   ///< Limits of the input space expressed as a matrix. The set is a rectangle
  Scalar captured_probability_;        ///< Probability captured by the Fourier expansion. NaN if not computed
};

}  // namespace lucid
