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
class TruncatedFourierFeatureMap final : public FeatureMap {
 public:
  TruncatedFourierFeatureMap(long num_frequencies, Dimension input_dimension, ConstVectorRef sigma_l);

  // Matrix operator()(ConstMatrixRef x) const;
  Vector operator()(ConstVectorRef x) const;

  /** @getter{frequency matrix, truncated Fourier feature map} */
  [[nodiscard]] const Matrix& omega() const { return omega_; }
  /** @getter{weights matrix, truncated Fourier feature map} */
  [[nodiscard]] const Vector& weights() const { return weights_; }
  /** @getter{number of frequencies per dimension, truncated Fourier feature map} */
  [[nodiscard]] long num_frequencies() const { return num_frequencies_per_dimension_; }

 private:
  const long num_frequencies_per_dimension_;  ///< Number of frequencies per dimension
  Matrix omega_;                              ///< Frequencies matrix
  Vector weights_;                            ///< Weights matrix
};

}  // namespace lucid
