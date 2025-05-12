/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GaussianKernel class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/math/Kernel.h"

namespace lucid {

/**
 * RKHS Gaussian kernel.
 * Given a vector space @X and two vectors @f$ x_1, x_2 \in \mathcal{X} @f$, the Gaussian kernel is defined as
 * @f[
 * k(x_1, x_2) = \sigma_f^2 \exp\left(-\frac{1}{2} (x_1 - x_2)^T\Sigma(x_1 - x_2)\right)
 * @f]
 * where  @f$ \Sigma = \text{diag}(\sigma_l^2) @f$.
 */
class GaussianKernel final : public Kernel {
 public:
  /**
   * Construct a new GaussianKernel object with the given parameters.
   * @param params kernel parameters. The first element is @f$ \sigma_f @f$ and the rest are @f$ \sigma_l @f$.
   */
  explicit GaussianKernel(Vector params);
  /**
   * Construct a new GaussianKernel object with the given parameters.
   * @param sigma_f @f$ \sigma_f @f$ value
   * @param sigma_l @f$ \sigma_l @f$ value
   */
  GaussianKernel(double sigma_f, const Vector& sigma_l);
  /**
   * Construct a new GaussianKernel object with the given parameters.
   * @param sigma_f @f$ \sigma_f @f$ value
   * @param sigma_l @f$ \sigma_l @f$ value. Is equal for all dimensions.
   * @param size dimension of the vector space
   */
  GaussianKernel(double sigma_f, double sigma_l, Dimension size);
  /** @getter{@f$ \sigma_f @f$ value, kernel} */
  [[nodiscard]] Scalar sigma_f() const { return parameters_(0); }
  /** @getter{@f$ \sigma_l @f$ value, kernel} */
  [[nodiscard]] ConstVectorBlock sigma_l() const { return parameters_.tail(parameters_.size() - 1); }

  Scalar operator()(const Vector& x1, const Vector& x2) const override;
  [[nodiscard]] std::unique_ptr<Kernel> clone() const override;
  [[nodiscard]] std::unique_ptr<Kernel> clone(const Vector& params) const override;

 private:
  Vector sigma_l_sq_diagonal_inv_;  ///< @f$ \text{diag}(\sigma_l^2) @f$ cached for performance.
                             ///< Being vector, `.asDiagonal()` is needed to convert it to a diagonal matrix before use
};

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GaussianKernel)

#endif
