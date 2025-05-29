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
#include "lucid/model/Kernel.h"

namespace lucid {

/**
 * RKHS Gaussian kernel.
 * Given a vector space @X and two vectors @f$ x_1, x_2 \in \mathcal{X} @f$, the Gaussian kernel is defined as
 * @f[
 * k(x_1, x_2) = \sigma_f^2 \exp\left(-\frac{1}{2} (x_1 - x_2)^T\Sigma(x_1 - x_2)\right)
 * @f]
 * where  @f$ \Sigma = \text{diag}( \sigma _l )^{-2} @f$.
 */
class GaussianKernel final : public Kernel {
 public:
  using Kernel::set;
  using Kernel::operator();
  /**
   * Construct a new GaussianKernel object with the given parameters.
   * @param sigma_l @sigma_l value
   * @param sigma_f @sigma_f value
   */
  explicit GaussianKernel(const Vector& sigma_l, double sigma_f = 1.0);
  /**
   * Construct a new GaussianKernel object with the given parameters.
   * @param dim dimension of the vector space
   * @param sigma_l @sigma_l value. It is equal for all dimensions.
   * @param sigma_f @sigma_f value
   */
  explicit GaussianKernel(Dimension dim, double sigma_l = 1.0, double sigma_f = 1.0);

  /** @getter{@sigma_f value, kernel} */
  [[nodiscard]] Scalar sigma_f() const { return sigma_f_; }
  /** @getter{@sigma_l value, kernel} */
  [[nodiscard]] const Vector& sigma_l() const { return sigma_l_; }
  /** @getter{dimension, kernel} */
  [[nodiscard]] Dimension dimension() const { return sigma_l_.size(); }
  /** @getter{cached @f$ \text{diag}(0.5 \sigma_l^2) @f$ value, kernel} */
  [[nodiscard]] const Vector& gamma() const { return gamma_; }

  [[nodiscard]] bool is_stationary() const override { return true; }
  [[nodiscard]] std::unique_ptr<Kernel> clone() const override;

  [[nodiscard]] bool has(Parameter parameter) const override;
  void set(Parameter parameter, double value) override;
  void set(Parameter parameter, const Vector& value) override;

 private:
  Matrix operator()(ConstMatrixRef x1, ConstMatrixRef x2, double* gradient) const override;
  [[nodiscard]] double get_d(Parameter parameter) const override;
  [[nodiscard]] const Vector& get_v(Parameter parameter) const override;

  Vector sigma_l_;  ///< @sigma_l value
  double sigma_f_;  ///< @sigma_f value
  Vector gamma_;    ///< @f$ \text{diag}(0.5 \sigma_l^2) @f$ cached for performance.
                    ///< Being vector, `.asDiagonal()` is needed to convert it to a diagonal matrix before use
};

using RadialBasisFunction = GaussianKernel;       ///< Alias for Gaussian kernel.
using SquaredExponentialKernel = GaussianKernel;  ///< Alias for Gaussian kernel.

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GaussianKernel)

#endif
