/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GaussianKernel class.
 */
#pragma once

#include <iosfwd>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/math/Kernel.h"

namespace lucid {

/**
 * RKHS Gaussian kernel.
 * The Gaussian kernel is defined as
 * @f[
 * k(x_1, x_2) = \sigma_f^2 \exp\left(-\frac{1}{2} (x_1 - x_2)^T\Sigma(x_1 - x_2)\right)
 * @f]
 * where @f$ \Sigma = \text{diag}(\sigma_l^2) @f$.
 */
class GaussianKernel final : public Kernel {
 public:
  GaussianKernel(const double sigma_f, Vector sigma_l) : sigma_f_{sigma_f}, sigma_l_{std::move(sigma_l)} {
    sigma_l_.array() = sigma_l_.array().unaryExpr([](const Scalar& sigma) { return 1.0 / (sigma * sigma); }).eval();
  }

  /** @getter{@f$ \sigma_f @f$ value, kernel} */
  [[nodiscard]] Scalar sigma_f() const { return sigma_f_; }
  /** @getter{@f$ \sigma_l @f$ value, kernel} */
  [[nodiscard]] const Vector& sigma_l() const { return sigma_l_; }

  Scalar operator()(const Vector& x1, const Vector& x2) const override;

 private:
  Scalar sigma_f_;  ///< @f$ \sigma_f @f$
  Vector sigma_l_;  ///< @f$ \sigma_l @f$
};

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GaussianKernel)

#endif
