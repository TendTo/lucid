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
#include <vector>

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
 * The type of @sigma_l on kernel construction determines whether the kernel is isotropic or anisotropic.
 * If @sigma_l is a vector, then the kernel is anisotropic and each dimension has its own length scale.
 * Such a kernel can only be used on inputs that belong to the same vector space
 * (i.e., the number of columns) as @sigma_l.
 * On the other hand, if @sigma_l is a single value, then the kernel is isotropic.
 * The same value is used for all dimensions, and the kernel can be used on inputs of any dimension,
 * since the internal @ref sigma_l_ will be automatically adjusted to match the input dimension.
 */
class GaussianKernel final : public Kernel {
 public:
  using Kernel::set;
  using Kernel::operator();
  /**
   * Construct a new anisotropic GaussianKernel object with the given parameters.
   * @pre `sigma_l` must contain at least one element all elements must be greater than 0
   * @param sigma_l @sigma_l value
   * @param sigma_f @sigma_f value
   */
  explicit GaussianKernel(Vector sigma_l, double sigma_f = 1.0);
  /**
   * Construct a new isotropic GaussianKernel object with the given parameters.
   * @pre `sigma_l` must be greater than 0
   * @param sigma_l @sigma_l value. It is equal for all dimensions.
   * @param sigma_f @sigma_f value
   */
  explicit GaussianKernel(double sigma_l = 1.0, double sigma_f = 1.0);

  /** @getter{@sigma_f value, kernel} */
  [[nodiscard]] Scalar sigma_f() const { return sigma_f_; }
  /**
   * Get read-only access to the @sigma_l of the kernel.
   * If the kernel is isotropic, the size of the vector will be 1 upon construction
   * and will be adapted to match to the latest inputs
   * @return dimension of the kernel
   */
  [[nodiscard]] const Vector& sigma_l() const { return sigma_l_; }

  [[nodiscard]] bool is_stationary() const override { return true; }
  /** @checker{isotropic, kernel} */
  [[nodiscard]] bool is_isotropic() const { return is_isotropic_; }
  [[nodiscard]] std::unique_ptr<Kernel> clone() const override;

  void set(Parameter parameter, double value) override;
  void set(Parameter parameter, const Vector& value) override;

 private:
  Matrix operator()(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const override;
  [[nodiscard]] double get_d(Parameter parameter) const override;
  [[nodiscard]] const Vector& get_v(Parameter parameter) const override;

  mutable Vector sigma_l_;  ///< @sigma_l value
  Vector log_sigma_l_;      ///< @sigma_l value in log space. Used for optimization
  double sigma_f_;          ///< @sigma_f value
  bool is_isotropic_;       ///< True if the kernel is isotropic (i.e., @sigma_l is the same for all dimensions)
};

using RadialBasisFunction = GaussianKernel;       ///< Alias for Gaussian kernel.
using SquaredExponentialKernel = GaussianKernel;  ///< Alias for Gaussian kernel.

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GaussianKernel)

#endif
