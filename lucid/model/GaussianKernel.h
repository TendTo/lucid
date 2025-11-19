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
 * The type of @sigmal on kernel construction determines whether the kernel is isotropic or anisotropic.
 * The kernel can be isotropic or anisotropic.
 * In the anisotropic case, each dimension has its own length scale (i.e., @sigmal value).
 * In the isotropic case, the same length scale is used for all dimensions.
 * If requested, the kernel can also compute the gradient of the kernel matrix with respect to @sigmal and @sigmaf.
 */
class GaussianKernel final : public Kernel {
 public:
  using Kernel::set;
  using Kernel::operator();
  /**
   * Construct a new anisotropic GaussianKernel object with the given parameters.
   * @pre `sigma_l` must contain at least one element all elements must be greater than 0
   * @note Even if all the elements of `sigma_l` are equal, the kernel will be anisotropic.
   * This may play a role during hyperparameter optimization.
   * @param sigma_l @sigmal value
   * @param sigma_f @sigmaf value
   */
  explicit GaussianKernel(Vector sigma_l, double sigma_f = 1.0);
  /**
   * Construct a new isotropic GaussianKernel object with the given parameters.
   * @pre `sigma_l` must be greater than 0
   * @param sigma_l @sigmal value. It is equal for all dimensions.
   * @param sigma_f @sigmaf value
   */
  explicit GaussianKernel(double sigma_l = 1.0, double sigma_f = 1.0);

  /** @getter{@sigmaf value, kernel} */
  [[nodiscard]] Scalar sigma_f() const { return sigma_f_; }
  /**
   * Get read-only access to the @sigmal of the kernel.
   * If the kernel is isotropic, the size of the vector will be 1, regardless of the input dimension.
   * @return dimension of the kernel
   */
  [[nodiscard]] const Vector& sigma_l() const { return sigma_l_; }

  [[nodiscard]] bool is_stationary() const override { return true; }
  /** @checker{isotropic, kernel} */
  [[nodiscard]] bool is_isotropic() const { return is_isotropic_; }
  [[nodiscard]] std::unique_ptr<Kernel> clone() const override;

  void set(Parameter parameter, double value) override;
  void set(Parameter parameter, const Vector& value) override;

  [[nodiscard]] std::string to_string() const override;

 private:
  Matrix apply_impl(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const override;
  [[nodiscard]] double get_d(Parameter parameter) const override;
  [[nodiscard]] const Vector& get_v(Parameter parameter) const override;

  mutable Vector sigma_l_;  ///< @sigmal value
  Vector log_parameters_;   ///< [ @sigmaf @sigmal ] value in log space. Used for optimization
  double sigma_f_;          ///< @sigmaf value
  bool is_isotropic_;       ///< True if the kernel is isotropic (i.e., @sigmal is the same for all dimensions)
};

using RadialBasisFunction = GaussianKernel;       ///< Alias for Gaussian kernel.
using SquaredExponentialKernel = GaussianKernel;  ///< Alias for Gaussian kernel.

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GaussianKernel)

#endif
