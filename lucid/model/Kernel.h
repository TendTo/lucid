/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Kernel class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"
#include "lucid/model/Parametrizable.h"

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel : public Parametrizable {
 public:
  Kernel() = default;

  /** @checker{is stationary, kernel} */
  [[nodiscard]] virtual bool is_stationary() const = 0;

  /**
   * Compute the kernel function on `x1` and `x2`.
   * @f[
   * K(x_1, x_2)
   * @f]
   * @param x1 first input vector
   * @param x2 second input vector
   * @return kernel value
   */
  Scalar operator()(const Vector& x1, const Vector& x2) const { return (*this)(x1, x2, nullptr).value(); }
  /**
   * Compute the kernel function on `x`.
   * @f[
   * K(x, x)
   * @f]
   * @param x input vector
   * @return kernel value
   */
  Scalar operator()(const Vector& x) const { return (*this)(x, x, nullptr).value(); }
  /**
   * Compute the kernel function on `x`.
   * @f[
   * K(x, x)
   * @f]
   * The gradient of the kernel function with respect to the parameters is computed and stored in `gradient`.
   * @param x input vector
   * @param[out] gradient pointer to store the gradient of the kernel function with respect to the parameters
   * @return kernel value
   */
  Scalar operator()(const Vector& x, double& gradient) const { return (*this)(x, x, &gradient).value(); }

  /**
   * Clone the kernel.
   * Create a new instance of the kernel with the same parameters.
   * @return new instance of the kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone() const = 0;

 protected:
  /**
   * Compute the kernel function on `x1` and `x2`.
   * @f[
   * K(x_1, x_2)
   * @f]
   * If `gradient` is not `nullptr`, the gradient of the kernel function with respect to the parameters
   * is computed and stored in `*gradient`.
   * @param x1 first input vector
   * @param x2 second input vector
   * @param[out] gradient pointer to store the gradient of the kernel function with respect to the parameters
   * @return kernel value
   */
  virtual Vector operator()(const Matrix& x1, const Matrix& x2, double* gradient) const = 0;
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
