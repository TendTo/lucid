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
#include "lucid/math/Parameter.h"
#include "lucid/math/Parametrizable.h"

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel : public Parametrizable {
 public:
  Kernel() = default;
  Kernel(const Kernel&) = default;
  Kernel(Kernel&&) = default;
  Kernel& operator=(const Kernel&) = default;
  Kernel& operator=(Kernel&&) = default;
  virtual ~Kernel() = default;

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
  virtual Scalar operator()(const Vector& x1, const Vector& x2) const = 0;
  /**
   * Compute the kernel function on `x`.
   * @f[
   * K(x, x)
   * @f]
   * @param x input vector
   * @return kernel value
   */
  Scalar operator()(const Vector& x) const { return (*this)(x, x); }

  /**
   * Clone the kernel.
   * Create a new instance of the kernel with the same parameters.
   * @return new instance of the kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone() const = 0;
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
