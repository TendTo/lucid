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

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel {
 public:
  explicit Kernel(const Dimension num_params = 0) : parameters_{num_params} {}
  explicit Kernel(Vector params) : parameters_{std::move(params)} {}
  virtual ~Kernel() = default;

  /** @getter{hyperparameters, kernel} */
  [[nodiscard]] const Vector& parameters() const { return parameters_; }

  /**
   * Apply the kernel function to two vectors.
   * @param x1 first vector
   * @param x2 second vector
   * @return kernel value
   */
  virtual Scalar operator()(const Vector& x1, const Vector& x2) const = 0;

  /**
   * Clone the kernel.
   * Create a new instance of the kernel with the same parameters.
   * @return new instance of the kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone() const = 0;
  /**
   * Clone the kernel with new parameters.
   * Create a new instance of the kernel with the given `params`.
   * @param params new parameters to use for the cloned kernel
   * @return new instance of the kernel
   */
  [[nodiscard]] virtual std::unique_ptr<Kernel> clone(const Vector& params) const = 0;

 protected:
  Vector parameters_;  ///< Kernel parameters
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
