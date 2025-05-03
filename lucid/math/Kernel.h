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

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel {
 public:
  explicit Kernel(const Dimension num_params = 0) : parameters_{num_params} {}
  explicit Kernel(Vector params) : parameters_{std::move(params)} {}
  Kernel(const Kernel&) = default;
  Kernel(Kernel&&) = default;
  Kernel& operator=(const Kernel&) = default;
  Kernel& operator=(Kernel&&) = default;
  virtual ~Kernel() = default;

  /** @getter{hyperparameters, kernel} */
  [[nodiscard]] const Vector& parameters() const { return parameters_; }

  /**
   * Compute the kernel function to two vectors.
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

 protected:
  Vector parameters_;  ///< Kernel parameters
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
