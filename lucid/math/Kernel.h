/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Kernel class.
 */
#pragma once

#include <iosfwd>

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Represents a kernel function.
 */
class Kernel {
 public:
  virtual ~Kernel() = default;

  /**
   * Apply the kernel function to two vectors.
   * @param x1 first vector
   * @param x2 second vector
   * @return kernel value
   */
  virtual Scalar operator()(const Vector& x1, const Vector& x2) const = 0;
};

std::ostream& operator<<(std::ostream& os, const Kernel& kernel);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Kernel)

#endif
