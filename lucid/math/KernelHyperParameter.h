/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * KernelParameter enum.
 */
#pragma once

#include <iosfwd>

namespace lucid {

enum class KernelHyperParameter {
  LENGTH_SCALE = 0,  ///< Length scale parameter
  VARIANCE = 1,      ///< Variance parameter
};

using KHP = KernelHyperParameter;  ///< Alias for KernelHyperParameter

std::ostream& operator<<(std::ostream& os, KernelHyperParameter name);

}  // namespace lucid
