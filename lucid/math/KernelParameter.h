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

enum class KernelParameter {
  // Mean parameter
  MEAN = 0,     ///< Mean parameter (@see KernelParameter::SIGMA_F)
  SIGMA_F = 0,  ///< Sigma_f parameter (@see KernelParameter::MEAN)
  // Length scale parameter
  LENGTH_SCALE = 1,  ///< Length scale parameter (@see KernelParameter::SIGMA_L, @see KernelParameter::COVARIANCE)
  COVARIANCE = 1,    ///< Covariance parameter (@see KernelParameter::SIGMA_L, @see KernelParameter::LENGTH_SCALE)
  SIGMA_L = 1,       ///< Sigma_l parameter (@see KernelParameter::LENGTH_SCALE, @see KernelParameter::COVARIANCE)
};

using KHP = KernelParameter;  ///< Alias for KernelHyperParameter

std::ostream& operator<<(std::ostream& os, KernelParameter name);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::KernelParameter)

#endif
