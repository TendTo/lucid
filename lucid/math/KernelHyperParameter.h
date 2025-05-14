/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * KernelHyperParameter enum.
 */
#pragma once

#include <iosfwd>

namespace lucid {

enum class KernelHyperParameter {
  // Mean parameter
  MEAN = 0,     ///< Mean parameter (@see KernelHyperParameter::SIGMA_F)
  SIGMA_F = 0,  ///< Sigma_f parameter (@see KernelHyperParameter::MEAN)
  // Length scale parameter
  LENGTH_SCALE = 1,  ///< Length scale parameter (@see KernelHyperParameter::SIGMA_L, @see KernelHyperParameter::COVARIANCE)
  COVARIANCE = 1,    ///< Covariance parameter (@see KernelHyperParameter::SIGMA_L, @see KernelHyperParameter::LENGTH_SCALE)
  SIGMA_L = 1,       ///< Sigma_l parameter (@see KernelHyperParameter::LENGTH_SCALE, @see KernelHyperParameter::COVARIANCE)
};

using KHP = KernelHyperParameter;  ///< Alias for KernelHyperParameter

std::ostream& operator<<(std::ostream& os, KernelHyperParameter name);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::KernelHyperParameter)

#endif
