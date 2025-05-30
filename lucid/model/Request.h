/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Request enum.
 */
#pragma once

#include <cstdint>
#include <iosfwd>

namespace lucid {

/**
 * List of available requests a tuner can make to an Estimator and, by extension, to a Kernel.
 * If the request is not supported by the Estimator, nothing will happen.
 * @note The values are offset in such a way that operating over them is very efficient.
 */
enum class Request : std::uint16_t {
  _ = 0,                     ///< No requests
  OBJECTIVE_VALUE = 1 << 0,  ///< Compute the objective value of the Estimator
  GRADIENT = 1 << 1,         ///< Compute the gradient of the objective value with respect to the parameters
};

std::ostream& operator<<(std::ostream& os, const Request& request);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Request);

#endif
