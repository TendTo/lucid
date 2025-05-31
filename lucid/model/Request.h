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
#include <vector>

#include "lucid/util/concept.h"
#include "lucid/util/definitions.h"

namespace lucid {

/**
 * List of available requests a tuner can make to an Estimator and, by extension, to a Kernel.
 * If the request is not supported by the Estimator, nothing will happen.
 * This enum behaves as a bitset.
 * It is possible to combine multiple requests using bitwise OR operations
 * or to check if a request is set using the AND operations.
 * @code
 * Request::_ // Empty set {}
 * Requests u = Request::GRADIENT | Request::STATS  // {GRADIENT} U {STATS} = {STATS, GRADIENT}
 * u | Request::STATS  // {STATS, GRADIENT} U {DEGREE} = {STATS, GRADIENT}
 * u & Request::STATS  // Set intersection {STATS, GRADIENT} ∩ {STATS} = {STATS}
 * u && Request::STATS  // Check if {STATS, GRADIENT} ∩ {STATS} = {STATS} is non-empty
 * u || Request::STATS  // Check if {STATS, GRADIENT} ∪ {STATS} = {STATS, GRADIENT} is non-empty
 * @endcode
 * @note The values are offset in such a way that operating over them is very efficient.
 */
enum class Request : std::uint16_t {
  _ = 0,                     ///< No requests. Used as the empty set placeholder.
  OBJECTIVE_VALUE = 1 << 0,  ///< Compute the objective value of the Estimator
  GRADIENT = 1 << 1,         ///< Compute the gradient of the objective value with respect to the Requests
  STATS = 1 << 2,            ///< Compute the statistics of the Estimator
};

using Requests = std::underlying_type_t<Request>;                   ///< Efficient set of requests
constexpr Requests NoRequests = static_cast<Requests>(Request::_);  ///< No request value

LUCID_FLAG_ENUMS(Request, Requests, GRADIENT)

std::ostream& operator<<(std::ostream& os, const Request& request);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Request);

#endif
