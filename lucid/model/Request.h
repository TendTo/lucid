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

#include "lucid/util/concept.h"
#include "lucid/util/definitions.h"

namespace lucid {

/**
 * List of available requests a tuner can make to an Estimator and, by extension, to a Kernel.
 * If the request is not supported by the Estimator, nothing will happen.
 * @note The values are offset in such a way that operating over them is very efficient.
 */
enum class Request : std::uint16_t {
  _ = 0,                     ///< No requests
  OBJECTIVE_VALUE = 1 << 0,  ///< Compute the objective value of the Estimator
  GRADIENT = 1 << 1,         ///< Compute the gradient of the objective value with respect to the Requests
};

using Requests = std::underlying_type_t<Request>;                   ///< Efficient set of requests
constexpr Requests NoRequests = static_cast<Requests>(Request::_);  ///< No request value

FLAG_ENUMS(Request, GRADIENT)

/**
 * Perform a bitwise OR operation on two Requests.
 * Efficient way of taking the union of two set of Requests.
 * @tparam LP left Request type, must be one of the `Request` enum values or a set of requests
 * @tparam RP right Request type, must be one of the `Request` enum values or a set of requests
 * @param lhs left-hand side Request
 * @param rhs right-hand side Request
 * @return the result of the bitwise OR operation as a `Requests`
 */
template <IsAnyOf<Request, Requests> LP, IsAnyOf<Request, Requests> RP>
constexpr Requests operator|(LP lhs, RP rhs) {
  return static_cast<Request>(static_cast<Requests>(lhs) | static_cast<Requests>(rhs));
}
/**
 * Perform a bitwise OR operation on two Requests and return the result as a boolean.
 * Efficient way of checking if two set of Requests have a non-empty union.
 * @code
 * Request::GRADIENT || Request::GRADIENT; // true
 * Request::OBJECTIVE_VALUE || Request::GRADIENT; // true
 * Request::GRADIENT || (Request::GRADIENT & Request::OBJECTIVE_VALUE); // true
 * Request::_ || Request::GRADIENT; // true
 * Request::_ || Request::_; // false
 * @endcode
 * @tparam LP left Request type, must be one of the `Request` enum values or a set of requests
 * @tparam RP right Request type, must be one of the `Request` enum values or a set of requests
 * @param lhs left-hand side Request
 * @param rhs right-hand side Request
 * @return the result of the bitwise OR operation as a boolean
 */
template <IsAnyOf<Request, Requests> LP, IsAnyOf<Request, Requests> RP>
constexpr bool operator||(LP lhs, RP rhs) {
  return static_cast<Requests>(lhs) | static_cast<Requests>(rhs);
}
/**
 * Perform a bitwise AND operation on two Requests.
 * Efficient way of taking the intersection of two set of Requests.
 * @tparam LP left Request type, must be one of the `Request` enum values or a set of requests
 * @tparam RP right Request type, must be one of the `Request` enum values or a set of requests
 * @param lhs left-hand side Request
 * @param rhs right-hand side Request
 * @return the result of the bitwise AND operation as a `Request`
 */
template <IsAnyOf<Request, Requests> LP, IsAnyOf<Request, Requests> RP>
constexpr Requests operator&(LP lhs, RP rhs) {
  return static_cast<Request>(static_cast<Requests>(lhs) & static_cast<Requests>(rhs));
}
/**
 * Perform a bitwise AND operation on two Requests and return the result as a boolean.
 * Efficient way of checking if two set of Requests have a non-empty intersection.
 * @code
 * Request::GRADIENT && Request::OBJECTIVE_VALUE; // true
 * Request::OBJECTIVE_VALUE && Request::GRADIENT; // false
 * Request::GRADIENT && (Request::GRADIENT & Request::OBJECTIVE_VALUE); // true
 * Request::_ && Request::GRADIENT; // false
 * Request::_ && Request::_; // false
 * @endcode
 * @tparam LP left Request type, must be one of the `Request` enum values or a set of requests
 * @tparam RP right Request type, must be one of the `Request` enum values or a set of requests
 * @param lhs left-hand side Request
 * @param rhs right-hand side Request
 * @return the result of the bitwise AND operation as a boolean
 */
template <IsAnyOf<Request, Requests> LP, IsAnyOf<Request, Requests> RP>
constexpr bool operator&&(LP lhs, RP rhs) {
  return static_cast<Requests>(lhs) & static_cast<Requests>(rhs);
}

std::ostream& operator<<(std::ostream& os, const Request& request);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Request);

#endif
