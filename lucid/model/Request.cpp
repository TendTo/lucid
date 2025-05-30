/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Request.h"

#include "lucid/util/error.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const Request& request) {
  switch (request) {
    case Request::_:
      return os << "Request( NoRequest )";
    case Request::OBJECTIVE_VALUE:
      return os << "Request( ObjectiveValue )";
    case Request::GRADIENT:
      return os << "Request( Gradient )";
    default:
      LUCID_UNREACHABLE();
  }
}

}  // namespace lucid