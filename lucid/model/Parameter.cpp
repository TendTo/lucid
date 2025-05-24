/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Parameter.h"

#include <ostream>

#include "lucid/util/error.h"

namespace lucid {
std::ostream& operator<<(std::ostream& os, const Parameter name) {
  switch (name) {
    case Parameter::SIGMA_L:
      return os << "Parameter( Sigma_l )";
    case Parameter::SIGMA_F:
      return os << "Parameter( Sigma_f )";
    case Parameter::REGULARIZATION_CONSTANT:
      return os << "Parameter( RegularizationConstant )";
    case Parameter::DEGREE:
      return os << "Parameter( Degree )";
    default:
      LUCID_UNREACHABLE();
  }
}
}  // namespace lucid
