/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Parameter.h"

#include <ostream>

#include "lucid/util/error.h"

namespace lucid {
std::ostream& operator<<(std::ostream& os, const Parameter name) {
  switch (name) {
    case Parameter::LENGTH_SCALE:
      return os << "Length scale";
    case Parameter::MEAN:
      return os << "Mean";
    case Parameter::REGULARIZATION_CONSTANT:
      return os << "Regularization constant";
    default:
      LUCID_UNREACHABLE();
  }
}
}  // namespace lucid
