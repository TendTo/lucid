/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include <ostream>

#include "lucid/math/Parameter.h"
#include "lucid/util/error.h"

namespace lucid {
std::ostream& operator<<(std::ostream& os, const Parameter name) {
  switch (name) {
    case Parameter::LENGTH_SCALE:
      return os << "Length scale";
    case Parameter::MEAN:
      return os << "Mean";
    default:
      LUCID_UNREACHABLE();
  }
}
}  // namespace lucid
