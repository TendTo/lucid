/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/KernelParameter.h"

#include <ostream>

#include "lucid/util/error.h"

namespace lucid {
std::ostream& operator<<(std::ostream& os, const KernelParameter name) {
  switch (name) {
    case KernelParameter::LENGTH_SCALE:
      return os << "Length scale";
    case KernelParameter::MEAN:
      return os << "Mean";
    default:
      LUCID_UNREACHABLE();
  }
}
}  // namespace lucid
