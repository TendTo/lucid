/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/KernelHyperParameter.h"

#include <ostream>

#include "lucid/util/error.h"

namespace lucid {
std::ostream& operator<<(std::ostream& os, const KernelHyperParameter name) {
  switch (name) {
    case KernelHyperParameter::LENGTH_SCALE:
      return os << "Length scale";
    case KernelHyperParameter::MEAN:
      return os << "Mean";
    default:
      LUCID_UNREACHABLE();
  }
}
}  // namespace lucid
