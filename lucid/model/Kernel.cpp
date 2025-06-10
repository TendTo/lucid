/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Kernel.h"

#include <ostream>

#include "lucid/model/GaussianKernel.h"
#include "lucid/util/error.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  if (const auto* casted = dynamic_cast<const GaussianKernel*>(&kernel)) return os << *casted;
  return os << "Kernel( )";
}

}  // namespace lucid
