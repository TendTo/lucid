/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Kernel.h"

#include <ostream>

#include "lucid/math/GaussianKernel.h"
#include "lucid/util/error.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  if (dynamic_cast<const GaussianKernel*>(&kernel)) return os << static_cast<const GaussianKernel&>(kernel);
  return os << "Kernel()";
}

}  // namespace lucid
