/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/Kernel.h"

#include <ostream>

namespace lucid {
std::ostream& operator<<(std::ostream& os, const Kernel&) { return os << "Kernel"; }
}  // namespace lucid
