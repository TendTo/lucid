/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Kernel.h"

#include <ostream>
#include <string>
#include <vector>

#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"

namespace lucid {

Matrix Kernel::operator()(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const {
  LUCID_TRACE_FMT("({}, {}, {})", LUCID_FORMAT_MATRIX(x1), LUCID_FORMAT_MATRIX(x2), gradient != nullptr);
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().kernel_timer : nullptr};
  if (Stats::Scoped::top()) Stats::Scoped::top()->value().num_kernel_applications++;
  return apply_impl(x1, x2, gradient);
}

std::string Kernel::to_string() const { return "Kernel( )"; }

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) { return os << kernel.to_string(); }

}  // namespace lucid
