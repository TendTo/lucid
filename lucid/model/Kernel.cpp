/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Kernel.h"

#include <ostream>
#include <vector>

#include "lucid/model/GaussianKernel.h"
#include "lucid/model/ValleePoussinKernel.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"

namespace lucid {

Matrix Kernel::operator()(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* gradient) const {
  LUCID_TRACE_FMT("({}, {}, {})", LUCID_FORMAT_MATRIX(x1), LUCID_FORMAT_MATRIX(x2), gradient != nullptr);
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().kernel_timer : nullptr};
  if (Stats::Scoped::top()) Stats::Scoped::top()->value().num_kernel_applications++;
  return apply_impl(x1, x2, gradient);
}

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  if (const auto* casted = dynamic_cast<const GaussianKernel*>(&kernel)) return os << *casted;
  if (const auto* casted = dynamic_cast<const ValleePoussinKernel*>(&kernel)) return os << *casted;
  return os << "Kernel( )";
}

}  // namespace lucid
