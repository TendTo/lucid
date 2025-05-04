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

template <IsAnyOf<int, double, const Vector&> T>
T Kernel::get_parameter(const KernelParameter parameter) const {
  if constexpr (std::is_same_v<T, int>) {
    return get_parameter_i(parameter);
  } else if constexpr (std::is_same_v<T, double>) {
    return get_parameter_d(parameter);
  } else if constexpr (std::is_same_v<T, const Vector&>) {
    return get_parameter_v(parameter);
  } else {
    LUCID_UNREACHABLE();
  }
}

void Kernel::set_parameter(KernelParameter parameter, int) { LUCID_INVALID_KERNEL_PARAMETER("kernel", parameter); }
void Kernel::set_parameter(KernelParameter parameter, double) { LUCID_INVALID_KERNEL_PARAMETER("kernel", parameter); }
void Kernel::set_parameter(KernelParameter parameter, Vector) { LUCID_INVALID_KERNEL_PARAMETER("kernel", parameter); }
int Kernel::get_parameter_i(KernelParameter parameter) const { LUCID_INVALID_KERNEL_PARAMETER("kernel", parameter); }
double Kernel::get_parameter_d(KernelParameter parameter) const { LUCID_INVALID_KERNEL_PARAMETER("kernel", parameter); }
const Vector& Kernel::get_parameter_v(KernelParameter parameter) const {
  LUCID_INVALID_KERNEL_PARAMETER("kernel", parameter);
}

std::ostream& operator<<(std::ostream& os, const Kernel& kernel) {
  if (dynamic_cast<const GaussianKernel*>(&kernel)) return os << static_cast<const GaussianKernel&>(kernel);
  return os << "Kernel()";
}

template int Kernel::get_parameter<int>(KernelParameter parameter) const;
template double Kernel::get_parameter<double>(KernelParameter parameter) const;
template const Vector& Kernel::get_parameter<const Vector&>(KernelParameter parameter) const;

}  // namespace lucid
