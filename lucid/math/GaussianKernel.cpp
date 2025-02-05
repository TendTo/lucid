/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/GaussianKernel.h"

#include <cmath>

#include "lucid/util/error.h"

namespace lucid {

Scalar GaussianKernel::operator()(const Vector& x1, const Vector& x2) const {
  if (x1.size() != x2.size()) LUCID_INVALID_ARGUMENT_EXPECTED("x1.size()", x1.size(), x2.size());
  if (x1.size() != sigma_l_.size()) LUCID_INVALID_ARGUMENT_EXPECTED("x.size()", x1.size(), sigma_l_.size());
  return sigma_f_ * sigma_f_ * std::exp(-0.5 * ((x1 - x2).transpose() * sigma_l_.asDiagonal() * (x1 - x2)).value());
}
std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel) {
  return os << "GaussianKernel\n"
            << "Sigma_f: " << kernel.sigma_f() << "\n"
            << "Sigma_l: \n[" << kernel.sigma_l() << "]";
}

}  // namespace lucid
