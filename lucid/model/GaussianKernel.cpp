/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/GaussianKernel.h"

#include <cmath>
#include <memory>
#include <utility>

#include "lucid/util/error.h"

namespace lucid {

GaussianKernel::GaussianKernel(const Vector& sigma_l, const double sigma_f)
    : sigma_l_{sigma_l}, sigma_f_{sigma_f}, gamma_{sigma_l.size()} {
  LUCID_CHECK_ARGUMENT_EXPECTED(sigma_l.size() > 0, "sigma_l.size()", sigma_l.size(), "at least 1");
  gamma_ = -0.5 * sigma_l.cwiseProduct(sigma_l).cwiseInverse();
}
GaussianKernel::GaussianKernel(const Dimension dim, const double sigma_l, const double sigma_f)
    : GaussianKernel{Vector::Constant(dim, sigma_l), sigma_f} {}

Scalar GaussianKernel::operator()(const Vector& x1, const Vector& x2) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.size() == x2.size(), "x1.size() != x2.size()", x1.size(), x2.size());
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.size() == gamma_.size(), "x1.size() != sigma_l().size()", x1.size(), gamma_.size());
  const auto diff = x1 - x2;
  LUCID_ASSERT((diff.transpose() * gamma_.asDiagonal() * diff).size() == 1u, "scalar result");
  return sigma_f() * sigma_f() * std::exp((diff.transpose() * gamma_.asDiagonal() * diff).value());
}

std::unique_ptr<Kernel> GaussianKernel::clone() const { return std::make_unique<GaussianKernel>(sigma_l_, sigma_f_); }

void GaussianKernel::set(const Parameter parameter, const double value) {
  switch (parameter) {
    case Parameter::SIGMA_F:
      sigma_f_ = value;
      break;
    default:
      Kernel::set(parameter, value);
  }
}
void GaussianKernel::set(const Parameter parameter, const Vector& value) {
  switch (parameter) {
    case Parameter::SIGMA_L:
      sigma_l_ = value;
      gamma_ = -0.5 * sigma_l_.cwiseProduct(sigma_l_).cwiseInverse();
      break;
    default:
      Kernel::set(parameter, value);
  }
}

double GaussianKernel::get_d(const Parameter parameter) const {
  switch (parameter) {
    case Parameter::SIGMA_F:
      return sigma_f_;
    default:
      return Kernel::get_d(parameter);
  }
}

const Vector& GaussianKernel::get_v(const Parameter parameter) const {
  switch (parameter) {
    case Parameter::SIGMA_L:
      return sigma_l_;
    default:
      return Kernel::get_v(parameter);
  }
}

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel) {
  return os << "GaussianKernel( "
            << "sigma_f( " << kernel.sigma_f() << " ), "
            << "sigma_l( " << kernel.sigma_l().transpose() << " ) )";
}

}  // namespace lucid
