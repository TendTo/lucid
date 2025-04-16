/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/GaussianKernel.h"

#include <cmath>
#include <utility>

#include "lucid/util/error.h"

namespace lucid {

GaussianKernel::GaussianKernel(Vector params) : Kernel{std::move(params)}, sigma_l_diagonal_{parameters_.size() - 1} {
  if (parameters_.size() < 2) LUCID_INVALID_ARGUMENT_EXPECTED("params.size()", parameters_.size(), "at least 2");
  const auto sigma_l_diagonal = parameters_.tail(parameters_.size() - 1);
  sigma_l_diagonal_ = sigma_l_diagonal.cwiseProduct(sigma_l_diagonal).cwiseInverse();
}
GaussianKernel::GaussianKernel(const double sigma_f, const Vector& sigma_l) : Kernel{sigma_l.size() + 1} {
  if (sigma_l.size() < 1) LUCID_INVALID_ARGUMENT_EXPECTED("sigma_l.size()", sigma_l.size(), "at least 1");
  parameters_(0) = sigma_f;
  parameters_.tail(sigma_l.size()) = sigma_l;
  sigma_l_diagonal_ = sigma_l.cwiseProduct(sigma_l).cwiseInverse();
}

Scalar GaussianKernel::operator()(const Vector& x1, const Vector& x2) const {
  if (x1.size() != x2.size()) LUCID_INVALID_ARGUMENT_EXPECTED("x1.size() != x2.size()", x1.size(), x2.size());
  if (x1.size() != sigma_l_diagonal_.size())
    LUCID_INVALID_ARGUMENT_EXPECTED("x.size() != sigma_l().size()", x1.size(), sigma_l_diagonal_.size());
  const auto diff = x1 - x2;
  return sigma_f() * sigma_f() * std::exp(-0.5 * (diff.transpose() * sigma_l_diagonal_.asDiagonal() * diff).value());
}
std::unique_ptr<Kernel> GaussianKernel::clone() const { return std::make_unique<GaussianKernel>(parameters_); }
std::unique_ptr<Kernel> GaussianKernel::clone(const Vector& params) const {
  return std::make_unique<GaussianKernel>(params);
}

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel) {
  return os << "GaussianKernel( "
            << "Sigma_f( " << kernel.sigma_f()  << " ), "
            << "Sigma_l( " << kernel.sigma_l().transpose() << " ) )";
}

}  // namespace lucid
