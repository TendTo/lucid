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
  LUCID_ASSERT(gamma_.size() == sigma_l.size(), "gamma_.size() != sigma_l.size()");
}
GaussianKernel::GaussianKernel(const Dimension dim, const double sigma_l, const double sigma_f)
    : GaussianKernel{Vector::Constant(dim, sigma_l), sigma_f} {}

Matrix GaussianKernel::operator()(ConstMatrixRef x1, ConstMatrixRef x2, double* gradient) const {
  LUCID_ASSERT(&x1 == &x2 || !gradient, "The gradient can be computed only for the same vector");
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.cols() == x2.cols(), "x1.cols() != x2.cols()", x1.cols(), x2.cols());
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.cols() == sigma_l_.size(), "x1.cols() != sigma_l().size()", x1.cols(),
                                sigma_l_.size());
  // TODO(tend): sklearn computes the kernel a bit differently. The result is the same, but which is more efficient?
  const Matrix dist{&x1 == &x2 ? static_cast<Matrix>(pdist<2, true>(x1.cwiseQuotient(sigma_l_)))  // Same vector
                               : pdist<2, true>((x1.array().rowwise() / sigma_l_.array()).matrix(),
                                                (x2.array().rowwise() / sigma_l_.array()).matrix())};
  const Matrix k{sigma_f() * sigma_f() * (-0.5 * dist.array()).exp()};
  LUCID_ASSERT(x1.rows() > 1 || x2.rows() > 1 || k.size() == 1,
               "If the comparison is between two vectors, the kernel should return a scalar");
  LUCID_TRACE_FMT("GaussianKernel::operator()({}, {}) = {}", x1, x2, k);
  if (gradient) {
    // Compute the gradient of the kernel function with respect to the parameters
    *gradient = 1.0;
  }
  return k;
}

bool GaussianKernel::is_isotropic() const {
  std::span view{sigma_l_.data(), static_cast<std::size_t>(sigma_l_.size())};
  return std::ranges::adjacent_find(view, std::ranges::not_equal_to()) == view.end();
}
std::unique_ptr<Kernel> GaussianKernel::clone() const { return std::make_unique<GaussianKernel>(sigma_l_, sigma_f_); }

bool GaussianKernel::has(const Parameter parameter) const {
  switch (parameter) {
    case Parameter::SIGMA_L:
    case Parameter::SIGMA_F:
      return true;
    default:
      return false;
  }
}

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
            << "sigma_l( " << kernel.sigma_l() << " ) "
            << "sigma_f( " << kernel.sigma_f() << " ) )";
}

}  // namespace lucid
