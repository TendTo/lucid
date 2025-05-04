/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/GaussianKernel.h"

#include <cmath>
#include <memory>
#include <utility>

#include "lucid/util/error.h"

namespace lucid {

GaussianKernel::GaussianKernel(Vector params) : Kernel{std::move(params)}, sigma_l_sq_diagonal_inv_{} {
  LUCID_CHECK_ARGUMENT_EXPECTED(parameters_.size() > 1, "params.size()", parameters_.size(), "at least 2");
  const auto sigma_l_diagonal = parameters_.tail(parameters_.size() - 1);
  sigma_l_sq_diagonal_inv_ = sigma_l_diagonal.cwiseProduct(sigma_l_diagonal).cwiseInverse();
}
GaussianKernel::GaussianKernel(const double sigma_f, const Vector& sigma_l)
    : Kernel{sigma_l.size() + 1}, sigma_l_sq_diagonal_inv_{} {
  LUCID_CHECK_ARGUMENT_EXPECTED(sigma_l.size() > 0, "sigma_l.size()", sigma_l.size(), "at least 1");
  parameters_(0) = sigma_f;
  parameters_.tail(sigma_l.size()) = sigma_l;
  sigma_l_sq_diagonal_inv_ = sigma_l.cwiseProduct(sigma_l).cwiseInverse();
}
GaussianKernel::GaussianKernel(const double sigma_f, const double sigma_l, Dimension size)
    : Kernel{size + 1}, sigma_l_sq_diagonal_inv_{} {
  LUCID_CHECK_ARGUMENT_EXPECTED(size > 0, "size", size, "at least 1");
  parameters_(0) = sigma_f;
  parameters_.tail(size) = Vector::Constant(size, sigma_l);
  sigma_l_sq_diagonal_inv_ = Vector::Constant(size, sigma_l * sigma_l).cwiseInverse();
}
GaussianKernel::GaussianKernel(const double sigma_f, const double sigma_l, const Dimension size)
    : GaussianKernel{sigma_f, Vector::Constant(size, sigma_l)} {}

Scalar GaussianKernel::operator()(const Vector& x1, const Vector& x2) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.size() == x2.size(), "x1.size() != x2.size()", x1.size(), x2.size());
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.size() == sigma_l_sq_diagonal_inv_.size(), "x1.size() != sigma_l().size()",
                                x1.size(), sigma_l_sq_diagonal_inv_.size());
  const auto diff = x1 - x2;
  LUCID_ASSERT((diff.transpose() * sigma_l_sq_diagonal_inv_.asDiagonal() * diff).size() == 1u, "scalar result");
  return sigma_f() * sigma_f() *
         std::exp(-0.5 * (diff.transpose() * sigma_l_sq_diagonal_inv_.asDiagonal() * diff).value());
}
Matrix GaussianKernel::apply(const Matrix& x1, const Matrix& x2) const {
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.cols() == x2.cols(), "x1.cols() != x2.cols()", x1.cols(), x2.cols());
  LUCID_CHECK_ARGUMENT_EXPECTED(x1.cols() == sigma_l_diagonal_.size(), "x1.cols() != sigma_l().size()", x1.cols(),
                                sigma_l_diagonal_.size());
  const auto diff = x1 - x2;
  return sigma_f_ * sigma_f_ * (-0.5 * (diff * sigma_l_diagonal_.asDiagonal() * diff.transpose())).array().exp();
}

std::unique_ptr<Kernel> GaussianKernel::clone() const { return std::make_unique<GaussianKernel>(sigma_f_, sigma_l_); }

double GaussianKernel::get_parameter_d(const KernelParameter parameter) const {
  switch (parameter) {
    case KernelParameter::SIGMA_F:
      return sigma_f_;
    default:
      return Kernel::get_parameter_d(parameter);
  }
}

const Vector& GaussianKernel::get_parameter_v(const KernelParameter parameter) const {
  switch (parameter) {
    case KernelParameter::SIGMA_L:
      return sigma_l_;
    default:
      return Kernel::get_parameter_v(parameter);
  }
}

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel) {
  return os << "GaussianKernel( "
            << "sigma_f( " << kernel.sigma_f() << " ), "
            << "sigma_l( " << kernel.sigma_l().transpose() << " ) )";
}

}  // namespace lucid
