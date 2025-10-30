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
#include <vector>

#include "lucid/util/error.h"

namespace lucid {

GaussianKernel::GaussianKernel(Vector sigma_l, const double sigma_f)
    : Kernel{Parameter::SIGMA_F | Parameter::SIGMA_L | Parameter::GRADIENT_OPTIMIZABLE},
      sigma_l_{std::move(sigma_l)},
      log_parameters_{sigma_l_.size() + 1},
      sigma_f_{sigma_f},
      is_isotropic_{false} {
  LUCID_TRACE_FMT("({}, {})", sigma_l_, sigma_f);
  LUCID_CHECK_ARGUMENT_CMP(sigma_l_.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(sigma_l_.minCoeff(), >, 0);
  log_parameters_ << std::log(sigma_f_), sigma_l_.array().log().matrix();
}
GaussianKernel::GaussianKernel(const double sigma_l, const double sigma_f)
    : GaussianKernel{Vector::Constant(1, sigma_l), sigma_f} {
  is_isotropic_ = true;
}

Matrix GaussianKernel::apply_impl(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix>* const gradient) const {
  LUCID_TRACE_FMT("({}, {}, gradient)", LUCID_FORMAT_MATRIX(x1), LUCID_FORMAT_MATRIX(x2));
  LUCID_ASSERT(&x1 == &x2 || !gradient, "The gradient can be computed only over the same vector");
  LUCID_ASSERT(sigma_l_.size() > 0, "sigma_l must have at least one element");
  LUCID_CHECK_ARGUMENT_CMP(x1.cols(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(x2.cols(), >, 0);
  LUCID_CHECK_ARGUMENT_EQ(x1.cols(), x2.cols());

  if (is_isotropic_ && sigma_l_.size() != x1.cols()) sigma_l_ = Vector::Constant(x1.cols(), sigma_l_.head<1>().value());
  LUCID_CHECK_ARGUMENT_EQ(x1.cols(), sigma_l_.size());

  const bool is_same_input = &x1 == &x2;
  // TODO(tend): for efficiency, we should implement a method similar to `squareform` so that we only compute a vector
  //  when we are working on a single input
  // Compute the gaussian kernel: σf^2 * exp(-0.5 * ||(x1 / σl) - (x2 / σl)||^2)
  Matrix dist{is_same_input ? pdist<2, true, true>((x1.array().rowwise() / sigma_l_.array()).matrix())
                            : pdist<2, true>((x1.array().rowwise() / sigma_l_.array()).matrix(),
                                             (x2.array().rowwise() / sigma_l_.array()).matrix())};
  Matrix k{sigma_f_ * sigma_f_ * (-0.5 * dist.array()).exp()};
  LUCID_ASSERT(k.size() == x1.rows() * x2.rows(), "The shape of the output matrix should be equal to n1 x n2");

  if (gradient) {  // If we have been provided a gradient vector, let's compute the gradients
    gradient->clear();
    gradient->reserve(sigma_l_.size() + 1);  // +1 for sigma_f

    // Gradient with respect to log sigma_f: ∂k/∂(log σf) = 2 * σf^2 * exp(-0.5 * ||(x1 / σl) - (x2 / σl)||^2) = 2 * k
    gradient->emplace_back(2.0 * k.array());

    // If sigma_l is isotropic (i.e., equal for all dimensions), just compute a single gradient w.r.t. log sigma_l
    if (is_isotropic_) {
      gradient->emplace_back(k.cwiseProduct(dist));
    } else {
      // otherwise, compute a separate gradient matrix for each dimension
      const Vector sigma_l_sq{sigma_l_.array().pow(2)};
      for (Index i = 0; i < sigma_l_.size(); ++i) {
        auto c = x1.col(i);
        Matrix temp = (c.replicate(1, c.rows()).rowwise() - c.transpose()).array().pow(2).matrix();
        gradient->emplace_back((temp.array() / sigma_l_sq(i)).matrix().cwiseProduct(k));
      }
    }
    LUCID_TRACE_FMT("gradient = {}", *gradient);
  }
  LUCID_TRACE_FMT("=> {}", LUCID_FORMAT_MATRIX(k));
  return k;
}

std::unique_ptr<Kernel> GaussianKernel::clone() const {
  LUCID_TRACE("Cloning");
  return is_isotropic_ ? std::make_unique<GaussianKernel>(sigma_l_.head<1>().value(), sigma_f_)
                       : std::make_unique<GaussianKernel>(sigma_l_, sigma_f_);
}

void GaussianKernel::set(const Parameter parameter, const double value) {
  switch (parameter) {
    case Parameter::SIGMA_F:
      sigma_f_ = value;
      log_parameters_(0) = std::log(sigma_f_);
      break;
    default:
      Kernel::set(parameter, value);
  }
}
void GaussianKernel::set(const Parameter parameter, const Vector& value) {
  switch (parameter) {
    case Parameter::SIGMA_L:
      LUCID_CHECK_ARGUMENT(is_isotropic_ || value.size() == sigma_l_.size(), "value.size", sigma_l_.size());
      LUCID_CHECK_ARGUMENT(!is_isotropic_ || (value.array() == value(0)).all(), "value components", value(0));
      sigma_l_ = is_isotropic_ ? value.head<1>() : value;
      log_parameters_.tail(sigma_l_.size()) = sigma_l_.array().log().matrix();
      break;
    case Parameter::GRADIENT_OPTIMIZABLE:
      LUCID_CHECK_ARGUMENT_EQ(value.size(), log_parameters_.size());
      LUCID_CHECK_ARGUMENT(!is_isotropic_ || (value.tail(log_parameters_.size() - 1).array() == value(1)).all(),
                           "value[1:] components", value(1));
      log_parameters_ = value;
      sigma_l_ = log_parameters_.tail(log_parameters_.size() - 1).array().exp().matrix();
      sigma_f_ = std::exp(log_parameters_(0));
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
    case Parameter::GRADIENT_OPTIMIZABLE:
      return log_parameters_;
    default:
      return Kernel::get_v(parameter);
  }
}

std::ostream& operator<<(std::ostream& os, const GaussianKernel& kernel) {
  return os << "GaussianKernel( "
            << "sigma_l( " << (kernel.is_isotropic() ? Vector{kernel.sigma_l().head<1>()} : kernel.sigma_l()) << " ) "
            << "sigma_f( " << kernel.sigma_f() << " ) "
            << "isotropic( " << kernel.is_isotropic() << " ) )";
}

}  // namespace lucid
