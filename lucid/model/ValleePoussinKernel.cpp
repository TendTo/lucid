/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * VallePoussinKernel class.
 */
#include "lucid/model/ValleePoussinKernel.h"

#include <ostream>

#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

ValleePoussinKernel::ValleePoussinKernel(const double a, const double b)
    : Kernel{Parameter::A | Parameter::B}, a_{a}, b_{b} {}

std::unique_ptr<Kernel> ValleePoussinKernel::clone() const { return std::make_unique<ValleePoussinKernel>(*this); }

double ValleePoussinKernel::get_d(const Parameter parameter) const {
  switch (parameter) {
    case Parameter::A:
      return a_;
    case Parameter::B:
      return b_;
    default:
      return Kernel::get_d(parameter);
  }
}

void ValleePoussinKernel::set(const Parameter parameter, const double value) {
  switch (parameter) {
    case Parameter::A:
      a_ = value;
      break;
    case Parameter::B:
      b_ = value;
      break;
    default:
      Kernel::set(parameter, value);
  }
}

Matrix ValleePoussinKernel::apply_impl(ConstMatrixRef x1, [[maybe_unused]] ConstMatrixRef x2,
                                       [[maybe_unused]] std::vector<Matrix>* const gradient) const {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(x1));
  LUCID_ASSERT(&x1 == &x2, "The kernel can be computed only with a single vector");
  LUCID_ASSERT(gradient == nullptr, "The kernel can be computed only with a single vector");
  LUCID_CHECK_ARGUMENT_CMP(x1.cols(), >, 0);
  LUCID_CHECK_ARGUMENT(gradient == nullptr, "gradient", "not-null");

  // Compute the Vallee-Poussin kernel
  // coeff = 1 / (b - a)^n
  const double coeff = 1.0 / std::pow(b_ - a_, static_cast<double>(x1.cols()));
  Matrix prod = Matrix::Ones(x1.rows(), 1);
  // For each dimension
  for (auto& col : x1.colwise()) {
    // num = sin( (b + a) / 2 * x_i ) * sin( (b - a) / 2 * x_i )
    const auto num = ((b_ + a_) / 2.0 * col.array()).sin() * ((b_ - a_) / 2.0 * col.array()).sin();
    // den = sin^2( x_i / 2 )
    const auto den = (col.array() / 2.0).sin().square();
    // fraction = num / den
    const auto fraction = (den != 0).select(num / den, b_ * b_ - a_ * a_);
    // Accumulate the product in prod
    prod.array() *= fraction;
  }
  return coeff * prod;
}

std::string ValleePoussinKernel::to_string() const {
  return fmt::format("ValleePoussinKernel( a( {} ) b( {} ) )", a_, b_);
}

std::ostream& operator<<(std::ostream& os, const ValleePoussinKernel& kernel) { return os << kernel.to_string(); }

}  // namespace lucid
