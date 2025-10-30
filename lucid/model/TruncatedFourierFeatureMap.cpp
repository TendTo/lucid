/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/TruncatedFourierFeatureMap.h"

#include <memory>
#include <numbers>
#include <utility>

#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"

namespace lucid {

TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(const int num_frequencies, const Matrix& prob_per_dim,
                                                       const Matrix& omega_per_dim, const Scalar sigma_f,
                                                       const RectSet& x_limits)
    : num_frequencies_per_dimension_{num_frequencies},
      omega_{combvec(omega_per_dim).transpose()},
      weights_{::lucid::pow(num_frequencies, x_limits.dimension()) * 2 - 1},
      sigma_f_{sigma_f},
      x_limits_{x_limits},
      periodic_coefficients_{2 * std::numbers::pi / omega_per_dim.col(1).array()} {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >=, 2);
  LUCID_CHECK_ARGUMENT_CMP(sigma_f, >, 0);
  LUCID_CHECK_ARGUMENT_EQ(prob_per_dim.rows(), x_limits.dimension());
  LUCID_ASSERT((omega_.array() >= 0).all(), "single_weights >= 0");
  LUCID_ASSERT(static_cast<std::size_t>(omega_.rows()) == ::lucid::pow(num_frequencies, x_limits.dimension()),
               "omega_.rows() == num_frequencies^dimension");
  LUCID_ASSERT(omega_.cols() == x_limits.dimension(), "omega_.cols() == dimension");

  const Vector prod{combvec(prob_per_dim).colwise().prod()};
  if (captured_probability_ = prod.sum(); captured_probability_ > 0.94)
    LUCID_DEBUG_FMT("Probability captured by Fourier expansion is {:.3f} percent", captured_probability_);
  else
    LUCID_WARN_FMT("Probability captured by Fourier expansion is only {:.3f} percent", captured_probability_);

  const Vector single_weights{prod.cwiseSqrt()};
  LUCID_ASSERT((single_weights.array() >= 0).all(), "single_weights >= 0");
  // TODO(tend): Repeat each column twice, except the first one, or repeat all?
  weights_(0) = single_weights(0);  // The 0th frequency does not need to be repeated
  for (Index i = 1; i < single_weights.size(); i++) {
    weights_(2 * i) = single_weights(i);
    weights_(2 * i - 1) = single_weights(i);
  }
}

double get_prob(const Matrix& prob_per_dim, Index dim, Index tot_dims, Index freq) {
  double prod = 1.0;
  for (Index other_dim = 0; other_dim < tot_dims; other_dim++) {
    prod *= other_dim == dim ? prob_per_dim(other_dim, freq) : prob_per_dim(other_dim, 0);
  }
  return prod;
}

TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(int num_frequencies, const Matrix& prob_per_dim,
                                                       const Matrix& omega_per_dim, Scalar sigma_f,
                                                       const RectSet& x_limits, const bool)
    : num_frequencies_per_dimension_{num_frequencies},
      omega_{combvec(omega_per_dim).transpose()},
      weights_{(num_frequencies - 1) * x_limits.dimension() * 2 + 1},
      sigma_f_{sigma_f},
      x_limits_{x_limits} {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >=, 0);
  LUCID_CHECK_ARGUMENT_CMP(sigma_f, >, 0);
  LUCID_CHECK_ARGUMENT_EQ(prob_per_dim.rows(), x_limits.dimension());
  LUCID_ASSERT((omega_.array() >= 0).all(), "single_weights >= 0");
  LUCID_NOT_IMPLEMENTED();

  Vector single_weights{(num_frequencies - 1) * x_limits.dimension() + 1};
  Index row = 0;
  single_weights(row++) = std::sqrt(get_prob(prob_per_dim, 0, x_limits_.dimension(), 0));
  for (Index current_dim = 0; current_dim < x_limits.dimension(); current_dim++) {
    for (Index freq = 1; freq < num_frequencies_per_dimension_; freq++) {
      single_weights(row++) = std::sqrt(get_prob(prob_per_dim, current_dim, x_limits_.dimension(), freq));
    }
  }

  if (captured_probability_ = single_weights.cwiseProduct(single_weights).sum(); captured_probability_ > 0.94)
    LUCID_DEBUG_FMT("Probability captured by Fourier expansion is {:.3f} percent", captured_probability_);
  else
    LUCID_WARN_FMT("Probability captured by Fourier expansion is only {:.3f} percent", captured_probability_);

  // TODO(tend): Repeat each column twice, except the first one, or repeat all?
  weights_(0) = single_weights(0);  // The 0th frequency does not need to be repeated
  for (Index i = 1; i < single_weights.size(); i++) {
    weights_(2 * i) = single_weights(i);
    weights_(2 * i - 1) = single_weights(i);
  }
}

Vector TruncatedFourierFeatureMap::map_vector(ConstVectorRef x) const {
  auto z = (x - x_limits_.lower_bound()).cwiseQuotient(x_limits_.upper_bound() - x_limits_.lower_bound());
  LUCID_ASSERT(z.size() == omega_.cols(), "z.size() == omega_.cols()");
  // TODO(tend): Does it become a problem if the input is outside the bounds?
  // LUCID_ASSERT((z.array() >= 0).all() && (z.array() <= 1).all(), "0 <= z <= 1");

  Vector z_proj = omega_ * z.transpose();  // It is also computing the 0th frequency, although it is not used later
  Vector trig{2 * z_proj.size() - 1};
  trig(0) = 1;
  for (Index i = 1; i < z_proj.size(); i++) {
    trig(2 * i - 1) = std::cos(z_proj(i));
    trig(2 * i) = std::sin(z_proj(i));
  }
  LUCID_ASSERT((trig.array() >= -1).all() && (trig.array() <= 1).all(), "-1 <= trig <= 1");

  const Vector basis = sigma_f_ * weights_.cwiseProduct(trig);
#ifndef NLOG
  if (Scalar checksum = (basis.cwiseProduct(basis).colwise().sum().array().sqrt() - sigma_f_).abs().maxCoeff();
      checksum > .06) {
    // TODO(tend): this will probably need to be a warning. Maybe only put it for the matrix case?
    LUCID_TRACE_FMT("Checksum failed: Fourier basis frequency bands don't add up: {} > 0.06", checksum);
  }
#endif
  return basis;
}
Matrix TruncatedFourierFeatureMap::map_matrix(ConstMatrixRef x) const { return (*this)(x); }

Matrix TruncatedFourierFeatureMap::apply_impl(ConstMatrixRef x) const {
  Matrix out{x.rows(), weights_.size()};
  for (Index row = 0; row < x.rows(); row++) out.row(row) = map_vector(x.row(row));
  return out;
}

}  // namespace lucid
