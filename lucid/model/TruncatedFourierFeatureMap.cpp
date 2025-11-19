/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/TruncatedFourierFeatureMap.h"

#include <memory>
#include <numbers>
#include <ostream>
#include <utility>

#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"

namespace lucid {
TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(const int num_frequencies, const Matrix& prob_per_dim,
                                                       const Matrix& omega_per_dim, ConstVectorRef sigma_l,
                                                       const Scalar sigma_f, const RectSet& X_bounds)
    : num_frequencies_per_dimension_{num_frequencies},
      omega_{combvec(omega_per_dim).transpose()},
      weights_{::lucid::pow<Index>(num_frequencies, X_bounds.dimension()) * 2 - 1},
      sigma_f_{sigma_f},
      sigma_l_{sigma_l},
      X_bounds_{X_bounds} {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >=, 2);
  LUCID_CHECK_ARGUMENT_CMP(sigma_f, >, 0);
  LUCID_CHECK_ARGUMENT_EQ(prob_per_dim.rows(), X_bounds.dimension());
  LUCID_ASSERT((omega_.array() >= 0).all(), "single_weights >= 0");
  LUCID_ASSERT(omega_.rows() == ::lucid::pow<Dimension>(num_frequencies, X_bounds.dimension()),
               "omega_.rows() == num_frequencies^dimension");
  LUCID_ASSERT(omega_.cols() == X_bounds.dimension(), "omega_.cols() == dimension");

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
TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(const int num_frequencies, const Matrix& prob_per_dim,
                                                       const Matrix& omega_per_dim, const double sigma_l,
                                                       const Scalar sigma_f, const RectSet& X_bounds)
    : TruncatedFourierFeatureMap{num_frequencies, prob_per_dim,
                                 omega_per_dim,   Vector::Constant(X_bounds.dimension(), sigma_l),
                                 sigma_f,         X_bounds} {}

double get_prob(const Matrix& prob_per_dim, const Index dim, const Index tot_dims, const Index freq) {
  double prod = 1.0;
  for (Index other_dim = 0; other_dim < tot_dims; other_dim++) {
    prod *= other_dim == dim ? prob_per_dim(other_dim, freq) : prob_per_dim(other_dim, 0);
  }
  return prod;
}

Vector TruncatedFourierFeatureMap::map_vector(ConstVectorRef x) const {
  LUCID_CHECK_ARGUMENT_EQ(x.size(), X_bounds_.dimension());
  auto z = (x - X_bounds_.lower_bound()).cwiseQuotient(X_bounds_.upper_bound() - X_bounds_.lower_bound());
  LUCID_ASSERT(z.size() == omega_.cols(), "z.size() == omega_.cols()");

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
Vector TruncatedFourierFeatureMap::invert_vector(ConstVectorRef y) const {
  LUCID_NOT_IMPLEMENTED();
  LUCID_CHECK_ARGUMENT_EQ(y.size(), weights_.size());
  // Step 1: Reverse the sigma_f and weights scaling
  Vector trig = y.cwiseQuotient(weights_) / sigma_f_;

  // Step 2: Extract z_proj from the trigonometric representation
  // trig = [1, cos(z_proj[1]), sin(z_proj[1]), ..., cos(z_proj[n]), sin(z_proj[n])]
  // We can recover z_proj[i] using atan2(sin, cos)
  LUCID_ASSERT(trig.size() % 2 == 1, "trig.size() must be odd");
  Vector z_proj{(trig.size() + 1) / 2};
  z_proj(0) = 0;  // The 0th frequency doesn't contribute to z_proj
  for (Index i = 1; i < z_proj.size(); i++) {
    const Scalar cos_val = trig(2 * i - 1);
    const Scalar sin_val = trig(2 * i);
    z_proj(i) = std::atan2(sin_val, cos_val);
  }

  fmt::println("z_proj shape: {}", z_proj.size());
  fmt::println("omega shape: {} x {}", omega_.rows(), omega_.cols());
  // Step 3: Solve for z using the pseudo-inverse of omega
  // z_proj = omega * z^T  =>  z^T = omega^+ * z_proj^T
  // Use least-squares solution via normal equations: (omega^T * omega) * z^T = omega^T * z_proj^T
  Vector z{(omega_.transpose() * omega_).ldlt().solve(omega_.transpose() * z_proj.transpose())};

  fmt::println("z shape: {}", z.size());
  fmt::println("X_bounds_.upper_bound(): {}", X_bounds_.upper_bound());

  // Step 4: Denormalize from [0, 1] to the original bounds
  return z.cwiseProduct(X_bounds_.upper_bound() - X_bounds_.lower_bound()) + X_bounds_.lower_bound();
}
Matrix TruncatedFourierFeatureMap::map_matrix(ConstMatrixRef x) const { return (*this)(x); }

Matrix TruncatedFourierFeatureMap::apply_impl(ConstMatrixRef x) const {
  Matrix out{x.rows(), weights_.size()};
  for (Index row = 0; row < x.rows(); row++) out.row(row) = map_vector(x.row(row));
  return out;
}
Matrix TruncatedFourierFeatureMap::invert_impl(ConstMatrixRef y) const {
  Matrix out{y.rows(), X_bounds_.dimension()};
  for (Index row = 0; row < y.rows(); row++) out.row(row) = invert_vector(y.row(row));
  return out;
}

RectSet TruncatedFourierFeatureMap::get_periodic_set() const { LUCID_NOT_IMPLEMENTED(); }

std::unique_ptr<FeatureMap> TruncatedFourierFeatureMap::clone() const {
  return std::make_unique<TruncatedFourierFeatureMap>(*this);
}

std::ostream& operator<<(std::ostream& os, const TruncatedFourierFeatureMap& map) { return os << map.to_string(); }

}  // namespace lucid
