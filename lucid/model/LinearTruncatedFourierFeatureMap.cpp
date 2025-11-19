/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/LinearTruncatedFourierFeatureMap.h"

#include <memory>
#include <numbers>

#include "lucid/util/error.h"

namespace lucid {

namespace {

/**
 * Compute the probability measures for each dimension of the input spaces.
 * The normal distribution is divided in @f$ 2 \times \text{num_frequencies} @f$ intervals of equal size so that they
 * fill the interval @f$ [0, 3 \sigma_l^{-1}] @f$, thus capturing 99.7% of the probability density.
 * @param num_frequencies number of frequencies per dimension
 * @param sigma_l inverse of the standard deviation we are going to use for the normal distribution
 * @return matrix containing the CDF values for each dimension
 */
Matrix get_prob_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);
  Matrix prob_per_dim{sigma_l.size(), num_frequencies};
  for (Dimension i = 0; i < sigma_l.size(); i++) {
    const double inverted_sigma_l = 1.0 / sigma_l(i);
    const double offset = 3 * inverted_sigma_l / (2 * num_frequencies - 1);
    Vector intervals{Vector::NullaryExpr(
        num_frequencies + 1, [offset](const Index idx) { return static_cast<double>(2 * idx - 1) * offset; })};
    // Reset the first interval to 0 instead of -offset
    intervals(0) = 0;
    prob_per_dim.row(i) = normal_cdf(intervals.tail(num_frequencies), 0, inverted_sigma_l) -
                          normal_cdf(intervals.head(num_frequencies), 0, inverted_sigma_l);
    prob_per_dim.row(i) *= 2;
  }
  return prob_per_dim;
}

/**
 * Compute the omega values for each dimension of the input spaces.
 * The omegas fall in the middle of each interval defined in get_prob_per_dim, with the first being at the origin.
 * @param num_frequencies number of frequencies per dimension
 * @param sigma_l inverse of the standard deviation we are going to use for the normal distribution
 * @return matrix containing the omega values for each dimension
 */
Matrix get_omega_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);
  return Matrix::NullaryExpr(sigma_l.size(), num_frequencies,
                             [&sigma_l, num_frequencies](const Index row, const Index col) {
                               const double inverted_sigma_l = 1.0 / sigma_l(row);
                               const double offset = 3 * inverted_sigma_l / (num_frequencies - 0.5);
                               return offset * static_cast<double>(col);
                             });
}

}  // namespace

LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, ConstVectorRef sigma_l,
                                                                   const Scalar sigma_f, const RectSet& X_bounds)
    : TruncatedFourierFeatureMap{num_frequencies,
                                 get_prob_per_dim(num_frequencies, sigma_l),
                                 get_omega_per_dim(num_frequencies, sigma_l),
                                 sigma_l,
                                 sigma_f,
                                 X_bounds} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                   const Scalar sigma_f, const RectSet& X_bounds)
    : LinearTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(X_bounds.dimension(), sigma_l), sigma_f,
                                       X_bounds} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, ConstVectorRef sigma_l,
                                                                   const Scalar sigma_f, const RectSet& X_bounds, bool)
    : TruncatedFourierFeatureMap{num_frequencies,
                                 get_prob_per_dim(num_frequencies, sigma_l),
                                 get_omega_per_dim(num_frequencies, sigma_l),
                                 sigma_f,
                                 X_bounds,
                                 true} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                   const Scalar sigma_f, const RectSet& X_bounds, bool)
    : LinearTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(X_bounds.dimension(), sigma_l), sigma_f,
                                       X_bounds, true} {}

RectSet LinearTruncatedFourierFeatureMap::get_periodic_set() const {
  // Divide the space of size 3 * sigma_l^{-1} into (2 * num_frequencies - 1) intervals.
  // Each interval will be normalized by double the dilation factor, and then extended to be between [0, 2pi].
  const double denom = static_cast<double>(num_frequencies_per_dimension_) - 0.5;
  const auto dilation = 3.0 * sigma_l_.cwiseInverse() / denom;

  const auto lengths = X_bounds_.upper_bound() - X_bounds_.lower_bound();
  LUCID_ASSERT(lengths.minCoeff() >= 0, "upper >= lower");

  const Vector new_upper = X_bounds_.lower_bound() + 2 * std::numbers::pi * lengths.cwiseQuotient(dilation);
  return {X_bounds_.lower_bound(), new_upper};
}

std::unique_ptr<FeatureMap> LinearTruncatedFourierFeatureMap::clone() const {
  LUCID_TRACE("Cloning");
  return std::make_unique<LinearTruncatedFourierFeatureMap>(*this);
}

std::string LinearTruncatedFourierFeatureMap::to_string() const {
  return fmt::format("LinearTruncatedFourierFeatureMap( num_frequencies( {} ) dimension( {} ) weights( {} ) )",
                     num_frequencies(), dimension(), weights_);
}

std::ostream& operator<<(std::ostream& os, const LinearTruncatedFourierFeatureMap& f) { return os << f.to_string(); }

}  // namespace lucid
