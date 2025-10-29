/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/LinearTruncatedFourierFeatureMap.h"

#include <memory>

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
                                                                   const Scalar sigma_f, const RectSet& x_limits)
    : TruncatedFourierFeatureMap{num_frequencies, get_prob_per_dim(num_frequencies, sigma_l),
                                 get_omega_per_dim(num_frequencies, sigma_l), sigma_f, x_limits} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                   const Scalar sigma_f, const RectSet& x_limits)
    : LinearTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(x_limits.dimension(), sigma_l), sigma_f,
                                       x_limits} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, ConstVectorRef sigma_l,
                                                                   const Scalar sigma_f, const RectSet& x_limits, bool)
    : TruncatedFourierFeatureMap{num_frequencies,
                                 get_prob_per_dim(num_frequencies, sigma_l),
                                 get_omega_per_dim(num_frequencies, sigma_l),
                                 sigma_f,
                                 x_limits,
                                 true} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                   const Scalar sigma_f, const RectSet& x_limits, bool)
    : LinearTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(x_limits.dimension(), sigma_l), sigma_f,
                                       x_limits, true} {}

std::unique_ptr<FeatureMap> LinearTruncatedFourierFeatureMap::clone() const {
  LUCID_TRACE("Cloning");
  return std::make_unique<LinearTruncatedFourierFeatureMap>(*this);
}

std::ostream& operator<<(std::ostream& os, const LinearTruncatedFourierFeatureMap& f) {
  return os << "LinearTruncatedFourierFeatureMap( "
            << "num_frequencies( " << f.num_frequencies() << " ) "
            << "dimension( " << f.dimension() << " ) "
            << "weights( " << f.weights() << " ) )";
}

RectSet LinearTruncatedFourierFeatureMap::periodic_x_limits(const int num_frequencies, ConstVectorRef sigma_l, 
                                                            const RectSet& x_limits) const {
  LUCID_CHECK_ARGUMENT_EQ(sigma_l.size(), x_limits.dimension());
  LUCID_CHECK_ARGUMENT_CMP(sigma_l.minCoeff(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);

  const double denom = 2.0 * static_cast<double>(num_frequencies) - 1.0;
  Vector dilation = (3.0 * sigma_l.cwiseInverse() / denom).matrix();

  const Vector lower = x_limits.lower_bound();
  const Vector upper = x_limits.upper_bound();
  LUCID_ASSERT(lower.size() == upper.size(), "lower and upper sizes match");
  const Vector lengths = upper - lower;
  LUCID_ASSERT((lengths.array() >= 0).all(), "upper >= lower");

  Vector new_lower = lower;
  Vector new_upper = lower + lengths.cwiseProduct(dilation);

  return RectSet(new_lower, new_upper);
}

}  // namespace lucid
