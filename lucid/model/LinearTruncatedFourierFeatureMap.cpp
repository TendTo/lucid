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
 * fill the interval @f$ [0, 3 \sigma_l] @f$, thus capturing 99.7% of the probability density.
 * @param num_frequencies number of frequencies per dimension
 * @param sigma_l standard deviation of the normal distribution
 * @return matrix containing the CDF values for each dimension
 */
Matrix get_prob_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);
  Matrix prob_per_dim{sigma_l.size(), num_frequencies};
  for (Dimension i = 0; i < sigma_l.size(); i++) {
    const double inverted_sigma_l = 1.0 / sigma_l(i);
    const double offset = 3 * inverted_sigma_l / (2 * num_frequencies - 1);
    Vector intervals{num_frequencies + 1};
    intervals(0) = 0;
    intervals(1) = offset;
    for (Index j = 2; j < intervals.size(); j++) intervals(j) = intervals(j - 1) + offset * 2;
    prob_per_dim.row(i) = normal_cdf(intervals.tail(num_frequencies), 0, inverted_sigma_l) -
                          normal_cdf(intervals.head(num_frequencies), 0, inverted_sigma_l);
    prob_per_dim.row(i) *= 2;
  }
  return prob_per_dim;
}

Matrix get_omega_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);
  return Matrix::NullaryExpr(sigma_l.size(), num_frequencies,
                             [&sigma_l, num_frequencies](const Index row, const Index col) {
                               const double offset = 3 * 1 / sigma_l(row) / (num_frequencies - 0.5);
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

}  // namespace lucid
