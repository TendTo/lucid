/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/LinearTruncatedFourierFeatureMap.h"

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
Matrix get_prob_dim_wise(const int num_frequencies, ConstVectorRef sigma_l) {
  Matrix prob_dim_wise{sigma_l.size(), num_frequencies};
  for (Dimension i = 0; i < sigma_l.size(); i++) {
    Vector intervals{Vector::LinSpaced(num_frequencies + 1, 0, 3 * sigma_l(i))};
    prob_dim_wise.row(i) = normal_cdf(intervals.head(num_frequencies), 0, sigma_l(i)) -
                           normal_cdf(intervals.tail(num_frequencies), 0, sigma_l(i));
    prob_dim_wise.row(i).rightCols(prob_dim_wise.cols() - 1) *= 2;
  }
  return prob_dim_wise;
}

}  // namespace

LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, ConstVectorRef sigma_l,
                                                                   const Scalar sigma_f, const RectSet& x_limits)
    : TruncatedFourierFeatureMap{num_frequencies, get_prob_dim_wise(num_frequencies, sigma_l), sigma_f, x_limits} {}
LinearTruncatedFourierFeatureMap::LinearTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                   const Scalar sigma_f, const RectSet& x_limits)
    : LinearTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(x_limits.dimension(), sigma_l), sigma_f,
                                       x_limits} {}

}  // namespace lucid
