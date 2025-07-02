/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"

#include <numbers>

#include "lucid/util/math.h"

namespace lucid {

namespace {

/**
 * Compute the probability measures for each dimension of the input spaces.
 * The normal distribution is divided in @f$ 2 \times \text{num_frequencies} @f$ intervals each of length @f$ 2\pi @f$.
 * @param num_frequencies number of frequencies per dimension
 * @param sigma_l standard deviation of the normal distribution
 * @return matrix containing the CDF values for each dimension
 */
Matrix get_prob_dim_wise(const int num_frequencies, ConstVectorRef sigma_l) {
  // Compute the weights for the feature map
  const Vector omega_dim_wise_lb = (2 * std::numbers::pi * arange(0, num_frequencies)).array() - std::numbers::pi;
  const Vector omega_dim_wise_ub = omega_dim_wise_lb.array() + 2 * std::numbers::pi;

  Matrix prob_dim_wise{sigma_l.size(), num_frequencies};
  for (Dimension i = 0; i < sigma_l.size(); i++) {
    prob_dim_wise.row(i) = normal_cdf(omega_dim_wise_ub, 0, sigma_l(i)) - normal_cdf(omega_dim_wise_lb, 0, sigma_l(i));
    prob_dim_wise.row(i).rightCols(prob_dim_wise.cols() - 1) *= 2;
  }
  return prob_dim_wise;
}

}  // namespace

ConstantTruncatedFourierFeatureMap::ConstantTruncatedFourierFeatureMap(const int num_frequencies,
                                                                       ConstVectorRef sigma_l, const Scalar sigma_f,
                                                                       const RectSet& x_limits)
    : TruncatedFourierFeatureMap{num_frequencies, get_prob_dim_wise(num_frequencies, sigma_l), sigma_f, x_limits} {}
ConstantTruncatedFourierFeatureMap::ConstantTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                       const Scalar sigma_f, const RectSet& x_limits)
    : ConstantTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(x_limits.dimension(), sigma_l), sigma_f,
                                         x_limits} {}

std::unique_ptr<FeatureMap> ConstantTruncatedFourierFeatureMap::clone() const {
  LUCID_TRACE("Cloning");
  return std::make_unique<ConstantTruncatedFourierFeatureMap>(*this);
}

std::ostream& operator<<(std::ostream& os, const ConstantTruncatedFourierFeatureMap& f) {
  return os << "ConstantTruncatedFourierFeatureMap( "
            << "num_frequencies( " << f.num_frequencies() << " ) "
            << "dimension( " << f.dimension() << " ) "
            << "weights( " << f.weights() << " ) )";
}

}  // namespace lucid
