/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"

#include <memory>
#include <numbers>

#include "lucid/util/error.h"
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
Matrix get_prob_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  // Compute the weights for the feature map
  const Vector omega_per_dim_lb = (2 * std::numbers::pi * arange(0, num_frequencies)).array() - std::numbers::pi;
  const Vector omega_per_dim_ub = omega_per_dim_lb.array() + 2 * std::numbers::pi;

  Matrix prob_per_dim{sigma_l.size(), num_frequencies};
  for (Dimension i = 0; i < sigma_l.size(); i++) {
    const double inverted_sigma_l = 1 / sigma_l(i);
    prob_per_dim.row(i) =
        normal_cdf(omega_per_dim_ub, 0, inverted_sigma_l) - normal_cdf(omega_per_dim_lb, 0, inverted_sigma_l);
    // No need to multiply the first column by 2, as it is centered at 0 and we are taking [-pi, pi] as interval.
    prob_per_dim.row(i).rightCols(prob_per_dim.cols() - 1) *= 2;
  }
  return prob_per_dim;
}

Matrix get_omega_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);
  const double offset = (2 * std::numbers::pi) / static_cast<double>(num_frequencies);
  return Matrix::NullaryExpr(sigma_l.size(), num_frequencies,
                             [offset](const Index, const Index col) { return offset * static_cast<double>(col); });
}

}  // namespace

ConstantTruncatedFourierFeatureMap::ConstantTruncatedFourierFeatureMap(const int num_frequencies,
                                                                       ConstVectorRef sigma_l, const Scalar sigma_f,
                                                                       const RectSet& X_bounds)
    : TruncatedFourierFeatureMap{num_frequencies,
                                 get_prob_per_dim(num_frequencies, sigma_l),
                                 get_omega_per_dim(num_frequencies, sigma_l),
                                 sigma_l,
                                 sigma_f,
                                 X_bounds} {
  LUCID_WARN("ConstantTruncatedFourierFeatureMap is deprecated. Use the LinearTruncatedFourierFeatureMap instead");
}
ConstantTruncatedFourierFeatureMap::ConstantTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                                       const Scalar sigma_f, const RectSet& X_bounds)
    : ConstantTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(X_bounds.dimension(), sigma_l), sigma_f,
                                         X_bounds} {}

RectSet ConstantTruncatedFourierFeatureMap::get_periodic_set() const { return X_bounds_; }

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
