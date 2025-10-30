/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/LogTruncatedFourierFeatureMap.h"

#include <memory>
#include <numbers>

#include "lucid/util/error.h"
#include "lucid/util/math.h"

namespace lucid {

namespace {

/**
 * Compute the probability measures for each dimension of the input spaces.
 * The normal distribution is divided in @f$ 2 \times \text{num_frequencies} @f$ intervals on a logarithmic scale,
 * fill the interval @f$ [0, 3 \sigma_l] @f$, thus capturing 99.7% of the probability density.
 * @param num_frequencies number of frequencies per dimension
 * @param sigma_l standard deviation of the normal distribution
 * @return matrix containing the CDF values for each dimension
 */
Matrix get_prob_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  Matrix prob_per_dim{sigma_l.size(), num_frequencies};
  for (Dimension i = 0; i < sigma_l.size(); i++) {
    const double inverted_sigma_l = 1 / sigma_l(i);
    Vector intervals{Vector::LinSpaced(num_frequencies + 1, 0, std::log(3 * inverted_sigma_l + 1))};
    intervals = (intervals.array().exp() - 1).eval();
    prob_per_dim.row(i) = normal_cdf(intervals.tail(num_frequencies), 0, inverted_sigma_l) -
                          normal_cdf(intervals.head(num_frequencies), 0, inverted_sigma_l);
    prob_per_dim.row(i).rightCols(prob_per_dim.cols() - 1) *= 2;
  }
  return prob_per_dim;
}

Matrix get_omega_per_dim(const int num_frequencies, ConstVectorRef sigma_l) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequencies, >, 0);
  Matrix output{Matrix::Zero(sigma_l.size(), num_frequencies)};
  for (Index row = 0; row < output.rows(); row++) {
    const double inverted_sigma_l = 1 / sigma_l(row);
    Vector intervals{Vector::LinSpaced(num_frequencies + 1, 0, std::log(3 * inverted_sigma_l + 1))};
    intervals = (intervals.array().exp() - 1).eval();
    for (Index col = 1; col < output.cols(); col++) output(row, col) = (intervals(col + 1) - intervals(col)) / 2;
  }
  return output;
}

}  // namespace

LogTruncatedFourierFeatureMap::LogTruncatedFourierFeatureMap(const int num_frequencies, ConstVectorRef sigma_l,
                                                             const Scalar sigma_f, const RectSet& x_limits)
    : TruncatedFourierFeatureMap{num_frequencies, get_prob_per_dim(num_frequencies, sigma_l),
                                 get_omega_per_dim(num_frequencies, sigma_l), sigma_f, x_limits} {}
LogTruncatedFourierFeatureMap::LogTruncatedFourierFeatureMap(const int num_frequencies, const double sigma_l,
                                                             const Scalar sigma_f, const RectSet& x_limits)
    : LogTruncatedFourierFeatureMap{num_frequencies, Vector::Constant(x_limits.dimension(), sigma_l), sigma_f,
                                    x_limits} {}

RectSet LogTruncatedFourierFeatureMap::get_periodic_x_limits(const int, ConstVectorRef) const { LUCID_NOT_IMPLEMENTED(); }

std::unique_ptr<FeatureMap> LogTruncatedFourierFeatureMap::clone() const {
  LUCID_TRACE("Cloning");
  return std::make_unique<LogTruncatedFourierFeatureMap>(*this);
}

std::ostream& operator<<(std::ostream& os, const LogTruncatedFourierFeatureMap& f) {
  return os << "LogTruncatedFourierFeatureMap( "
            << "num_frequencies( " << f.num_frequencies() << " ) "
            << "dimension( " << f.dimension() << " ) "
            << "weights( " << f.weights() << " ) )";
}

}  // namespace lucid
