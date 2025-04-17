/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/TruncatedFourierFeatureMap.h"

#include <numbers>
#include <utility>

#include "lucid/util/IndexIterator.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"

namespace lucid {

TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(const long num_frequencies, const Dimension input_dimension,
                                                       ConstVectorRef sigma_l, Scalar sigma_f, Matrix x_limits)
    : num_frequencies_per_dimension_{num_frequencies},
      omega_{::lucid::pow(num_frequencies, input_dimension), input_dimension},
      weights_{::lucid::pow(num_frequencies, input_dimension) * 2 - 1},
      sigma_f_{sigma_f},
      x_limits_{std::move(x_limits)} {
  // Iterate over all possible combinations where the values in the vector can go from 0 to num_frequencies_ - 1
  // [ 0, ..., 0 ] -> [ 0, ..., 1 ] -> ... -> [ num_frequencies_ - 1, ..., num_frequencies_ - 1, ]
  IndexIterator<Index> it{static_cast<std::size_t>(input_dimension), num_frequencies_per_dimension_};
  for (Index row = 0; it; ++it, ++row) {
    // For each combination, compute the product of the sines and cosines of the values in the vector
    // TODO(tend): We can probably remove the reverse
    Index col = 0;
    for (const Index val : std::views::reverse(it.indexes()))
      omega_(row, col++) = 2 * std::numbers::pi * static_cast<double>(val);
  }

  // Compute the weights for the feature map
  const Vector omega_dim_wise_lb =
      (2 * std::numbers::pi * arange(0, num_frequencies_per_dimension_)).array() - std::numbers::pi;
  const Vector omega_dim_wise_ub = omega_dim_wise_lb.array() + 2 * std::numbers::pi;

  Matrix prob_dim_wise{input_dimension, num_frequencies_per_dimension_};
  for (Dimension i = 0; i < input_dimension; i++) {
    prob_dim_wise.row(i) = normal_cdf(omega_dim_wise_ub, 0, sigma_l(i)) - normal_cdf(omega_dim_wise_lb, 0, sigma_l(i));
    prob_dim_wise.row(i).rightCols(prob_dim_wise.cols() - 1) *= 2;
  }

  const Matrix comb = combvec(prob_dim_wise);
  auto prod = comb.colwise().prod().transpose();
  if (Scalar sum = prod.sum(); sum > 0.9)
    LUCID_INFO_FMT("Probability captured by Fourier expansion is {:.3f} percent", sum);
  else
    LUCID_WARN_FMT("Probability captured by Fourier expansion is only {:.3f} percent", sum);

  const auto single_weights = prod.transpose().cwiseSqrt();
  // TODO(tend): Repeat each column twice, except the first one, or repeat all?
  for (Index i = 0; i < single_weights.size(); i++) {
    weights_(2 * i) = single_weights(i);
    if (i != 0) weights_(2 * i - 1) = single_weights(i);
  }
}
TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(long num_frequencies, Dimension input_dimension,
                                                       ConstVectorRef sigma_l, Scalar sigma_f, const RectSet& x_limits)
    : TruncatedFourierFeatureMap{num_frequencies, input_dimension, sigma_l, sigma_f, static_cast<Matrix>(x_limits)} {}

Vector TruncatedFourierFeatureMap::map_vector(ConstVectorRef x) const {
  auto z = (x.transpose() - x_limits_.row(0)).cwiseQuotient(x_limits_.row(1) - x_limits_.row(0));

  Vector z_proj = omega_ * z.transpose();
  Vector trig{2 * z_proj.size() - 1};
  trig(0) = 1;
  for (Index i = 1; i < z_proj.size(); i++) {
    trig(2 * i - 1) = std::cos(z_proj(i));
    trig(2 * i) = std::sin(z_proj(i));
  }

  auto basis = sigma_f_ * weights_.cwiseProduct(trig);
  if (Scalar checksum = (basis.cwiseProduct(basis).colwise().sum().array().sqrt() - sigma_f_).abs().maxCoeff();
      checksum > 1e-3) {
    // TODO(tend): this will probably need to be a warning. Maybe only put it for the matrix case?
    LUCID_TRACE_FMT("Checksum failed: Fourier basis frequency bands don't add up: {} > 1e-3", checksum);
  }
  return basis;
}
Matrix TruncatedFourierFeatureMap::map_matrix(ConstMatrixRef x) const { return (*this)(x); }

Matrix TruncatedFourierFeatureMap::operator()(ConstMatrixRef x) const {
  Matrix out{x.rows(), weights_.size()};
  for (Index row = 0; row < x.rows(); row++) out.row(row) = map_vector(x.row(row)).transpose();
  return out;
}

}  // namespace lucid
