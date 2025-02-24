/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/TruncatedFourierFeatureMap.h"

#include "lucid/util/IndexIterator.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"

namespace lucid {

TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(const long num_frequencies, const Dimension input_dimension,
                                                       ConstVectorRef sigma_l)
    : num_frequencies_per_dimension_{num_frequencies},
      omega_{::lucid::pow(num_frequencies, input_dimension), input_dimension},
      weights_{::lucid::pow(num_frequencies, input_dimension) * 2} {
  // Iterate over all possible combinations where the values in the vector can go from 0 to num_frequencies_ - 1
  // [ 0, ..., 0 ] -> [ 0, ..., 1 ] -> ... -> [ num_frequencies_ - 1, ..., num_frequencies_ - 1, ]
  IndexIterator it{static_cast<std::size_t>(input_dimension), num_frequencies_per_dimension_};
  for (Index row = 0; it; ++it, ++row) {
    // For each combination, compute the product of the sines and cosines of the values in the vector
    // TODO(tend): We can probably remove the revrse
    Index col = 0;
    for (const Index val : std::views::reverse(it.indexes())) omega_(row, col++) = 2 * M_PI * static_cast<double>(val);
  }

  // Compute the weights for the feature map
  const Vector omega_dim_wise_lb = (2 * M_PI * arange(0, num_frequencies_per_dimension_)).array() - M_PI;
  const Vector omega_dim_wise_ub = omega_dim_wise_lb.array() + 2 * M_PI;

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
    weights_(2 * i + 1) = single_weights(i);
  }
}

Vector TruncatedFourierFeatureMap::operator()(ConstVectorRef x) const { return Vector::Zero(2 * x.size()); }

}  // namespace lucid