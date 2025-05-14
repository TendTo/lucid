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
#include "lucid/util/error.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"

namespace lucid {

TruncatedFourierFeatureMap::TruncatedFourierFeatureMap(const int num_frequencies, const Matrix& prob_dim_wise,
                                                       const Scalar sigma_f, const RectSet& x_limits)
    : num_frequencies_per_dimension_{num_frequencies},
      omega_{::lucid::pow(num_frequencies, x_limits.dimension()), x_limits.dimension()},
      weights_{::lucid::pow(num_frequencies, x_limits.dimension()) * 2 - 1},
      sigma_f_{sigma_f},
      x_limits_{x_limits} {
  LUCID_CHECK_ARGUMENT_EXPECTED(prob_dim_wise.rows() == x_limits.dimension(), "sigma_l.size() == x_limits.dimension()",
                                prob_dim_wise.rows(), x_limits.dimension());
  // Iterate over all possible combinations where the values in the vector can go from 0 to num_frequencies_ - 1
  // [ 0, ..., 0 ] -> [ 0, ..., 1 ] -> ... -> [ num_frequencies_ - 1, ..., num_frequencies_ - 1, ]
  IndexIterator<Index> it{static_cast<std::size_t>(x_limits_.dimension()), num_frequencies_per_dimension_};
  for (Index row = 0; it; ++it, ++row) {
    // For each combination, compute the product of the sines and cosines of the values in the vector
    // TODO(tend): We can probably remove the reverse
    Index col = 0;
    for (const Index val : std::views::reverse(it.indexes()))
      omega_(row, col++) = 2 * std::numbers::pi * static_cast<double>(val);
  }
  LUCID_ASSERT((omega_.array() >= 0).all(), "single_weights >= 0");

  const Matrix comb = combvec(prob_dim_wise);
  const auto prod = comb.colwise().prod();
  if (captured_probability_ = prod.sum(); captured_probability_ > 0.9)
    LUCID_INFO_FMT("Probability captured by Fourier expansion is {:.3f} percent", captured_probability_);
  else
    LUCID_WARN_FMT("Probability captured by Fourier expansion is only {:.3f} percent", captured_probability_);

  const auto single_weights = prod.cwiseSqrt();
  LUCID_ASSERT((single_weights.array() >= 0).all(), "single_weights >= 0");
  // TODO(tend): Repeat each column twice, except the first one, or repeat all?
  for (Index i = 0; i < single_weights.size(); i++) {
    weights_(2 * i) = single_weights(i);
    if (i != 0) weights_(2 * i - 1) = single_weights(i);
  }
}

Vector TruncatedFourierFeatureMap::map_vector(ConstVectorRef x) const {
  auto z = (x - x_limits_.lower_bound()).cwiseQuotient(x_limits_.upper_bound() - x_limits_.lower_bound());
  LUCID_ASSERT(z.size() == omega_.cols(), "z.size() == omega_.cols()");
  LUCID_ASSERT((z.array() >= 0).all() && (z.array() <= 1).all(), "0 <= z <= 1");

  Vector z_proj = omega_ * z;  // It is also computing the 0th frequency, although it is not used later
  Vector trig{2 * z_proj.size() - 1};
  trig(0) = 1;
  for (Index i = 1; i < z_proj.size(); i++) {
    trig(2 * i - 1) = std::cos(z_proj(i));
    trig(2 * i) = std::sin(z_proj(i));
  }
  LUCID_ASSERT((trig.array() >= -1).all() && (trig.array() <= 1).all(), "-1 <= trig <= 1");

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
