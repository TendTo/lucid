/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LinearTruncatedFourierFeatureMap class.
 */
#pragma once

#include <iosfwd>
#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/model/RectSet.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"

namespace lucid {

/**
 * Truncated Fourier feature map using same-sized intervals between each interval of the normal distribution,
 * from the origin to 3 standard deviations on each side.
 * The weights are computed from the n-dimensional cumulative distribution function
 * of the normal distribution with standard deviation @sigmal, i.e.
 * @f[
 *  w_j^2 := \int_{(2\hat{\zeta}_j-1)\pi}^{(2\hat{\zeta}_j+1)\pi}\mathcal{N}(d\xi|0,\Sigma), 1 \le j \le M .
 * @f]
 * where @f$ \hat{\zeta}_j @f$ is a linear transformation of the original @f$ \zeta_j @f$.
 * For example, with @f$ \sigma_l = 3 @f$ and @f$ M = 4 @f$, we split the normal distribution into 4 intervals
 * on each side of the origin.
 * Then, we sum each interval with its specular counterpart.
 * @plot
 * {
 * "f": "(x) => (1 / (3.0 * Math.sqrt(2 * Math.PI))) * Math.exp(-((x - 0.0) ** 2) / (2 * 3.0 ** 2))",
 * "xBounds": [-9, 9],
 * "n": 100,
 * "vBoundedLines": [{"x": 0, "c": "red", "s": "solid"}, {"x": 1.28571}, {"x": 3.85714}, {"x": 6.42857}, {"x": 9},
 * {"x": -1.28571}, {"x": -3.85714}, {"x": -6.42857}, {"x": -9}]
 * }
 * @endplot
 * @see TruncatedFourierFeatureMap
 */
class LinearTruncatedFourierFeatureMap final : public TruncatedFourierFeatureMap {
 public:
  LinearTruncatedFourierFeatureMap(int num_frequencies, ConstVectorRef sigma_l, Scalar sigma_f,
                                   const RectSet& x_limits);
  LinearTruncatedFourierFeatureMap(int num_frequencies, double sigma_l, Scalar sigma_f, const RectSet& x_limits);
  LinearTruncatedFourierFeatureMap(int num_frequencies, ConstVectorRef sigma_l, Scalar sigma_f, const RectSet& x_limits,
                                   bool);
  LinearTruncatedFourierFeatureMap(int num_frequencies, double sigma_l, Scalar sigma_f, const RectSet& x_limits, bool);

  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override;
};

std::ostream& operator<<(std::ostream& os, const LinearTruncatedFourierFeatureMap& f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::LinearTruncatedFourierFeatureMap)

#endif  // LUCID_INCLUDE_FMT
