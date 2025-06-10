/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LogTruncatedFourierFeatureMap class.
 */
#pragma once

#include <iosfwd>

#include "lucid/lib/eigen.h"
#include "lucid/model/RectSet.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"

namespace lucid {

/**
 * Truncated Fourier feature map using log-sized intervals between each interval of the normal distribution,
 * from the origin to 3 standard deviations on each side.
 * The weights are computed from the n-dimensional cumulative distribution function
 * of the normal distribution with standard deviation @sigma_l, i.e.
 * @f[
 *  w_j^2 := \int_{(2\hat{\zeta}_j-1)\pi}^{(2\hat{\zeta}_j+1)\pi}\mathcal{N}(d\xi|0,\Sigma), 1 \le j \le M .
 * @f]
 * where @f$ \hat{\zeta}_j @f$ is a transformation of the original @f$ \zeta_j @f$.
 * @see TruncatedFourierFeatureMap
 */
class LogTruncatedFourierFeatureMap final : public TruncatedFourierFeatureMap {
 public:
  LogTruncatedFourierFeatureMap(int num_frequencies, ConstVectorRef sigma_l, Scalar sigma_f, const RectSet& x_limits);
  LogTruncatedFourierFeatureMap(int num_frequencies, double sigma_l, Scalar sigma_f, const RectSet& x_limits);
};

std::ostream& operator<<(std::ostream& os, const LogTruncatedFourierFeatureMap& f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::LogTruncatedFourierFeatureMap)

#endif  // LUCID_INCLUDE_FMT
