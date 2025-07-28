/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ConstantTruncatedFourierFeatureMap class.
 */
#pragma once

#include <iosfwd>
#include <memory>

#include "TruncatedFourierFeatureMap.h"
#include "lucid/lib/eigen.h"
#include "lucid/model/RectSet.h"

namespace lucid {

/**
 * Truncated Fourier feature map using fixed intervals between each interval of the normal distribution.
 * The weights are computed from the n-dimensional cumulative distribution function
 * of the normal distribution with standard deviation @sigmal, i.e.
 * @f[
 *  w_j^2 := \int_{(2\zeta_j-1)\pi}^{(2\zeta_j+1)\pi}\mathcal{N}(d\xi|0,\Sigma), 1 \le j \le M .
 * @f]
 * @see TruncatedFourierFeatureMap
 */
class ConstantTruncatedFourierFeatureMap final : public TruncatedFourierFeatureMap {
 public:
  ConstantTruncatedFourierFeatureMap(int num_frequencies, ConstVectorRef sigma_l, Scalar sigma_f,
                                     const RectSet& x_limits);
  ConstantTruncatedFourierFeatureMap(int num_frequencies, double sigma_l, Scalar sigma_f, const RectSet& x_limits);

  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override;
};

std::ostream& operator<<(std::ostream& os, const ConstantTruncatedFourierFeatureMap& f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ConstantTruncatedFourierFeatureMap)

#endif
