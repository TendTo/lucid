/**
 * @author Room 6.030
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
 * @plot
 * {
 * "f": "(x) => (1 / (3.0 * Math.sqrt(2 * Math.PI))) * Math.exp(-((x - 0.0) ** 2) / (2 * 3.0 ** 2))",
 * "xBounds": [-22, 22],
 * "n": 100,
 * "fill": "tozeroy",
 * "fillGradientH": [[0,"transparent"], [0.5,"transparent"],
 * [0.500001,"rgb(56,230,195)"], [0.5714285, "rgb(56,230,195)"],
 * [0.5714286,"rgb(137,243,195)"], [0.714286,"rgb(137,243,195)"],
 * [0.714287,"rgb(113,226,244)"], [0.85714285,"rgb(113,226,244)"],
 * [0.85714286,"rgb(0,140,186)"], [1,"rgb(0,140,186)"]
 * ],
 * "vLines": [
 * {"x": 0, "c": "red", "s": "solid", "label": "ω₀"},
 * {"x": 6.2831, "yMax": 0.005, "c": "transparent", "s": "solid", "label": "ω₁"},
 * {"x": 12.566, "yMax": 0.005, "c": "transparent", "s": "solid", "label": "ω₂"},
 * {"x": 18.849, "yMax": 0.005, "c": "transparent", "s": "solid", "label": "ω₃"},
 * {"x": 3.1415, "s": "dash"}, {"x": 9.4247, "s": "dash"}, {"x": 15.7079, "s": "dash"}, {"x": 21.991, "s": "dash"},
 * {"x": -3.1415, "s": "dash"}, {"x": -9.4247, "s": "dash"}, {"x": -15.7079, "s": "dash"}, {"x": -21.991, "s": "dash"}],
 * "annotations": [
 *  {"x": 1.7, "y": 0.1, "text": "w₀²/2"},
 *  {"x": 6.2831, "y": 0.01, "text": "w₁²/2"},
 *  {"x": 12.566, "y": 0.0, "text": "w₂²/2"},
 *  {"x": 18.849, "y": 0.0, "text": "w₃²/2"}
 *  ]
 * }
 * @endplot
 * @see TruncatedFourierFeatureMap
 */
class ConstantTruncatedFourierFeatureMap final : public TruncatedFourierFeatureMap {
 public:
  ConstantTruncatedFourierFeatureMap(int num_frequencies, ConstVectorRef sigma_l, Scalar sigma_f,
                                     const RectSet& x_limits);
  ConstantTruncatedFourierFeatureMap(int num_frequencies, double sigma_l, Scalar sigma_f, const RectSet& x_limits);

  [[nodiscard]] RectSet get_periodic_set(ConstVectorRef sigma_l) const override;

  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override;
};

std::ostream& operator<<(std::ostream& os, const ConstantTruncatedFourierFeatureMap& f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ConstantTruncatedFourierFeatureMap)

#endif
