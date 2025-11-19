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
 * Recall the structure of the feature map
 * @phiMx_extended
 * After defining the quantity @f$ \vartheta = 6 \sigma_l^{-1} / (2M-1) @f$,
 * we compute the the @f$ \omega_j @f$ as
 * @f[
 * \omega_j = \text{diag}(\zeta_j) \cdot \vartheta, \quad 1 \le j \le M ,
 * @f]
 * and the weights from the n-dimensional cumulative distribution function
 * of the normal distribution with standard deviation @f$ \sigma_l^{-1} @f$, i.e.
 * @f[
 *  w_j^2 = \int_{\omega_j - \vartheta/2}^{\omega_j + \vartheta/2}\mathcal{N}(d\xi|0,\Sigma^{-1}), \quad 0 \le j \le M .
 * @f]
 * where @f$ \zeta_j \in \mathbb{N}^n_{\ge 0}@f$ is a multi-index vector and @f$ \Sigma = \text{diag}(\sigma_l)^2 @f$.
 * For example, let @f$ \sigma_l^{-1} = 3 @f$ and @f$ M = 4 @f$ frequencies.
 * We split the normal distribution into 4 intervals on each side of the origin, as shown in the plot below.
 * We only highlight the positive side, as the negative side is symmetric.
 * Then, we sum each interval with its symmetric counterpart to get the weights.
 * @plot
 * {
 * "f": "(x) => (1 / (3.0 * Math.sqrt(2 * Math.PI))) * Math.exp(-((x - 0.0) ** 2) / (2 * 3.0 ** 2))",
 * "xBounds": [-9, 9],
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
 * {"x": 2.57142, "yMax": 0.005, "c": "transparent", "s": "solid", "label": "ω₁"},
 * {"x": 5.14285, "yMax": 0.005, "c": "transparent", "s": "solid", "label": "ω₂"},
 * {"x": 7.71428, "yMax": 0.005, "c": "transparent", "s": "solid", "label": "ω₃"},
 * {"x": 1.28571, "s": "dash"}, {"x": 3.85714, "s": "dash"}, {"x": 6.42857, "s": "dash"}, {"x": 9, "s": "dash"},
 * {"x": -1.28571, "s": "dash"}, {"x": -3.85714, "s": "dash"}, {"x": -6.42857, "s": "dash"}, {"x": -9, "s": "dash"}],
 * "annotations": [
 *  {"x": 0.64285, "y": 0.1, "text": "w₀²/2"},
 *  {"x": 2.57142, "y": 0.06, "text": "w₁²/2"},
 *  {"x": 5.14285, "y": 0.02, "text": "w₂²/2"},
 *  {"x": 7.71428, "y": 0.003, "text": "w₃²/2"}
 *  ]
 * }
 * @endplot
 * @see TruncatedFourierFeatureMap
 */
class LinearTruncatedFourierFeatureMap final : public TruncatedFourierFeatureMap {
 public:
  LinearTruncatedFourierFeatureMap(int num_frequencies, ConstVectorRef sigma_l, Scalar sigma_f,
                                   const RectSet& X_bounds);
  LinearTruncatedFourierFeatureMap(int num_frequencies, double sigma_l, Scalar sigma_f, const RectSet& X_bounds);

  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override;

  /**
   * Return the periodic input domain for this linear truncated Fourier map.
   * Given a linear truncated feature map @phiMx, its smallest frequency is
   * @f[
   * \frac{6\sigma_l^{-1}}{2M-1} P(x) ,
   * @f]
   * where @f$ P(x) @f$ is the projection to the unit hypercube of the original domain and
   * @f$ M @f$ is the number of frequencies per dimension (including the zero frequency).
   * We want to find the upper bound of the periodic domain @f$ \bar{x} @f$ such that
   * @f[
   * \frac{6\sigma_l^{-1}}{2M-1} P(\bar{x}) = 2\pi .
   * @f]
   * Since @f$ P(x) = \frac{x - x_{min}}{x_{max} - x_{min}} @f$, we have that
   * @f[
   * \bar{x} = x_{min} + 2\pi \frac{2M-1}{6\sigma_l^{-1}} (x_{max} - x_{min}) .
   * @f]
   * Graphically,
   * @code{.unparsed}
   *                                New max (x̄)
   *   ┌───────────────────────────────●
   *   │                               │
   *   │                               │
   *   │                     Old max   │
   *   ├───────────────────────●       │
   *   │                       │       │
   *   │                       │       │
   *   │                       │       │
   *   │                       │       │
   *   │ Min                   │       │
   *   ●───────────────────────┴───────┘
   * @endcode
   * Notice how the lower bound remains fixed, while the upper bound is shifted to create the periodic domain.
   * @note The periodic domain could be smaller than the original domain, depending on the values of @f$ \sigma_l @f$.
   * @pre @sigmal must have the same dimension as the input space.
   * @pre All values in @sigmal must be greater than 0.
   * @return new RectSet representing the periodic input domain
   */
  [[nodiscard]] RectSet get_periodic_set() const override;

  [[nodiscard]] std::string to_string() const override;
};

std::ostream& operator<<(std::ostream& os, const LinearTruncatedFourierFeatureMap& f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::LinearTruncatedFourierFeatureMap)

#endif  // LUCID_INCLUDE_FMT
