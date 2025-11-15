/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FeatureMap class.
 */
#pragma once

#include <iosfwd>
#include <memory>

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Map input vectors to a feature space.
 * The output space usually has a higher dimensionality than the input space.
 */
class FeatureMap {
 public:
  explicit FeatureMap() = default;
  FeatureMap(const FeatureMap &) = default;
  FeatureMap(FeatureMap &&) = default;
  FeatureMap &operator=(const FeatureMap &) = default;
  FeatureMap &operator=(FeatureMap &&) = default;
  virtual ~FeatureMap() = default;

  /**
   * Apply the feature map to a vector.
   * @param x @nxd input vector
   * @return @f$ n \times M @f$ output, where @f$ M @f$ is the dimension of the feature space
   */
  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const;

  /**
   * Apply the inverse feature map to a vector.
   * @param y @f$ n \times M @f$ input vector in the feature space
   * @return @nxd output in the original space
   */
  [[nodiscard]] Matrix invert(ConstMatrixRef y) const;

  /**
   * Clone the feature map.
   * Create a new instance of the feature map with the same parameters.
   * @return new instance of the feature map
   */
  [[nodiscard]] virtual std::unique_ptr<FeatureMap> clone() const = 0;

 protected:
  /**
   * Concrete implementation of @ref operator()().
   * @param x @nxd input vector
   * @return @f$ n \times M @f$ output, where @f$ M @f$ is the dimension of the feature space
   */
  [[nodiscard]] virtual Matrix apply_impl(ConstMatrixRef x) const = 0;
  /**
   * Concrete implementation of @ref invert().
   * @param y @f$ n \times M @f$ input vector in the feature space
   * @return @nxd output in the original space
   */
  [[nodiscard]] virtual Matrix invert_impl(ConstMatrixRef y) const = 0;
};

std::ostream &operator<<(std::ostream &os, const FeatureMap &f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::FeatureMap)

#endif
