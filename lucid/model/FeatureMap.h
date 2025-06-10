/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FeatureMap class.
 */
#pragma once

#include <iosfwd>

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
   * @return @f$ n \times N @f$ output, where @f$ N @f$ is the dimension of the feature space
   */
  [[nodiscard]] virtual Matrix operator()(ConstMatrixRef x) const = 0;
};

std::ostream &operator<<(std::ostream &os, const FeatureMap &f);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::FeatureMap)

#endif
