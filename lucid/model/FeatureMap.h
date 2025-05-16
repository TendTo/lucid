/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FeatureMap class.
 */
#pragma once

#include "lucid/lib/eigen.h"

namespace lucid {

class FeatureMap {
 public:
  virtual ~FeatureMap() = default;

  /**
   * Given an @nxd dimensional matrix @x, project each row vector to the unit hypercube @f$ [0, 1]^d @f$,
   * then compute the feature map.
   * @param x input vector
   * @return @f$ n \times 2 M + 1 @f$ dimensional feature map
   */
  [[nodiscard]] virtual Matrix operator()(ConstMatrixRef x) const = 0;
};

}  // namespace lucid
