/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Regression class.
 */
#pragma once

#include <iosfwd>

#include "lucid/lib/eigen.h"
#include "lucid/math/FeatureMap.h"

namespace lucid {

/**
 * Given two vector spaces @f$ \mathcal{X}, \mathcal{Y} @f$ and a map @f$ f: \mathcal{X} \to \mathcal{Y} @f$,
 * the goal is to produce a model @f$ f^*:\mathcal{X} \to \mathcal{Y} @f$ that best approximates @f$ f @f$.
 */
class Regression {
 public:
  Regression() = default;
  Regression(const Regression&) = default;
  Regression(Regression&&) = default;
  Regression& operator=(const Regression&) = default;
  Regression& operator=(Regression&&) = default;
  virtual ~Regression() = default;
  /**
   * A model is a function that takes a @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X}
   * @f$ and returns a @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$.
   * @param x @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
   * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
   */
  [[nodiscard]] virtual Matrix operator()(ConstMatrixRef x) const = 0;
  [[nodiscard]] virtual Matrix operator()(ConstMatrixRef x, const FeatureMap& feature_map) const = 0;
};

std::ostream& operator<<(std::ostream& os, const Regression&);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Regression)

#endif
