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

namespace lucid {

/**
 * A model is a function that takes a @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X} @f$
 * and returns a @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$.
 * @param input @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
 * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
 */
using Model = std::function<Matrix(ConstMatrixRef input)>;

/**
 * Given two vector spaces @f$ \mathcal{X}, \mathcal{Y} @f$ and a map @f$ f: \mathcal{X} \to \mathcal{Y} @f$,
 * the goal is to produce a model @f$ f^*:\mathcal{X} \to \mathcal{Y} @f$ that best approximates @f$ f @f$.
 */
class Regression {
 public:
  virtual ~Regression() = default;

  /** @getter{model that is the current best approximation of @f$ f @f$, regression} */
  [[nodiscard]] const Model& model() const { return model_; }

  /**
   * Apply the model to a matrix of input vectors.
   * @param x @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
   * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
   */
  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const { return model_(x); }

 protected:
  Model model_;  ///< Predictor function
};

std::ostream& operator<<(std::ostream& os, const Regression&);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Regression)

#endif
