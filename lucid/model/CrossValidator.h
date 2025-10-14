/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * CrossValidator class.
 */
#pragma once

#include "lucid/model/Estimator.h"

namespace lucid {

class CrossValidator {
 public:
  virtual ~CrossValidator() = default;
  void fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
           const Tuner* tuner) const;
  void fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const;
  void fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
           const Tuner& tuner) const;

 private:
  [[nodiscard]] virtual std::vector<std::vector<bool>> compute_folds(ConstMatrixRef training_inputs) const = 0;
};

}  // namespace lucid
