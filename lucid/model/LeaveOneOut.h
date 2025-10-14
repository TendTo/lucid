/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LeaveOneOut class.
 */
#pragma once

#include "lucid/model/CrossValidator.h"

namespace lucid {

class LeaveOneOut final : public CrossValidator {
 private:
  [[nodiscard]] std::vector<std::vector<bool>> compute_folds(ConstMatrixRef training_inputs) const override;
};

}  // namespace lucid
