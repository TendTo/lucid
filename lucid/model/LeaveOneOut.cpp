/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LeaveOneOut class.
 */
#include "lucid/model/LeaveOneOut.h"

#include "lucid/util/logging.h"

namespace lucid {

std::vector<std::vector<bool>> LeaveOneOut::compute_folds(ConstMatrixRef training_inputs) const {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(training_inputs));
  std::vector<std::vector<bool>> folds(training_inputs.rows());
  for (Index i = 0; i < training_inputs.rows(); ++i) {
    folds[i].resize(training_inputs.rows(), false);
    folds[i][i] = false;
  }
  LUCID_DEBUG_FMT("=> {}", folds);
  return folds;
}

}  // namespace lucid
