/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LeaveOneOut class.
 */
#include "lucid/model/LeaveOneOut.h"

#include <ostream>
#include <utility>

#include "lucid/util/logging.h"

namespace lucid {

std::pair<LeaveOneOut::SliceSelector, LeaveOneOut::SliceSelector> LeaveOneOut::compute_folds(
    ConstMatrixRef training_inputs) const {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(training_inputs));
  SliceSelector train_folds(training_inputs.rows());
  SliceSelector val_folds(training_inputs.rows());
  for (Index i = 0; i < training_inputs.rows(); ++i) {
    train_folds[i].resize(training_inputs.rows() - 1);
    val_folds[i].resize(1);
    for (Index j = 0, k = 0; j < training_inputs.rows(); ++j) {
      if (j == i) {
        val_folds[i][0] = j;
      } else {
        train_folds[i][k++] = j;
      }
    }
  }
  LUCID_DEBUG_FMT("=> ({}, {})", train_folds, val_folds);
  return {train_folds, val_folds};
}

std::string LeaveOneOut::to_string() const { return "LeaveOneOut( )"; }

std::ostream& operator<<(std::ostream& os, const LeaveOneOut& lo) { return os << lo.to_string(); }

}  // namespace lucid
