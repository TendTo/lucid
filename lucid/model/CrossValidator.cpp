/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * CrossValidator class.
 */
#include "lucid/model/CrossValidator.h"

#include "lucid/util/logging.h"

namespace lucid {

void CrossValidator::fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                         const Tuner* tuner) const {
  LUCID_TRACE_FMT("({}, {}, {})", estimator, LUCID_FORMAT_MATRIX(training_inputs),
                  LUCID_FORMAT_MATRIX(training_outputs));
  double best_score = -std::numeric_limits<double>::infinity();
  std::unique_ptr<Estimator> best_estimator{estimator.clone()};
  for (const auto& fold : compute_folds(training_inputs)) {
    auto train_inputs{training_inputs(fold, Eigen::all)};
    auto train_outputs{training_outputs(fold, Eigen::all)};
    auto val_inputs{training_inputs(fold, Eigen::all)};
    auto val_outputs{training_outputs(fold, Eigen::all)};

    if (tuner) {
      estimator.fit(train_inputs, train_outputs, *tuner);
    } else {
      estimator.fit(train_inputs, train_outputs);
    }

    if (const double new_score = estimator.score(val_inputs, val_outputs); new_score > best_score) {
      LUCID_DEBUG_FMT("New best score: {} > {}", new_score, best_score);
      LUCID_DEBUG_FMT("New best estimator: {}, {}", estimator, *best_estimator);
      best_score = new_score;
      best_estimator = estimator.clone();
    }
  }
  estimator = *best_estimator;
  LUCID_TRACE_FMT("=> {}", estimator);
}

void CrossValidator::fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const {
  fit(estimator, training_inputs, training_outputs, nullptr);
}
void CrossValidator::fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                         const Tuner& tuner) const {
  fit(estimator, training_inputs, training_outputs, &tuner);
}

}  // namespace lucid
