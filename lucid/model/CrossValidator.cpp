/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * CrossValidator class.
 */
#include "lucid/model/CrossValidator.h"

#include <iostream>

#include "lucid/util/logging.h"

namespace lucid {

double CrossValidator::fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                           const Tuner* tuner, const scorer::Scorer& scorer) const {
  LUCID_TRACE_FMT("({}, {}, {}, {}, {})", estimator, LUCID_FORMAT_MATRIX(training_inputs),
                  LUCID_FORMAT_MATRIX(training_outputs), tuner == nullptr ? "no_tuner" : "with_tuner",
                  scorer == nullptr ? "default_scorer" : "custom_scorer");

  std::unique_ptr<Estimator> best_estimator;
  double best_score = -std::numeric_limits<double>::infinity();

  const auto [train_folds, val_folds] = compute_folds(training_inputs);
  for (std::size_t i = 0; i < train_folds.size(); ++i) {
    LUCID_DEBUG_FMT("Cross validation progress {}/{}", i + 1, train_folds.size());
    std::unique_ptr<Estimator> current_estimator{estimator.clone()};

    auto train_inputs{training_inputs(train_folds[i], Eigen::all)};
    auto train_outputs{training_outputs(train_folds[i], Eigen::all)};
    auto val_inputs{training_inputs(val_folds[i], Eigen::all)};
    auto val_outputs{training_outputs(val_folds[i], Eigen::all)};

    if (tuner) {
      try {
        current_estimator->fit(train_inputs, train_outputs, *tuner);
      } catch (const std::runtime_error& e) {
        LUCID_ERROR_FMT("Tuning failed: {}", e.what());
        continue;
      }
    } else {
      current_estimator->fit(train_inputs, train_outputs);
    }

    const double new_score = scorer == nullptr ? current_estimator->score(val_inputs, val_outputs)
                                               : scorer(*current_estimator, val_inputs, val_outputs);
    if (new_score > best_score) {
      LUCID_DEBUG_FMT("New best score: {} > {}", new_score, best_score);
      LUCID_DEBUG_FMT("New best estimator: {}", *current_estimator);
      best_score = new_score;
      best_estimator = current_estimator->clone();
    } else {
      LUCID_WARN_FMT("Score did not improve: {} <= {}", new_score, best_score);
      LUCID_WARN_FMT("Current estimator: {}", *current_estimator);
    }
  }

  estimator = *best_estimator;
  estimator.load(*best_estimator);  // This is needed to ensure all parameters are copied
  LUCID_TRACE_FMT("=> {} produced by {}", best_score, estimator);
  return best_score;
}

double CrossValidator::fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                           const scorer::Scorer& scorer) const {
  return fit(estimator, training_inputs, training_outputs, nullptr, scorer);
}

double CrossValidator::fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                           const Tuner& tuner, const scorer::Scorer& scorer) const {
  return fit(estimator, training_inputs, training_outputs, &tuner, scorer);
}

std::vector<double> CrossValidator::score(const Estimator& estimator, ConstMatrixRef inputs, ConstMatrixRef outputs,
                                          const scorer::Scorer& scorer) const {
  LUCID_TRACE_FMT("({}, {}, {}, {})", estimator, LUCID_FORMAT_MATRIX(inputs), LUCID_FORMAT_MATRIX(outputs),
                  scorer == nullptr ? "default_scorer" : "custom_scorer");

  const std::unique_ptr<Estimator> estimator_copy{estimator.clone()};
  std::vector<double> scores;
  const auto [train_folds, val_folds] = compute_folds(inputs);
  scores.reserve(train_folds.size());

  for (std::size_t i = 0; i < train_folds.size(); ++i) {
    LUCID_DEBUG_FMT("Cross validation scoring progress {}/{}", i + 1, train_folds.size());

    auto train_inputs{inputs(train_folds[i], Eigen::all)};
    auto train_outputs{outputs(train_folds[i], Eigen::all)};
    auto val_inputs{inputs(val_folds[i], Eigen::all)};
    auto val_outputs{outputs(val_folds[i], Eigen::all)};

    estimator_copy->consolidate(train_inputs, train_outputs);

    const double new_score = scorer == nullptr ? estimator_copy->score(val_inputs, val_outputs)
                                               : scorer(*estimator_copy, val_inputs, val_outputs);
    scores.push_back(new_score);
  }

  LUCID_TRACE_FMT("=> {}", scores);
  return scores;
}

}  // namespace lucid
