/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GridSearchTuner class.
 */
#include "lucid/model/GridSearchTuner.h"

#include <iostream>

#include "lucid/model/Estimator.h"
#include "lucid/model/ParameterValue.h"
#include "lucid/model/ParameterValues.h"
#include "lucid/util/error.h"

namespace lucid {

namespace {

/**
 * Utility class to perform grid search tuning on an Estimator avoining changes to the tuner state.
 * It recursively fixes the first parameter and tunes the rest, keeping track of the best parameter indices.
 */
class GridSearchTuning {
 public:
  /**
   * Create a new GridSearchTuning object to tune the given `estimator` with the provided training inputs and outputs.
   * @param estimator estimator to tune
   * @param training_inputs training inputs used for tuning
   * @param training_outputs training outputs used for tuning
   */
  GridSearchTuning(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs)
      : estimator_{estimator},
        training_inputs_{training_inputs},
        training_outputs_{training_outputs},
        best_parameters_indices_{},
        best_score_{std::numeric_limits<double>::min()} {
    LUCID_CHECK_ARGUMENT_EXPECTED(training_inputs.rows() == training_outputs.rows(), "training_inputs.rows()",
                                  training_inputs.rows(), "== training_outputs.rows()");
  }

  /**
   * Tune the estimator with the provided parameters.
   * @param parameters span of the parameters to tune
   */
  void tune(const std::span<const ParameterValues> parameters) {
    // Resize the vector to hold the best parameter indices for each parameter
    best_parameters_indices_.resize(parameters.size());
    current_parameters_indices_ = std::vector<std::size_t>(parameters.size(), 0);
    // Start the tuning process
    tune_internal(parameters);
    LUCID_DEBUG_FMT("Best score: {}", best_score_);
    LUCID_DEBUG_FMT("Best parameters indices: {}", best_parameters_indices_);
  }

  /** @getter{estimator, tuning process} */
  [[nodiscard]] const Estimator& estimator() const { return estimator_; }
  /** @getter{training inputs, tuning process} */
  [[nodiscard]] ConstMatrixRef training_inputs() const { return training_inputs_; }
  /** @getter{training outputs, tuning process} */
  [[nodiscard]] ConstMatrixRef training_outputs() const { return training_outputs_; }
  /** @getter{best parameters indices, tuning process} */
  [[nodiscard]] const std::vector<std::size_t>& best_parameters_indices() const { return best_parameters_indices_; }
  /** @getter{best score, tuning process} */
  [[nodiscard]] double best_score() const { return best_score_; }

 private:
  /**
   * Utility function to create a callback for tuning a specific parameter type.
   * @tparam T type of the parameter values to tune
   * @tparam R return type of the tuning function
   * @param parameters span of the parameters to tune
   * @return a function that tunes the parameter and returns the best score
   */
  template <class T, class R>
  std::function<R()> tune_cb(const std::span<const ParameterValues> parameters) {
    // Keep track of the best score
    // For each value in the currently fixed parameter
    return [this, parameters]() {
      LUCID_ASSERT(!parameters.empty(), "Empty span should be captured by the base case of the recursion");
      LUCID_ASSERT(std::holds_alternative<std::vector<T>>(parameters.front().values()),
                   "Parameter values are not of the expected type");
      const std::vector<T>& values = std::get<std::vector<T>>(parameters.front().values());

      for (std::size_t i = 0; i < values.size(); ++i) {
        // fix the parameter value
        estimator_.set(parameters.front().parameter(), values[i]);
        current_parameters_indices_.at(best_parameters_indices_.size() - parameters.size()) = i;
        // Tune the rest of the parameters in the next iteration or just pass an empty span if there are none
        tune_internal(parameters.size() > 1 ? parameters.subspan(1) : std::span<const ParameterValues>{});
      }
    };
  }

  /**
   * Internal method to perform the tuning process.
   * It fixes the first parameter and recursively tunes the remaining ones.
   * When there are no longer parameters to change,
   * it consolidates the estimator with the current parameter values and evaluates it, returning the score.
   * @param parameters span of the parameters to tune
   * @return the best score achieved with the current parameter values
   */
  void tune_internal(const std::span<const ParameterValues> parameters) {
    // Base case: consolidate and evaluate the estimator with the current parameter values, producing a score
    if (parameters.empty()) {
      estimator_.consolidate(training_inputs_, training_outputs_);
      const double score = estimator_.score(training_inputs_, training_outputs_);
      // If the new score is better than the best score, update the best score and keep track of the parameter index
      LUCID_DEBUG_FMT("Consolidated with parameters: {}, score: {}", current_parameters_indices_, score);
      if (score > best_score_) {
        LUCID_DEBUG_FMT("New best score: {} > {}", score, best_score_);
        best_score_ = score;
        best_parameters_indices_ = current_parameters_indices_;
      }
      return;
    }

    // Recursive case: fix the first parameter and tune the rest
    dispatch(parameters.front().parameter(), tune_cb<int, void>(parameters), tune_cb<double, void>(parameters),
             tune_cb<Vector, void>(parameters));
  }

  Estimator& estimator_;                                 ///< Estimator to tune
  ConstMatrixRef training_inputs_;                       ///< Training inputs used for tuning
  ConstMatrixRef training_outputs_;                      ///< Training inputs and outputs used for tuning
  std::vector<std::size_t> best_parameters_indices_;     ///< Indices of the best parameter with respect to scoring
  std::vector<std::size_t> current_parameters_indices_;  ///< Indices of the current parameters being tuned
  double best_score_;  ///< Best score achieved during the tuning process, initialized to a very low value
};

}  // namespace

GridSearchTuner::GridSearchTuner(std::vector<ParameterValues> parameters) : parameters_{std::move(parameters)} {}

void GridSearchTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                                ConstMatrixRef training_outputs) const {
  // Start the tuning process
  GridSearchTuning tuning{estimator, training_inputs, training_outputs};
  tuning.tune(parameters_);
  // Set the best parameter values in the estimator
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const auto& parameter = parameters_[i];
    estimator.set(parameter.parameter(), tuning.best_parameters_indices()[i], parameter.values());
  }
}

}  // namespace lucid
