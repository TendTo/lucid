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

double tune_internal(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                     std::span<const ParameterValues> parameters, std::vector<size_t>& best_parameters_indices);

template <class T, class R>
std::function<R()> tune_cb(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                           const std::span<const ParameterValues> parameters,
                           std::vector<size_t>& best_parameters_indices) {
  // Keep track of the best score
  // For each value in the currently fixed parameter
  return [&estimator, &training_inputs, &training_outputs, parameters, &best_parameters_indices]() {
    double best_score = -1.0;
    LUCID_ASSERT(std::holds_alternative<std::vector<T>>(parameters.front().values()),
                 "Parameter values are not of the expected type");
    const std::vector<T>& values = std::get<std::vector<T>>(parameters.front().values());

    for (std::size_t i = 0; i < values.size(); ++i) {
      // fix the parameter value
      estimator.set(parameters.front().parameter(), values[i]);
      // then tune the rest of the parameters
      if (const double score =
              tune_internal(estimator, training_inputs, training_outputs,
                            // Pass the rest of the parameters to the next iteration or an empty span if there are none
                            parameters.size() > 1 ? parameters.subspan(1) : std::span<const ParameterValues>{},
                            best_parameters_indices);
          score > best_score) {
        // If the new score is better than the best score, update the best score and keep track of the parameter index
        best_score = score;
        best_parameters_indices.at(best_parameters_indices.size() - parameters.size()) = i;
      }
    }
    return best_score;
  };
}

double tune_internal(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                     const std::span<const ParameterValues> parameters,
                     std::vector<std::size_t>& best_parameters_indices) {
  // Consolidate and evaluate the estimator with the current parameter values, producing a score
  if (parameters.empty()) {
    estimator.consolidate(training_inputs, training_inputs);
    return estimator.score(training_inputs, training_outputs);
  }
  return dispatch(
      parameters.front().parameter(),
      tune_cb<int, double>(estimator, training_inputs, training_outputs, parameters, best_parameters_indices),
      tune_cb<double, double>(estimator, training_inputs, training_outputs, parameters, best_parameters_indices),
      tune_cb<Vector, double>(estimator, training_inputs, training_outputs, parameters, best_parameters_indices));
}

}  // namespace

GridSearchTuner::GridSearchTuner(std::vector<ParameterValues> parameters) : parameters_{std::move(parameters)} {}

void GridSearchTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                                ConstMatrixRef training_outputs) const {
  // Keep track of the best value for each parameter
  std::vector<std::size_t> best_parameters_indices(parameters_.size());
  // Start the tuning process
  tune_internal(estimator, training_inputs, training_outputs, parameters_, best_parameters_indices);
  // Set the best parameter values in the estimator
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const auto& parameter = parameters_[i];
    estimator.set(parameter.parameter(), best_parameters_indices[i], parameter.values());
  }
}

}  // namespace lucid
