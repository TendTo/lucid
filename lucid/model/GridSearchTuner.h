/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GridSearchTuner class.
 */
#pragma once

#include <concepts>
#include <vector>

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/LinearTruncatedFourierFeatureMap.h"
#include "lucid/model/LogTruncatedFourierFeatureMap.h"
#include "lucid/model/ParameterValues.h"
#include "lucid/model/RectSet.h"
#include "lucid/model/Tuner.h"
#include "lucid/util/concept.h"

namespace lucid {

/**
 * Grid search tuning strategy for model hyperparameter optimisation.
 * The GridSearchTuner is responsible for iterating over a predefined grid of hyperparameter combinations
 * to optimise the performance of a given estimator.
 * This tuner is suitable for scenarios where exhaustive search over a finite hyperparameter space is required.
 */
class GridSearchTuner final : public Tuner {
 public:
  /**
   * Construct a new GridSearchTuner object with the given `parameters`.
   * @param parameters parameters to be tuned, with the values to be tested
   * @param n_jobs number of parallel jobs to run during tuning. If set to 0, it defaults to max(1, CPU cores - 2).
   */
  explicit GridSearchTuner(std::vector<ParameterValues> parameters, std::size_t n_jobs = 0);

  /** @getter{number of parallel jobs, grid search} */
  [[nodiscard]] std::size_t n_jobs() const { return n_jobs_; }
  /** @getter{parameters, grid search} */
  [[nodiscard]] const std::vector<ParameterValues>& parameters() const { return parameters_; }

  template <
      IsAnyOf<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap, LogTruncatedFourierFeatureMap> T>
  void tune(Estimator& estimator, ConstMatrixRef training_inputs, const ConstMatrixRef& training_outputs,
            int num_frequencies, const RectSet& x_limits) const;
  template <
      IsAnyOf<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap, LogTruncatedFourierFeatureMap> T>
  void tune_online(Estimator& estimator, ConstMatrixRef training_inputs, const OutputComputer& training_outputs,
                   int num_frequencies, const RectSet& x_limits) const;

 private:
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                 const OutputComputer& training_outputs) const override;

  template <class T>
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs, const OutputComputer& training_outputs,
                 int num_frequencies, const RectSet& x_limits) const;

  std::size_t n_jobs_;                         ///< Number of parallel jobs to run during tuning
  std::vector<ParameterValues> parameters_;    ///< List of parameter values to be tuned, with the values to be tested
  std::vector<Index> parameters_max_indices_;  ///< Maximum indices for each parameter, used to iterate over the grid
};

std::ostream& operator<<(std::ostream& os, const GridSearchTuner& tuner);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::GridSearchTuner)

#endif
