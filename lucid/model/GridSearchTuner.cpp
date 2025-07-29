/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * GridSearchTuner class.
 */
#include "lucid/model/GridSearchTuner.h"

#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "lucid/model/Estimator.h"
#include "lucid/model/ParameterValue.h"
#include "lucid/model/ParameterValues.h"
#include "lucid/util/IndexIterator.h"
#include "lucid/util/error.h"

namespace lucid {

namespace {

struct Null {};
const RectSet x_limits_empty{{0}, {1}};  ///< Default empty limits for the input space

/**
 * Utility class to perform grid search tuning on an Estimator in parallel.
 * It captures all the relevant data for the task and then launches a future to solve the tuning process.
 * @todo The mutexes used here are not the most efficient.
 * They force synchronization on every index increment and score update.
 * While that is sure not to be the bottleneck in all real-world scenarios,
 * we could consider splitting the tuning process a-priori and let the main thread update the score if needed.
 */
template <
    IsAnyOf<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap, LogTruncatedFourierFeatureMap, Null>
        T>
class GridSearchTuning {
 public:
  /**
   * Create a new GridSearchTuning object to tune the given `estimator` with the provided training inputs and outputs.
   * @param index_mutex mutex protecting the access to the index iterator
   * @param score_mutex mutex protecting the access to the score updating
   * @param estimator estimator to tune
   * @param parameters parameters to tune
   * @param training_inputs training inputs used for tuning
   * @param training_outputs training outputs used for tuning
   * @param best_parameters_indices indices of the parameter values that produce the best score
   * @param it index interator going over all combinations of parameter values
   * @param best_score current best score found
   * @param num_frequencies number of frequencies to use in the estimator
   * @param x_limits limits of the input space for the estimator
   */
  GridSearchTuning(std::mutex& index_mutex, std::mutex& score_mutex, Estimator& estimator,
                   const std::vector<ParameterValues>& parameters, ConstMatrixRef training_inputs,
                   const OutputComputer& training_outputs, std::vector<Index>& best_parameters_indices,
                   IndexIterator<std::vector<Index>>& it, double& best_score, const int num_frequencies,
                   const RectSet& x_limits)
      : index_mutex_{index_mutex},
        score_mutex_{score_mutex},
        estimator_{estimator},
        parameters_{parameters},
        training_inputs_{training_inputs},
        training_outputs_{training_outputs},
        best_parameters_indices_{best_parameters_indices},
        it_{it},
        best_score_{best_score},
        current_parameters_indices_{},
        num_frequencies_{num_frequencies},
        x_limits_{x_limits} {}

  /**
   * Launch the tuning process.
   * Should be followed by a call to @ref wait.
   **/
  void launch() {
    future_ = std::async(std::launch::async, [this]() { tune(); });
  }

  /**
   * Wait for the tuning to end.
   * @pre @ref launch should have been called before this method.
   **/
  void wait() {
    LUCID_ASSERT(future_.valid(), "Future is not valid. Did you call launch()?");
    future_.wait();  // Wait for the tuning process to complete
  }

 private:
  /** Tune the estimator */
  void tune() {
    // Iterate over all possible combinations of parameter values
    while (increase_index()) {
      double score = std::numeric_limits<double>::lowest();
      if constexpr (std::is_same_v<T, Null>) {
        auto training_outputs = training_outputs_(estimator_, training_inputs_);
        estimator_.consolidate(training_inputs_, training_outputs);
        score = estimator_.score(training_inputs_, training_outputs);
      } else {
        T feature_map{num_frequencies_, estimator_.get<Parameter::SIGMA_L>(), estimator_.get<Parameter::SIGMA_F>(),
                      x_limits_};
        Matrix training_outputs{feature_map(training_outputs_(estimator_, training_inputs_))};
        estimator_.consolidate(training_inputs_, training_outputs);
        score = estimator_.score(training_inputs_, training_outputs);
      }

      // If the new score is better than the best score, update the best score and keep track of the parameter index
      LUCID_TRACE_FMT("parameters = {}, score = {}", current_parameters_indices_, score);
      update_score(score);
    }
    LUCID_TRACE("Stopping");
  }

  /**
   * Increase the index iterator to go to the next parameter combination
   * @note This is a critical zone. Therefore, it is protected by the mutex.
   * @return true if there are still parameter combinations to explore
   * @return false if all possible parameter combinations have been exhausted; the process should stop
   */
  bool increase_index() {
    std::lock_guard<std::mutex> lock(index_mutex_);
    // Increment the index of the current parameter
    if (!it_) return false;  // No more indices to increment. The tuning is complete.
    for (std::size_t i = 0; i < parameters_.size(); ++i) {
      const ParameterValues& parameter = parameters_[i];
      estimator_.set(parameter.parameter(), it_[i], parameter.values());
    }
    current_parameters_indices_ = it_.indexes();
    ++it_;
    return true;
  }
  /**
   * Update the best score and parameter indices if the current score is better.
   * @note This is a critical zone. Therefore, it is protected by the mutex.
   * @param score new score found
   */
  void update_score(double score) {
    std::lock_guard<std::mutex> lock(score_mutex_);
    // Update the best score if the new score is better
    if (score > best_score_) {
      LUCID_DEBUG_FMT("New best score: {} > {}", score, best_score_);
      best_score_ = score;
      best_parameters_indices_ = current_parameters_indices_;
    }
  }

  std::future<void> future_;  ///< Future to hold the asynchronous tuning process
  std::mutex& index_mutex_;   ///< Mutex to protect access to the index iterator
  std::mutex& score_mutex_;   ///< Mutex to protect access to the score during tuning
  Estimator& estimator_;      ///< Estimator to tune
  const std::vector<ParameterValues>&
      parameters_;                               ///< List of parameter values to be tuned, with the values to be tested
  ConstMatrixRef training_inputs_;               ///< Training inputs used for tuning
  const OutputComputer& training_outputs_;       ///< Training outputs used for tuning, computed every iteration
  std::vector<Index>& best_parameters_indices_;  ///< Indices of the best parameter with respect to scoring
  IndexIterator<std::vector<Index>>& it_;        ///< Index iterator for iterating over parameter values
  double& best_score_;  ///< Best score achieved during the tuning process, initialized to a very low value
  std::vector<Index> current_parameters_indices_;  ///< Current parameter indices being tested
  int num_frequencies_;                            ///< Number of frequencies for the estimator, if applicable
  const RectSet& x_limits_;                        ///< Limits of the input space for the estimator, if applicable
};

}  // namespace

GridSearchTuner::GridSearchTuner(std::vector<ParameterValues> parameters, const std::size_t n_jobs)
    : n_jobs_{n_jobs > 0 ? n_jobs
                         : (std::thread::hardware_concurrency() > 2 ? std::thread::hardware_concurrency() - 2 : 1)},
      parameters_{std::move(parameters)},
      parameters_max_indices_{} {
  LUCID_ASSERT(n_jobs_ > 0, "The number of jobs must be greater than 0.");
  std::ranges::transform(parameters_, std::back_inserter(parameters_max_indices_),
                         [](const ParameterValues& p) { return static_cast<Index>(p.size()); });
  LUCID_ASSERT(parameters_.size() == parameters_max_indices_.size(),
               "The number of parameters must match the number of maximum indices.");
}

void GridSearchTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                                const OutputComputer& training_outputs) const {
  tune_impl<Null>(estimator, training_inputs, training_outputs, 0, x_limits_empty);
}

template <
    IsAnyOf<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap, LogTruncatedFourierFeatureMap> T>
void GridSearchTuner::tune(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                           int num_frequencies, const RectSet& x_limits) const {
  tune_impl<T>(
      estimator, training_inputs, [training_outputs](const Estimator&, ConstMatrixRef) { return training_outputs; },
      num_frequencies, x_limits);
}
template <
    IsAnyOf<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap, LogTruncatedFourierFeatureMap> T>
void GridSearchTuner::tune_online(Estimator& estimator, ConstMatrixRef training_inputs,
                                  const OutputComputer& training_outputs, const int num_frequencies,
                                  const RectSet& x_limits) const {
  tune_impl<T>(estimator, training_inputs, training_outputs, num_frequencies, x_limits);
}

template <class T>
void GridSearchTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                                const OutputComputer& training_outputs, const int num_frequencies,
                                const RectSet& x_limits) const {
  // Mutex to protect access to the best parameter indices during tuning
  std::mutex index_mutex, score_mutex;
  // Prepare the shared data: the best score and the best parameter indices and the index iterator
  double best_score = std::numeric_limits<double>::min();
  std::vector<Index> best_parameters_indices(parameters_.size(), 0);
  IndexIterator<std::vector<Index>> it{parameters_max_indices_};

  // Create a vector of n_jobs GridSearchTuning objects
  std::vector<GridSearchTuning<T>> tuners;
  tuners.reserve(parameters_.size());
  std::vector<std::unique_ptr<Estimator>> estimators;
  estimators.reserve(n_jobs_ - 1);
  for (std::size_t i = 0; i < n_jobs_; ++i) {
    if (i > 0) estimators.emplace_back(estimator.clone());
    tuners.emplace_back(index_mutex, score_mutex, i == 0 ? estimator : *estimators.back(), parameters_, training_inputs,
                        training_outputs, best_parameters_indices, it, best_score, num_frequencies, x_limits);
  }

  LUCID_ASSERT(tuners.size() == n_jobs_, "The number of tuners must match the number of jobs.");
  LUCID_ASSERT(estimators.size() == n_jobs_ - 1, "The number of tuners must match the number of jobs - 1.");

  LUCID_TRACE_FMT("jobs = {}", n_jobs_);

  // Launch the tuning process for each parameter in parallel
  for (auto& tuner : tuners) tuner.launch();

  // Wait for all tuning processes to complete
  for (auto& tuner : tuners) tuner.wait();

  LUCID_DEBUG_FMT("best_parameters_idxs = {}, best_score = {}", best_parameters_indices, best_score);
  // Set the best parameter values in the estimator
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const auto& parameter = parameters_[i];
    estimator.set(parameter.parameter(), best_parameters_indices[i], parameter.values());
  }
  if constexpr (std::is_same_v<T, Null>) {
    // If no feature map is used, we can directly consolidate the estimator
    estimator.consolidate(training_inputs, training_outputs(estimator, training_inputs));
  } else {
    T feature_map{num_frequencies, estimator.get<Parameter::SIGMA_L>(), estimator.get<Parameter::SIGMA_F>(), x_limits};
    estimator.consolidate(training_inputs, feature_map(training_outputs(estimator, training_inputs)));
  }
}

std::ostream& operator<<(std::ostream& os, const GridSearchTuner& tuner) {
  return os << "GridSearchTuner( parameters( " << fmt::format("{}", tuner.parameters()) << " ) n_jobs( "
            << tuner.n_jobs() << " )";
}

template void GridSearchTuner::tune<ConstantTruncatedFourierFeatureMap>(Estimator&, ConstMatrixRef, ConstMatrixRef, int,
                                                                        const RectSet&) const;
template void GridSearchTuner::tune<LinearTruncatedFourierFeatureMap>(Estimator&, ConstMatrixRef, ConstMatrixRef, int,
                                                                      const RectSet&) const;
template void GridSearchTuner::tune<LogTruncatedFourierFeatureMap>(Estimator&, ConstMatrixRef, ConstMatrixRef, int,
                                                                   const RectSet&) const;
template void GridSearchTuner::tune_online<ConstantTruncatedFourierFeatureMap>(Estimator&, ConstMatrixRef,
                                                                               const OutputComputer&, int,
                                                                               const RectSet&) const;
template void GridSearchTuner::tune_online<LinearTruncatedFourierFeatureMap>(Estimator&, ConstMatrixRef,
                                                                             const OutputComputer&, int,
                                                                             const RectSet&) const;
template void GridSearchTuner::tune_online<LogTruncatedFourierFeatureMap>(Estimator&, ConstMatrixRef,
                                                                          const OutputComputer&, int,
                                                                          const RectSet&) const;

}  // namespace lucid
