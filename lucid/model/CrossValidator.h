/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * CrossValidator class.
 */
#pragma once

#include <iosfwd>
#include <utility>
#include <vector>
#include <string>

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"

namespace lucid {

/**
 * During cross-validation, the available data is split into `k` folds.
 * The model is trained `k` times, each time using `k-1` folds for training and the remaining fold for validation.
 * The final model is the one that achieved the best validation score across all folds.
 * This technique helps to mitigate overfitting and provides a more robust estimate of the model's performance.
 */
class CrossValidator {
 public:
  using SliceSelector = std::vector<std::vector<Index>>;  ///< Type alias for slice selector

  virtual ~CrossValidator() = default;

  /**
   * Fit the `estimator` using cross-validation on the provided `training_inputs` and `training_outputs`.
   * The `scorer` is used to evaluate the performance of the model on the validation folds.
   * If no `scorer` is provided, the estimator's default scoring method is used.
   * At the end, the `estimator` is updated to the best model found during cross-validation.
   * @pre The number of samples @n in `training_inputs` must be at least equal to the number of folds @ref num_folds.
   * @param[in,out] estimator estimator to fit. It will be updated to the best model found
   * @param training_inputs @nxdx training input data
   * @param training_outputs @nxdy training output data
   * @param scorer scoring function to evaluate the model's performance
   * @return best score achieved during cross-validation
   */
  double fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
             const scorer::Scorer& scorer = nullptr) const;
  /**
   * Fit the `estimator` using cross-validation on the provided `training_inputs` and `training_outputs` using the
   * tuner `tuner` to optimize @hp.
   * The `scorer` is used to evaluate the performance of the model on the validation folds.
   * If no `scorer` is provided, the estimator's default scoring method is used.
   * At the end, the `estimator` is updated to the best model found during cross-validation.
   * @pre The number of samples @n in `training_inputs` must be at least equal to the number of folds @ref num_folds.
   * @param[in,out] estimator estimator to fit. It will be updated to the best model found
   * @param training_inputs @nxdx training input data
   * @param training_outputs @nxdy training output data
   * @param tuner tuner to optimize @hp
   * @param scorer scoring function to evaluate the model's performance
   * @return best score achieved during cross-validation
   */
  double fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs, const Tuner& tuner,
             const scorer::Scorer& scorer = nullptr) const;

  /**
   * Evaluate the `estimator` using cross-validation on the provided `inputs` and `outputs`.
   * No fitting or tuning is performed, hence the `estimator` will only be consolidated on each fold,
   * but no @hp will be optimized.
   * The `scorer` is used to evaluate the performance of the model on each fold.
   * If no `scorer` is provided, the estimator's default scoring method is used.
   * The result is a vector of scores, one for each fold.
   * @pre The number of samples @n in `training_inputs` must be at least equal to the number of folds @ref num_folds.
   * @param estimator estimator to evaluate
   * @param inputs @nxdx input data
   * @param outputs @nxdy output data
   * @param scorer scoring function to evaluate the model's performance
   * @return vector of scores, one for each fold
   */
  [[nodiscard]] std::vector<double> score(const Estimator& estimator, ConstMatrixRef inputs, ConstMatrixRef outputs,
                                          const scorer::Scorer& scorer = nullptr) const;

  /**
   * Get the number of folds used in the cross-validation on the provided `training_inputs`.
   * This is determined by the specific cross-validation strategy implemented in the derived class.
   * @param training_inputs @nxdx training input data
   * @return number of folds
   */
  [[nodiscard]] virtual Dimension num_folds(ConstMatrixRef training_inputs) const = 0;

  /** @to_string */
  [[nodiscard]] virtual std::string to_string() const;

 protected:
  /**
   * Fit the `estimator` using cross-validation on the provided `training_inputs` and `training_outputs` using the
   * tuner `tuner` to optimize @hp, if provided.
   * The `scorer` is used to evaluate the performance of the model on the validation folds.
   * If no `scorer` is provided, the estimator's default scoring method is used.
   * At the end, the `estimator` is updated to the best model found during cross-validation.
   * @pre The number of samples @n in `training_inputs` must be at least equal to the number of folds @ref num_folds.
   * @param[in,out] estimator estimator to fit. It will be updated to the best model found
   * @param training_inputs @nxdx training input data
   * @param training_outputs @nxdy training output data
   * @param scorer scoring function to evaluate the model's performance
   * @return best score achieved during cross-validation
   */
  double fit(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs, const Tuner* tuner,
             const scorer::Scorer& scorer) const;

  /**
   * Compute the training and validation folds for cross-validation.
   * Each fold is represented as a pair of index vectors,
   * where the first vector contains the indices for the training set
   * and the second vector contains the indices for the validation set.
   * @param training_inputs @nxdx training input data
   * @return pair of slice selectors for training and validation folds
   */
  [[nodiscard]] virtual std::pair<SliceSelector, SliceSelector> compute_folds(ConstMatrixRef training_inputs) const = 0;
};

std::ostream& operator<<(std::ostream& os, const CrossValidator& cv);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::CrossValidator)

#endif  // LUCID_INCLUDE_FMT
