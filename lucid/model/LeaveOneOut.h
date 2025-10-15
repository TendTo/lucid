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

/**
 * Leave-One-Out cross-validation (LOOCV) is a specific type of cross-validation where the number of folds equals the
 * number of data points in the dataset.
 * In LOOCV, each data point is used once as a validation set while the remaining data points form the training set.
 * This means that for a dataset with `n` samples, the model is trained `n` times,
 * each time leaving out one sample for validation.
 * LOOCV is particularly useful for small datasets, as it maximizes the amount of training data used in each iteration.
 * However, it can be computationally expensive for large datasets
 * due to the high number of training iterations required.
 * Moreover, LOOCV can lead to high variance in the model's performance estimate,
 * as each training set is very similar to the others, differing by only one sample.
 * @mermaid
 * ---
 * title: "Leave one out"
 * config:
 *     packet:
 *         bitWidth: 64
 *         bitsPerRow: 10
 *         showBits: false
 * ---
 * packet
 * +9: "Training"
 * +1: "Validation"
 * +8: "Training"
 * +1: "Validation"
 * +1: "Training"
 * +10: "..."
 * +1: "Training"
 * +1: "Validation"
 * +8: "Training"
 * +1: "Validation"
 * +9: "Training"
 * @endmermaid
 */
class LeaveOneOut final : public CrossValidator {
 public:
  /**
   * In the Leave-One-Out cross-validation (LOOCV) strategy,
   * the number of folds is equal to the number of samples in the dataset, i.e., @n.
   * @param training_inputs @nxdx training input data
   * @return number of folds (equal to the number of samples in the dataset)
   */
  [[nodiscard]] Dimension num_folds(ConstMatrixRef training_inputs) const override { return training_inputs.rows(); }

 private:
  [[nodiscard]] std::pair<SliceSelector, SliceSelector> compute_folds(ConstMatrixRef training_inputs) const override;
};

}  // namespace lucid
