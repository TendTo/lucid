/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * KFold class.
 */
#pragma once

#include <iosfwd>
#include <string>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/model/CrossValidator.h"

namespace lucid {

/**
 * Generic K-Fold cross-validation, where the dataset is divided into `k` folds.
 * The model is trained `k` times, each time using `k-1` folds for training and the remaining fold for validation.
 * The final model is the one that achieved the best validation score across all folds.
 * This technique helps to mitigate overfitting and provides a more robust estimate of the model's performance.
 * @mermaid
 * ---
 * title: "K fold [k = 4]"
 * config:
 *     packet:
 *         bitWidth: 64
 *         bitsPerRow: 12
 *         showBits: false
 * ---
 * packet
 * +9: "Training"
 * +3: "Validation"
 * +6: "Training"
 * +3: "Validation"
 * +3: "Training"
 * +3: "Training"
 * +3: "Validation"
 * +6: "Training"
 * +3: "Validation"
 * +9: "Training"
 * @endmermaid
 */
class KFold final : public CrossValidator {
 public:
  /**
   * Construct a new KFold cross-validator with the specified number of folds.
   * @pre `num_folds` must be at least 2
   * @param num_folds number of folds to use in cross-validation
   * @param shuffle whether to shuffle the data before splitting into folds
   */
  explicit KFold(Dimension num_folds = 5, bool shuffle = true);

  /**
   * Get the number of folds used in cross-validation.
   * In KFold cross-validation, the dataset is divided into the number of folds specified during construction.
   * @pre The number of samples in `training_inputs` must be at least equal to the number of folds
   * @param training_inputs @nxdx training input data
   * @return number of folds
   */
  [[nodiscard]] Dimension num_folds(ConstMatrixRef training_inputs) const override;

  /** @getter{num_folds, the cross-validation} */
  [[nodiscard]] Dimension num_folds() const { return num_folds_; }
  /** @checker{data, to be shuffled before being split into folds} */
  [[nodiscard]] bool shuffle() const { return shuffle_; }

  [[nodiscard]] std::string to_string() const override;

 private:
  [[nodiscard]] std::pair<SliceSelector, SliceSelector> compute_folds(ConstMatrixRef training_inputs) const override;

  Dimension num_folds_;  ///< Number of folds to use in cross-validation
  bool shuffle_;         ///< Whether to shuffle the data before splitting into folds
};

std::ostream& operator<<(std::ostream& os, const KFold& kf);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::KFold)

#endif  // LUCID_INCLUDE_FMT
