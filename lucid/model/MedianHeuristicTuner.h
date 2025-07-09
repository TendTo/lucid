/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * MedianHeuristicTuner class.
 */
#pragma once

#include "lucid/model/Tuner.h"

namespace lucid {

/**
 * Tuner that adjusts the kernel parameters using the median heuristic method.
 * It computes the median of pairwise distances between input data points for each dimension.
 * These computed medians are used to update the @sigmal parameters of the kernel in the given estimator.
 * More specifically, given a set of training inputs @f$ \{x_1, x_2, \ldots, x_n\} @f$ from @XsubRd,
 * @f[
 * \sigma_l = \sqrt{\text{median}\{\|x_i - x_j\|^2 : 1 \le i < j \le n\}}
 * @f]
 * Once the kernel parameters are updated, the estimator is consolidated using the training data.
 * @note Based on the paper [Large sample analysis of the median heuristic](https://arxiv.org/abs/1707.07269).
 */
class MedianHeuristicTuner final : public Tuner {
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                 const OutputComputer& training_outputs) const override;
};

std::ostream& operator<<(std::ostream& os, const MedianHeuristicTuner& tuner);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::MedianHeuristicTuner)

#endif
