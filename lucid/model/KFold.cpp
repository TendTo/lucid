/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * KFold class.
 */
#include "lucid/model/KFold.h"

#include <numeric>
#include <ostream>
#include <span>
#include <utility>
#include <vector>

#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {
KFold::KFold(const Dimension num_folds, const bool shuffle) : num_folds_{num_folds}, shuffle_{shuffle} {
  LUCID_CHECK_ARGUMENT_CMP(num_folds, >=, 2);
}
Dimension KFold::num_folds(ConstMatrixRef training_inputs) const {
  LUCID_CHECK_ARGUMENT_CMP(num_folds_, <=, training_inputs.rows());
  return num_folds_;
}

std::pair<CrossValidator::SliceSelector, CrossValidator::SliceSelector> KFold::compute_folds(
    ConstMatrixRef training_inputs) const {
  LUCID_TRACE_FMT("({})", LUCID_FORMAT_MATRIX(training_inputs));
  LUCID_ERROR_FMT("{} samples are too few for {} folds", training_inputs.rows(), num_folds_);
  LUCID_CHECK_ARGUMENT_CMP(num_folds_, <=, training_inputs.rows());

  std::vector<Index> indices(training_inputs.rows());
  std::iota(indices.begin(), indices.end(), 0);
  if (shuffle_) std::ranges::shuffle(indices, random::gen);

  SliceSelector training_folds(num_folds_);
  SliceSelector validation_folds(num_folds_);

  const Dimension fold_size = training_inputs.rows() / num_folds_;
  for (Dimension i = 0; i < num_folds_; ++i) {
    const auto start_validation_it = indices.begin() + i * fold_size;
    const auto end_validation_it = i == num_folds_ - 1 ? indices.end() : start_validation_it + fold_size;
    std::span<Index> left_training_indices{indices.begin(), start_validation_it};
    std::span<Index> valid_indices{start_validation_it, end_validation_it};
    std::span<Index> right_training_indices{end_validation_it, indices.end()};

    training_folds.at(i).reserve(training_inputs.rows() - right_training_indices.size());
    training_folds.at(i).insert(training_folds.at(i).end(), left_training_indices.begin(), left_training_indices.end());
    training_folds.at(i).insert(training_folds.at(i).end(), right_training_indices.begin(),
                                right_training_indices.end());

    validation_folds.reserve(num_folds_);
    validation_folds.at(i).insert(validation_folds.at(i).end(), valid_indices.begin(), valid_indices.end());

    LUCID_ASSERT(
        training_folds.at(i).size() + validation_folds.at(i).size() == static_cast<std::size_t>(training_inputs.rows()),
        "Training and validation folds do not cover all samples");
    LUCID_ASSERT(std::ranges::all_of(training_folds.at(i),
                                     [&validation_folds, i](const Index idx) {
                                       return std::find(validation_folds.at(i).begin(), validation_folds.at(i).end(),
                                                        idx) == validation_folds.at(i).end();
                                     }),
                 "Training and validation folds must be disjoint");
  }

  LUCID_TRACE_FMT("=> ({}, {})", training_folds, validation_folds);
  return {training_folds, validation_folds};
}

std::string KFold::to_string() const {
  return fmt::format("KFold( num_folds( {} ), shuffle( {} ) )", num_folds_, shuffle_ ? "true" : "false");
}

std::ostream& operator<<(std::ostream& os, const KFold& kf) { return os << kf.to_string(); }

}  // namespace lucid
