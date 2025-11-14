/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ModelEstimator class.
 */
#pragma once

#include <iosfwd>

#include "lucid/model/Estimator.h"

namespace lucid {

/**
 * Dummy estimator that uses a user-defined function to make predictions.
 * Useful for testing, in a context that assumes perfect knowledge of the system dynamics.
 */
class ModelEstimator final : public Estimator {
 public:
  /**
   * Construct a new Model Estimator object.
   * @param model_function Function that defines the model.
   */
  explicit ModelEstimator(std::function<Matrix(ConstMatrixRef)> model_function);

  [[nodiscard]] std::unique_ptr<Estimator> clone() const override;

  [[nodiscard]] Matrix predict(ConstMatrixRef x) const override;

  [[nodiscard]] double score(ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) const override;

 private:
  Estimator& consolidate_impl(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                              Requests requests) override;

  std::function<Matrix(ConstMatrixRef)> model_function_;  ///< Function that defines the model
};

std::ostream& operator<<(std::ostream& out, const ModelEstimator& model);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ModelEstimator)

#endif
