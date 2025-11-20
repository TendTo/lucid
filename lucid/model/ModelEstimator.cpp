/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ModelEstimator class.
 */
#include "lucid/model/ModelEstimator.h"

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "Scorer.h"
#include "lucid/util/error.h"

namespace lucid {

ModelEstimator::ModelEstimator(std::function<Matrix(ConstMatrixRef)> model_function)
    : model_function_{std::move(model_function)} {
  LUCID_ASSERT(model_function_ != nullptr, "model_function must not be null");
}
std::unique_ptr<Estimator> ModelEstimator::clone() const { return std::make_unique<ModelEstimator>(*this); }
Matrix ModelEstimator::predict(ConstMatrixRef x) const { return model_function_(x); }
double ModelEstimator::score(ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) const {
  return scorer::r2_score(*this, evaluation_inputs, evaluation_outputs);
}

Estimator& ModelEstimator::consolidate_impl(ConstMatrixRef, ConstMatrixRef, Requests) { return *this; }

std::string ModelEstimator::to_string() const { return "ModelEstimator( )"; }

std::ostream& operator<<(std::ostream& out, const ModelEstimator& model) { return out << model.to_string(); }

}  // namespace lucid
