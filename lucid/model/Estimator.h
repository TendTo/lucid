/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Estimator class.
 */
#pragma once

#include <iosfwd>
#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parametrizable.h"
#include "lucid/model/Request.h"

namespace lucid {

// Forward declarations
class Estimator;
class Tuner;
/**
 * Instead of providing the outputs directly to the estimator, this function can be used to compute them on the fly.
 * Useful when the outputs are not available at the time of fitting/scoring, or depend on changing parameters.
 * @param estimator Estimator object involved in the computation
 * @param inputs @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X} @f$
 * @return @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$
 */
using OutputComputer = std::function<ConstMatrixRefCopy(const Estimator&, ConstMatrixRef)>;
namespace scorer {
/**
 * Function type used to score the estimator.
 * @param estimator Estimator object to score
 * @param evaluation_inputs @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X} @f$
 * @param evaluation_outputs @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$
 */
using Scorer = std::function<double(const Estimator&, ConstMatrixRef, ConstMatrixRef)>;
}  // namespace scorer

/**
 * Given two vector spaces, @XsubRdx and @YsubRdy, and a map @f$ f: \mathcal{X} \to \mathcal{Y} @f$,
 * the Estimator's goal is to produce a model @f$ f^*:\mathcal{X} \to \mathcal{Y} @f$ that best approximates @f$ f @f$.
 */
class Estimator : public Parametrizable {
 public:
  /**
   * Construct a new Estimator object.
   * If no tuner is provided, the estimator will not use any during the fitting process,
   * keeping the hyperparameters it was initialised with.
   * @param parameters parameters of the estimator, default to no specific parameter
   * @param tuner Tuner object used to find the best hyperparameters for the model.
   * If none has been provided, the estimator will not use any tuner during the fitting process,
   * keeping the hyperparameters it was initialised with.
   */
  explicit Estimator(Parameters parameters = NoParameters, const std::shared_ptr<const Tuner>& tuner = nullptr);
  /**
   * A model is a function that takes a @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X}
   * @f$ and returns a @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$.
   * @pre The estimator should be fitted before calling this method.
   * @param x @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
   * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
   */
  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const;
  /**
   * A model is a function that takes a @f$ n \times d_x @f$ matrix of row vectors in the input space @f$ \mathcal{X}
   * @f$ and returns a @f$ n \times d_y @f$ matrix of row vectors in the output space @f$ \mathcal{Y} @f$.
   * @pre The estimator should be fitted before calling this method.
   * @param x @f$ n \times d_x @f$ matrix of row vectors in @f$ \mathcal{X} @f$
   * @return @f$ n \times d_y @f$ matrix of row vectors in @f$ \mathcal{Y} @f$
   */
  [[nodiscard]] virtual Matrix predict(ConstMatrixRef x) const = 0;
  /**
   * Fit the model to the given data.
   * This method will use the provided `tuner` to find the best hyperparameters for the model.
   * After the process is completed, the estimator can be used to make predictions on new data.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param training_inputs training input data
   * @param training_outputs training output data
   * @param tuner Tuner object used to find the best hyperparameters for the model
   * @return reference to this estimator after it has been fitted
   */
  Estimator& fit(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs, const Tuner& tuner);
  /**
   * Fit the model to the given data.
   * This method will use the object's tuner to find the best hyperparameters for the model.
   * After the process is completed, the estimator can be used to make predictions on new data.
   * If no tuner has been provided during construction,
   * the method is equivalent to @ref consolidate with no specific request.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param training_inputs training input data
   * @param training_outputs training output data
   * @return reference to this estimator after it has been fitted
   */
  Estimator& fit(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs);
  /**
   * Fit the model to the data produced by the `training_outputs` OutputComputer.
   * This method will use the provided `tuner` to find the best hyperparameters for the model.
   * After the process is completed, the estimator can be used to make predictions on new data.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param training_inputs training input data
   * @param training_outputs training output data. It uses an OutputComputer to compute the outputs when needed
   * @param tuner Tuner object used to find the best hyperparameters for the model
   * @return reference to this estimator after it has been fitted
   */
  Estimator& fit_online(ConstMatrixRef training_inputs, const OutputComputer& training_outputs, const Tuner& tuner);
  /**
   * Fit the model to the data produced by the `training_outputs` OutputComputer.
   * This method will use the object's tuner to find the best hyperparameters for the model.
   * After the process is completed, the estimator can be used to make predictions on new data.
   * If no tuner has been provided during construction,
   * the method is equivalent to @ref consolidate with no specific request.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param training_inputs training input data
   * @param training_outputs training output data. It uses an OutputComputer to compute the outputs when needed
   * @return reference to this estimator after it has been fitted
   */
  Estimator& fit_online(ConstMatrixRef training_inputs, const OutputComputer& training_outputs);
  /**
   * Consolidate the model, making sure it is ready for use.
   * No fitting process is performed, and the hyperparameters are not updated,
   * but the estimator may change its internal state so it can be used for predictions.
   * After the process is completed, the estimator can be used to make predictions on new data.
   * The `request` parameter is used to specify an additional option for the consolidation process,
   * like whether to compute some additional values used by some tuners.
   * @note This is a low-level method that allows the user great control over the behaviour of the estimator.
   * It is usually recommended to use @ref fit with the desired tuner instead.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param training_inputs training input data. The number of rows should be equal to the number of training outputs
   * @param training_outputs training output data. The number of rows should be equal to the number of training inputs
   * @param request request for the consolidation process. Defaults to no specific request being made
   * @return reference to the estimator
   */
  Estimator& consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                         const Request request = Request::_) {
    return consolidate(training_inputs, training_outputs, request | Request::_);
  }
  /**
   * Consolidate the model, making sure it is ready for use.
   * No fitting process is performed, and the hyperparameters are not updated,
   * but the estimator may change its internal state so it can be used for predictions.
   * After the process is completed, the estimator can be used to make predictions on new data.
   * The `requests` parameter can be used to specify additional options for the consolidation process,
   * like whether to compute some additional values used by some tuners.
   * @note This is a low-level method that allows the user great control over the behaviour of the estimator.
   * It is usually recommended to use @ref fit with the desired tuner instead.
   * @pre The number of rows in the training inputs should be equal to the number of rows in the training outputs.
   * @param training_inputs training input data
   * @param training_outputs training output data
   * @param requests requests for the consolidation process
   * @return reference to the estimator
   */
  virtual Estimator& consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                                 Requests requests) = 0;
  /**
   * Score the estimator assigning a numerical value to its accuracy in predicting the `evaluation_outputs`
   * given the `evaluation_inputs`.
   * Given the evaluation inputs @f$ x = \{ x_1, \dots, x_n \} @f$,
   * where @f$ x_i \in \mathcal{X} \subseteq \mathbb{R}^{d_x}, 0 \le i \le n @f$,
   * we want to give a numerical score to the model's predictions @f$ \hat{y} = \{ \hat{y}_1, \dots, \hat{y}_n \} @f$
   * with respect to the true outputs @f$ y = \{ y_1, \dots, y_n \} @f$,
   * where @f$ y_i \in \mathcal{Y} \subseteq \mathbb{R}^{d_y}, 0 \le i \le n @f$.
   * @note By default, the score is computed using the @ref scorer::r2_score.
   * @pre The methods @ref fit or @ref consolidate should have been called at least once before calling this method.
   * @param evaluation_inputs @nxdx evaluation input data
   * @param evaluation_outputs @nxdy evaluation output data
   * @return score of the model
   */
  [[nodiscard]] virtual double score(ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) const = 0;

  /** @getter{tuner, estimator, Can be null} */
  [[nodiscard]] const std::shared_ptr<const Tuner>& tuner() const { return tuner_; }
  /** @getsetter{tuner, estimator, Can be null} */
  std::shared_ptr<const Tuner>& m_tuner() { return tuner_; }

  /**
   * Clone the estimator by creating a new instance with the same parameters.
   * @return new instance of the estimator
   */
  [[nodiscard]] virtual std::unique_ptr<Estimator> clone() const = 0;

 private:
  std::shared_ptr<const Tuner>
      tuner_;  ///< Tuner used during fitting if no other turner is provided. If null, no tuning is performed
};

std::ostream& operator<<(std::ostream& os, const Estimator&);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Estimator)

#endif
