/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Scorer module.
 */
#pragma once

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"

/**
 * @namespace lucid::scorer
 * Collection of utilities used to score the accuracy of estimators.
 */
namespace lucid::scorer {

/**
 * Score the `estimator` assigning a numerical value to its accuracy in predicting the `evaluation_outputs`
 * given the `evaluation_inputs`.
 * Given the evaluation inputs @f$ x = \{ x_1, \dots, x_n \} @f$,
 * where @f$ x_i \in \mathcal{X} \subseteq \mathbb{R}^{d_x}, 0 \le i \le n @f$,
 * we want to give a numerical score to the model's predictions @f$ \hat{y} = \{ \hat{y}_1, \dots, \hat{y}_n \} @f$
 * where @f$ \hat{y}_i \in \mathcal{Y} \subseteq \mathbb{R}^{d_y}, 0 \le i \le n @f$,
 * with respect to the true outputs @f$ y = \{ y_1, \dots, y_n \} @f$
 * where @f$ y_i \in \mathcal{Y} , 0 \le i \le n @f$.
 * The score is computed as the coefficient of determination, also known as @f$ R^2 @f$ score, defined as
 * @f[
 * R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
 * @f]
 * where @f$ \bar{y} @f$ is the mean of the true outputs @f$ y @f$ and the variance of the true output,
 * @f$ \sum_{i=1}^n (y_i - \bar{y})^2 @f$, is greater than 0.
 * The score belongs in the range @f$ [-\infty, 1 ] @f$, where @f$ 1 @f$ indicates a perfect fit,
 * and @f$ 0 @f$ indicates that the model is no better than simply predicting the expected value of the true outputs.
 * @pre The estimator must be able to make predictions,
 * i.e., it should have been fitted or consolidated before calling this method.
 * @pre The estimator's prediction must belong to a vector space with the same number of dimensions
 * as the one the evaluation outputs inhabit.
 * @pre The variance of the `evaluation_outputs` must be greater than 0.
 * This is trivially false if all outputs are equal or only one row is present.
 * If this precondition is not met, the result may be `NaN`.
 * @pre The number of rows in `evaluation_inputs` must be equal to the number of rows in `evaluation_outputs`.
 * @param estimator estimator to score
 * @param evaluation_inputs @nxdx evaluation input data
 * @param evaluation_outputs @nxdy evaluation output data
 * @return score of the model
 */
double r2_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs);

/**
 * Compute the mean squared error (MSE) score of the `estimator` on the given evaluation data.
 * Given the evaluation inputs @f$ x = \{ x_1, \dots, x_n \} @f$,
 * where @f$ x_i \in \mathcal{X} \subseteq \mathbb{R}^{d_x}, 0 \le i \le n @f$,
 * we want to compute the mean squared error of the model's predictions
 * @f$ \hat{y} = \{ \hat{y}_1, \dots, \hat{y}_n \} @f$
 * where @f$ \hat{y}_i \in \mathcal{Y} \subseteq \mathbb{R}^{d_y}, 0 \le i \le n @f$,
 * with respect to the true outputs
 * @f$ y = \{ y_1, \dots, y_n \} @f$
 * where @f$ y_i \in \mathcal{Y} , 0 \le i \le n @f$.
 * The score belongs in the range @f$ [-\infty, 0 ] @f$, where @f$ 0 @f$ indicates a perfect fit,
 * and more negative values indicates a worse fit.
 * @f[
 * \text{MSE} = -\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
 * @f]
 * where @f$ n @f$ is the number of rows in the evaluation data.
 * The MSE score is always non-negative, and a lower value indicates a better fit.
 * @warning The MSE score is non-positive by definition in this implementation.
 * @pre The estimator must be able to make predictions,
 * i.e., it should have been fitted or consolidated before calling this method.
 * @pre The estimator's prediction must belong to a vector space with the same number of dimensions
 * as the one the evaluation outputs inhabit.
 * @pre The number of rows in `evaluation_inputs` must be equal to the number of rows in `evaluation_outputs`.
 * @param estimator estimator to score
 * @param evaluation_inputs @nxdx evaluation input data
 * @param evaluation_outputs @nxdy evaluation output data
 * @return mean squared error score of the model
 */
double mse_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs);

/**
 * Compute the root mean squared error (RMSE) score of the `estimator` on the given evaluation data.
 * Given the evaluation inputs @f$ x = \{ x_1, \dots, x_n \} @f$,
 * where @f$ x_i \in \mathcal{X} \subseteq \mathbb{R}^{d_x}, 0 \le i \le n @f$,
 * we want to compute the root mean squared error of the model's predictions
 * @f$ \hat{y} = \{ \hat{y}_1, \dots, \hat{y}_n \} @f$
 * where @f$ \hat{y}_i \in \mathcal{Y} \subseteq \mathbb{R}^{d_y}, 0 \le i \le n @f$,
 * with respect to the true outputs
 * @f$ y = \{ y_1, \dots, y_n \} @f$
 * where @f$ y_i \in \mathcal{Y} , 0 \le i \le n @f$.
 * The score belongs in the range @f$ [-\infty, 0 ] @f$, where @f$ 0 @f$ indicates a perfect fit,
 * and more negative values indicates a worse fit.
 * @f[
 * \text{RMSE} = -\sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
 * @f]
 * where @f$ n @f$ is the number of rows in the evaluation data.
 * The RMSE score is always non-negative, and a lower value indicates a better fit.
 * @warning The RMSE score is non-positive by definition in this implementation.
 * @pre The estimator must be able to make predictions,
 * i.e., it should have been fitted or consolidated before calling this method.
 * @pre The estimator's prediction must belong to a vector space with the same number of dimensions
 * as the one the evaluation outputs inhabit.
 * @pre The number of rows in `evaluation_inputs` must be equal to the number of rows in `evaluation_outputs`.
 * @param estimator estimator to score
 * @param evaluation_inputs @nxdx evaluation input data
 * @param evaluation_outputs @nxdy evaluation output data
 * @return root mean squared error score of the model
 */
double rmse_score(const Estimator& estimator, ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs);

}  // namespace lucid::scorer
