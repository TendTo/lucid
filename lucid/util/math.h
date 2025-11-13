/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of math functions.
 */
#pragma once

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order): sorting non-issue
#include <cmath>                                // NOLINT(build/include_order): sorting non-issue

namespace lucid {
/**
 * Peaks function defined over a pair of values.
 * Useful for demonstrating graphics functions, such as contour, mesh, pcolor, and surf.
 * It is obtained by translating and scaling Gaussian distributions and is defined as
 * @f[
 * f(x, y) = 3(1 - x)^2 \exp(-x^2 - (y + 1)^2) - 10\left(\frac{x}{5} - x^3 - y^5\right) \exp(-x^2 - y^2) - \frac{1}{3}
 * \exp(-(x + 1)^2 - y^2)
 * @f]
 * @param x x value
 * @param y y value
 * @return value of the peaks function at the given point
 */
inline double peaks(const double x, const double y) {
  return 3.0 * std::pow((1.0 - x), 2) * std::exp(-std::pow(x, 2) - std::pow(y + 1.0, 2)) -
         10.0 * (x / 5.0 - std::pow(x, 3) - std::pow(y, 5)) * std::exp(-std::pow(x, 2) - std::pow(y, 2)) -
         1.0 / 3.0 * std::exp(-std::pow(x + 1.0, 2) - std::pow(y, 2));
}
/**
 * Compute the Cumulative distribution function (CDF) of the normal distribution at the point `x`.
 * @param x point at which to compute the CDF
 * @param sigma_f @sigmaf value used in the normal distribution (mean)
 * @param sigma_l @sigmal value used in the normal distribution (standard deviation)
 * @return value of the CDF at the given point
 */
inline double normal_cdf(const double x, const double sigma_f, const double sigma_l) {
  return boost::math::cdf(boost::math::normal_distribution<double>{sigma_f, sigma_l}, x);
}

/**
 * Raise a base to the power of an exponent: @f$ b^e @f$.
 * @param base @f$ b @f$ base
 * @param exp @f$ e @f$ exponent
 * @return base raised to the power of exp
 */
inline std::size_t pow(std::size_t base, std::size_t exp) {
  std::size_t result = 1;
  while (exp) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

/**
 * Raise a base to the power of an exponent: @f$ b^e @f$.
 * @param base @f$ b @f$ base
 * @param exp @f$ e @f$ exponent
 * @return base raised to the power of exp
 */
inline int pow(int base, int exp) {
  int result = 1;
  while (exp) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}


}  // namespace lucid
