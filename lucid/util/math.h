/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of math functions.
 */
#pragma once

#include <cmath>

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
}  // namespace lucid
