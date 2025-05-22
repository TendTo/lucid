/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ParameterValues class.
 */
#pragma once

#include <variant>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"

namespace lucid {

/**
 * Collection of parameter values used in a configurable system or application.
 * It associates to the same parameter an arbitrarily large number of values.
 * All values must have the same type, but the type can be different for each parameter.
 */
class ParameterValues {
 public:
  using ParameterValuesType = std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>>;

  /**
   * Construct a new ParameterValues object for the given `parameter` associated with the given `values`.
   * @param parameter parameter to set
   * @param values all values assigned to the parameter
   */
  ParameterValues(const Parameter parameter, std::vector<double> values) : parameter_(parameter), values_{values} {}
  /**
   * Construct a new ParameterValues object for the given `parameter` associated with the given `values`.
   * @param parameter parameter to set
   * @param values all values assigned to the parameter
   */
  ParameterValues(const Parameter parameter, std::vector<int> values) : parameter_(parameter), values_{values} {}
  /**
   * Construct a new ParameterValues object for the given `parameter` associated with the given `values`.
   * @param parameter parameter to set
   * @param values all values assigned to the parameter
   */
  ParameterValues(const Parameter parameter, std::vector<Vector> values) : parameter_(parameter), values_{values} {}

  /** @getter{parameter, parameter values} */
  [[nodiscard]] Parameter parameter() const { return parameter_; }
  /** @getter{values, parameter values} */
  [[nodiscard]] ParameterValuesType values() const { return values_; }
  /**
   * Get the values of this parameter.
   * @tparam T type of the value to retrieve
   * @return value of the parameter
   */
  template <class T>
  [[nodiscard]] const std::vector<T> &get() const {
    return std::get<std::vector<T>>(values_);
  }

 private:
  Parameter parameter_;         ///< Parameter the values are assigned to
  ParameterValuesType values_;  ///< All values assigned to the parameter
};

std::ostream &operator<<(std::ostream &os, const ParameterValues &parameter_value);

}  // namespace lucid
