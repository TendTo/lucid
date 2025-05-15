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
#include "lucid/math/Parameter.h"

namespace lucid {

class ParameterValues {
 public:
  using ParameterValuesType = std::variant<std::vector<double>, std::vector<int>, std::vector<Vector>>;

  ParameterValues(const Parameter parameter, std::vector<double> values) : parameter_(parameter), values_{values} {}
  ParameterValues(const Parameter parameter, std::vector<int> values) : parameter_(parameter), values_{values} {}
  ParameterValues(const Parameter parameter, std::vector<Vector> values) : parameter_(parameter), values_{values} {}

  [[nodiscard]] Parameter parameter() const { return parameter_; }
  [[nodiscard]] ParameterValuesType values() const { return values_; }
  template <class T>
  [[nodiscard]] const std::vector<T> &get() const {
    return std::get<std::vector<T>>(values_);
  }

 private:
  Parameter parameter_;
  ParameterValuesType values_;
};

std::ostream &operator<<(std::ostream &os, const ParameterValues &parameter_value);

}  // namespace lucid
