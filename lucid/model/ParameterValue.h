/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HyperParameterValue class.
 */
#pragma once

#include <iosfwd>
#include <variant>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"

namespace lucid {

class ParameterValue {
 public:
  using ParameterValueType = std::variant<double, int, Vector>;

  ParameterValue(const Parameter parameter, double value) : parameter_{parameter}, value_{value} {}
  ParameterValue(const Parameter parameter, int value) : parameter_{parameter}, value_{value} {}
  ParameterValue(const Parameter parameter, Vector value) : parameter_{parameter}, value_{value} {}

  [[nodiscard]] Parameter parameter() const { return parameter_; }
  [[nodiscard]] const ParameterValueType &value() const { return value_; }
  template <class T>
  [[nodiscard]] const T &get() const {
    return std::get<T>(value_);
  }

 private:
  Parameter parameter_;
  ParameterValueType value_;
};

std::ostream &operator<<(std::ostream &os, const ParameterValue &parameter_value);

}  // namespace lucid
