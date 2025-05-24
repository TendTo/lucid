/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ParameterValue class.
 */
#pragma once

#include <iosfwd>
#include <variant>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"

namespace lucid {

/**
 * Parameter value used in a configurable system or application.
 * It associates to the same parameter a single value.
 * The value can be of different types, i.e., int, double, and eigen Vectors.
 */
class ParameterValue {
 public:
  using ParameterValueType = std::variant<double, int, Vector>;

  /**
   * Construct a new ParameterValue object for the given `parameter` associated with the given `value`.
   * @param parameter parameter to set
   * @param value value assigned to the parameter
   */
  ParameterValue(const Parameter parameter, double value) : parameter_{parameter}, value_{value} {}
  /**
   * Construct a new ParameterValue object for the given `parameter` associated with the given `value`.
   * @param parameter parameter to set
   * @param value value assigned to the parameter
   */
  ParameterValue(const Parameter parameter, int value) : parameter_{parameter}, value_{value} {}
  /**
   * Construct a new ParameterValue object for the given `parameter` associated with the given `value`.
   * @param parameter parameter to set
   * @param value value assigned to the parameter
   */
  ParameterValue(const Parameter parameter, Vector value) : parameter_{parameter}, value_{value} {}

  /** @getter{parameter, parameter value} */
  [[nodiscard]] Parameter parameter() const { return parameter_; }
  /** @getter{value, parameter value} */
  [[nodiscard]] const ParameterValueType &value() const { return value_; }
  /**
   * Get the value of this parameter.
   * @tparam T type of the value to retrieve
   * @return value of the parameter
   */
  template <class T>
  [[nodiscard]] const T &get() const {
    return std::get<T>(value_);
  }

 private:
  Parameter parameter_;       ///< Parameter associated with the value
  ParameterValueType value_;  ///< Value assigned to the parameter
};

std::ostream &operator<<(std::ostream &os, const ParameterValue &parameter_value);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ParameterValue)

#endif
