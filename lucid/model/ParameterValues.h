/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ParameterValues class.
 */
#pragma once

#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"
#include "lucid/util/concept.h"

namespace lucid {

/**
 * Collection of parameter values used in a configurable system or application.
 * It associates with the same parameter an arbitrarily large number of values.
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
  ParameterValues(const Parameter parameter, std::vector<double> values)
      : parameter_(parameter), size_{values.size()}, values_{std::move(values)} {}
  /**
   * Construct a new ParameterValues object for the given `parameter` associated with the given `values`.
   * @param parameter parameter to set
   * @param values all values assigned to the parameter
   */
  ParameterValues(const Parameter parameter, std::vector<int> values)
      : parameter_(parameter), size_{values.size()}, values_{std::move(values)} {}
  /**
   * Construct a new ParameterValues object for the given `parameter` associated with the given `values`.
   * @param parameter parameter to set
   * @param values all values assigned to the parameter
   */
  ParameterValues(const Parameter parameter, std::vector<Vector> values)
      : parameter_(parameter), size_{values.size()}, values_{std::move(values)} {}

  template <IsAnyOf<int, double, Vector>... T>
    requires(sizeof...(T) > 0)
  explicit ParameterValues(const Parameter parameter, T... values)
      : parameter_(parameter),
        size_{sizeof...(T)},
        values_{std::vector<std::tuple_element_t<0, std::tuple<T...>>>{
            std::forward<std::tuple_element_t<0, std::tuple<T...>>>(values)...}} {}

  /** @getter{parameter, parameter values} */
  [[nodiscard]] Parameter parameter() const { return parameter_; }
  /** @getter{size, parameter values} */
  [[nodiscard]] std::size_t size() const { return size_; }
  /** @getter{values, parameter values} */
  [[nodiscard]] const ParameterValuesType &values() const { return values_; }
  /**
   * Get the values of this parameter.
   * @tparam T type of the value to retrieve
   * @return value of the parameter
   */
  template <IsAnyOf<int, double, Vector> T>
  [[nodiscard]] const std::vector<T> &get() const {
    return std::get<std::vector<T>>(values_);
  }

  /** @equal_to{parameter values objects} */
  [[nodiscard]] bool operator==(const ParameterValues &o) const;

 private:
  Parameter parameter_;         ///< Parameter the values are assigned to
  std::size_t size_;            ///< Number of values assigned to the parameter
  ParameterValuesType values_;  ///< All values assigned to the parameter
};

std::ostream &operator<<(std::ostream &os, const ParameterValues &parameter_values);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ParameterValues)

#endif
