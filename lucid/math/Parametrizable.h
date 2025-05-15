/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Parametrizable class.
 */
#pragma once

#include <variant>

#include "lucid/lib/eigen.h"
#include "lucid/math/Parameter.h"
#include "lucid/util/concept.h"

namespace lucid {

class Parametrizable {
 public:
  virtual ~Parametrizable() = default;

  /**
   * Get the value of the specified `parameter`.
   * @tparam T type of the value to retrieve
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  template <IsAnyOf<int, double, const Vector&> T>
  [[nodiscard]] T get(Parameter parameter) const;

  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to
   * @param value value to assign to the specified parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  void set(Parameter parameter, const std::variant<int, double, Vector>& value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to
   * @param value value to assign to the specified parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  virtual void set(Parameter parameter, int value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to
   * @param value value to assign to the specified parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  virtual void set(Parameter parameter, double value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to set
   * @param value value to assign to the specified parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  virtual void set(Parameter parameter, const Vector& value);
  /**
   * Check wether the `parameter` is present in this object.
   * @param parameter parameter to check
   * @return true if the parameter is present
   * @return false if the parameter is not present
   */
  virtual bool has(Parameter parameter);

 protected:
  /**
   * Get the value of the specified `parameter`.
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  [[nodiscard]] virtual int get_i(Parameter parameter) const;
  /**
   * Get the value of the specified `parameter`.
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  [[nodiscard]] virtual double get_d(Parameter parameter) const;
  /**
   * Get the value of the specified `parameter`.
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @throw LucidInvalidArgument if the parameter is not valid for this object
   */
  [[nodiscard]] virtual const Vector& get_v(Parameter parameter) const;
};

}  // namespace lucid
