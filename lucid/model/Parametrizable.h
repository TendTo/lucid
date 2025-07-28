/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Parametrizable class.
 */
#pragma once

#include <variant>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/model/Parameter.h"
#include "lucid/util/concept.h"

namespace lucid {

/**
 * Interface for objects that allow setting and getting parameters dynamically.
 * Parametrizable objects can have parameters of different types, i.e., int, double, and eigen Vectors.
 * @code
 * std::unique_ptr<Parametrizable> obj{make_unique<...>(...)};
 * obj->has(Parameter::SIGMA_L);  // true if the parameter is present, false otherwise
 * obj->set(Parameter::SIGMA_L, 1.0); // set the parameter to 1.0
 * obj->get<double>(Parameter::SIGMA_L); // get the value of the parameter
 * @endcode
 */
class Parametrizable {
 public:
  explicit Parametrizable(const Parameters parameters = NoParameters) : parameters_{parameters} {}
  Parametrizable(const Parametrizable&) = default;
  Parametrizable(Parametrizable&&) = default;
  Parametrizable& operator=(const Parametrizable&) = default;
  Parametrizable& operator=(Parametrizable&&) = default;
  virtual ~Parametrizable() = default;

  /**
   * Get the value of the specified `parameter`.
   * @tparam T type of the value to retrieve
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @pre The `parameter` must be present and be associated with a value of type `T`.
   */
  template <IsAnyOf<int, double, const Vector&> T>
  [[nodiscard]] T get(Parameter parameter) const;
  /**
   * Get the value of the specified `parameter`.
   * @tparam P parameter to retrieve
   * @return value of the parameter
   * @pre The `parameter` must be present.
   */
  template <Parameter P>
  [[nodiscard]] typename internal::ParameterType<P>::ref_type get() const {
    return get<typename internal::ParameterType<P>::ref_type>(P);
  }

  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to set
   * @param value value to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value matching the variant.
   */
  void set(Parameter parameter, const std::variant<int, double, Vector>& value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @tparam P parameter to set
   * @param value value to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value matching the variant.
   */
  template <Parameter P>
  void set(const std::variant<int, double, Vector>& value) {
    set(P, std::get<typename internal::ParameterType<P>::type>(value));
  }

  /**
   * Set the `parameter` to the `index`-th value among the indicated `values`.
   * @param parameter parameter to set
   * @param idx index of the value to assign
   * @param values values to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value matching the variant.
   */
  void set(Parameter parameter, std::size_t idx,
           const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>>& values);
  /**
   * Set the `parameter` to the `index`-th value among the indicated `values`.
   * @tparam P parameter to set
   * @param idx index of the value to assign
   * @param values values to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value matching the variant.
   */
  template <Parameter P>
  void set(std::size_t idx, const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>>& values) {
    set(P, std::get<std::vector<typename internal::ParameterType<P>::type>>(values).at(idx));
  }

  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to set
   * @param value value to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value of type `int`.
   */
  virtual void set(Parameter parameter, int value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to set
   * @param value value to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value of type `double`.
   */
  virtual void set(Parameter parameter, double value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @param parameter parameter to set
   * @param value value to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value of type `Vector`.
   */
  virtual void set(Parameter parameter, const Vector& value);
  /**
   * Set the `parameter` to the indicated `value`.
   * @tparam P parameter to set
   * @param value value to assign to the specified parameter
   * @pre The `parameter` must be present and be associated with a value of type `Vector`.
   */
  template <Parameter P>
  void set(typename internal::ParameterType<P>::ref_type value) {
    set(P, value);
  }

  /**
   * Check whether the `parameter` is present in this object.
   * @param parameter parameter to check
   * @return true if the parameter is present
   * @return false if the parameter is not present
   */
  [[nodiscard]] bool has(const Parameter parameter) const { return parameter && parameters_; }

  /** @getter{parameters, parametrizable object,
   * The parameters are stored in compressed form\, needing bitwise operation to be accessed.} */
  [[nodiscard]] Parameters parameters() const { return parameters_; }
  /** @getter{list of parameters, parametrizable object} */
  [[nodiscard]] std::vector<Parameter> parameters_list() const;

 protected:
  /**
   * Get the value of the specified `parameter`.
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @pre The `parameter` must be present and be associated with a value of type `int`.
   */
  [[nodiscard]] virtual int get_i(Parameter parameter) const;
  /**
   * Get the value of the specified `parameter`.
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @pre The `parameter` must be present and be associated with a value of type `double`.
   */
  [[nodiscard]] virtual double get_d(Parameter parameter) const;
  /**
   * Get the value of the specified `parameter`.
   * @param parameter parameter to retrieve
   * @return value of the parameter
   * @pre The `parameter` must be present and be associated with a value of type `Vector`.
   */
  [[nodiscard]] virtual const Vector& get_v(Parameter parameter) const;

  Parameters parameters_;  ///< Parameters supported by this object
};

}  // namespace lucid
