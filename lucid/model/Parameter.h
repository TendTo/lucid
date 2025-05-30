/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Parameter enum.
 */
#pragma once

#include <functional>
#include <iosfwd>

#include "lucid/lib/eigen.h"
#include "lucid/util/concept.h"
#include "lucid/util/exception.h"

namespace lucid {

/**
 * List of available parameters used parametrizable objects (e.g., Estimator and Kernel).
 * To check whether an object supports a specific parameter, use the `has(Parameter)` method.
 * @note The parameter numerical values are offset in such a way that operating over them is very efficient.
 */
enum class Parameter : std::uint16_t {
  _ = 0,                             ///< No parameters
  SIGMA_F = 1 << 0,                  ///< Sigma_f parameter
  SIGMA_L = 1 << 1,                  ///< Sigma_l parameter
  REGULARIZATION_CONSTANT = 1 << 2,  ///< Regularization constant parameter
  DEGREE = 1 << 3,                   ///< Degree of the polynomial
};

using HP = Parameter;  ///< Alias for HyperParameter

namespace internal {

template <Parameter>
struct ParameterType {};
template <>
struct ParameterType<Parameter::SIGMA_F> {
  using type = double;
  using ref_type = double;
};
template <>
struct ParameterType<Parameter::SIGMA_L> {
  using type = Vector;
  using ref_type = const Vector&;
};
template <>
struct ParameterType<Parameter::REGULARIZATION_CONSTANT> {
  using type = double;
  using ref_type = double;
};
template <>
struct ParameterType<Parameter::DEGREE> {
  using type = int;
  using ref_type = int;
};

}  // namespace internal

/**
 * Perform a bitwise OR operation on two parameters.
 * @tparam LP left parameter type, must be one of the `Parameter` enum values or its underlying type
 * @tparam RP right parameter type, must be one of the `Parameter` enum values or its underlying type
 * @param lhs left-hand side parameter
 * @param rhs right-hand side parameter
 * @return the result of the bitwise OR operation as a `Parameter`
 */
template <IsAnyOf<Parameter, std::underlying_type_t<Parameter>> LP,
          IsAnyOf<Parameter, std::underlying_type_t<Parameter>> RP>
constexpr Parameter operator|(LP lhs, RP rhs) {
  return static_cast<Parameter>(static_cast<std::underlying_type_t<Parameter>>(lhs) |
                                static_cast<std::underlying_type_t<Parameter>>(rhs));
}
/**
 * Perform a bitwise OR operation on two parameters and return the result as a boolean.
 * @code
 * Parameter::SIGMA_L || Parameter::SIGMA_L; // true
 * Parameter::SIGMA_F || Parameter::DEGREE; // true
 * Parameter::_ || Parameter::SIGMA_L; // true
 * Parameter::_ || Parameter::_; // false
 * @endcode
 * @tparam LP left parameter type, must be one of the `Parameter` enum values or its underlying type
 * @tparam RP right parameter type, must be one of the `Parameter` enum values or its underlying type
 * @param lhs left-hand side parameter
 * @param rhs right-hand side parameter
 * @return the result of the bitwise OR operation as a boolean
 */
template <IsAnyOf<Parameter, std::underlying_type_t<Parameter>> LP,
          IsAnyOf<Parameter, std::underlying_type_t<Parameter>> RP>
constexpr bool operator||(LP lhs, RP rhs) {
  return static_cast<std::underlying_type_t<Parameter>>(lhs) | static_cast<std::underlying_type_t<Parameter>>(rhs);
}
/**
 * Perform a bitwise AND operation on two parameters.
 * @tparam LP left parameter type, must be one of the `Parameter` enum values or its underlying type
 * @tparam RP right parameter type, must be one of the `Parameter` enum values or its underlying type
 * @param lhs left-hand side parameter
 * @param rhs right-hand side parameter
 * @return the result of the bitwise AND operation as a `Parameter`
 */
template <IsAnyOf<Parameter, std::underlying_type_t<Parameter>> LP,
          IsAnyOf<Parameter, std::underlying_type_t<Parameter>> RP>
constexpr Parameter operator&(LP lhs, RP rhs) {
  return static_cast<Parameter>(static_cast<std::underlying_type_t<Parameter>>(lhs) &
                                static_cast<std::underlying_type_t<Parameter>>(rhs));
}
/**
 * Perform a bitwise AND operation on two parameters and return the result as a boolean.
 * @code
 * Parameter::SIGMA_L && Parameter::SIGMA_L; // true
 * Parameter::SIGMA_F && Parameter::DEGREE; // false
 * Parameter::_ && Parameter::SIGMA_L; // false
 * Parameter::_ && Parameter::_; // false
 * @endcode
 * @tparam LP left parameter type, must be one of the `Parameter` enum values or its underlying type
 * @tparam RP right parameter type, must be one of the `Parameter` enum values or its underlying type
 * @param lhs left-hand side parameter
 * @param rhs right-hand side parameter
 * @return the result of the bitwise AND operation as a boolean
 */
template <IsAnyOf<Parameter, std::underlying_type_t<Parameter>> LP,
          IsAnyOf<Parameter, std::underlying_type_t<Parameter>> RP>
constexpr bool operator&&(LP lhs, RP rhs) {
  return static_cast<std::underlying_type_t<Parameter>>(lhs) & static_cast<std::underlying_type_t<Parameter>>(rhs);
}

/**
 * Dispatch the correct function call depending on type associated with the `parameter`.
 * @tparam R return type, the same for all the functions
 * @tparam P parameter determining which function to dispatch
 * @param fun_int function called if the parameter is integer valued
 * @param fun_double function called if the parameter is double valued
 * @param fun_vector function called if the parameter is vector valued
 * @return value returned by the function that ended up being called
 */
template <class R, Parameter P>
R dispatch(const std::function<R()>& fun_int, const std::function<R()>& fun_double,
           const std::function<R()>& fun_vector) {
  if constexpr (std::is_same_v<typename internal::ParameterType<P>::type, int>) {
    return fun_int();
  } else if constexpr (std::is_same_v<typename internal::ParameterType<P>::type, double>) {
    return fun_double();
  } else if constexpr (std::is_same_v<typename internal::ParameterType<P>::type, Vector>) {
    return fun_vector();
  } else {
    throw exception::LucidUnreachableException{};
  }
}
/**
 * Dispatch the correct function call depending on type associated with the `parameter`.
 * @tparam R return type, the same for all the functions
 * @param parameter the parameter determining which function to dispatch
 * @param fun_int function called if the parameter is integer valued
 * @param fun_double function called if the parameter is double valued
 * @param fun_vector function called if the parameter is vector valued
 * @return value returned by the function that ended up being called
 */
template <class R>
R dispatch(const Parameter parameter, const std::function<R()>& fun_int, const std::function<R()>& fun_double,
           const std::function<R()>& fun_vector) {
  switch (parameter) {
    case Parameter::DEGREE:
      return dispatch<R, Parameter::DEGREE>(fun_int, fun_double, fun_vector);
    case Parameter::SIGMA_F:
    case Parameter::REGULARIZATION_CONSTANT:
      return dispatch<R, Parameter::SIGMA_F>(fun_int, fun_double, fun_vector);
    case Parameter::SIGMA_L:
      return dispatch<R, Parameter::SIGMA_L>(fun_int, fun_double, fun_vector);
    default:
      throw exception::LucidUnreachableException{};
  }
}

std::ostream& operator<<(std::ostream& os, Parameter name);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Parameter)

#endif
