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
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/util/concept.h"
#include "lucid/util/definitions.h"
#include "lucid/util/exception.h"

namespace lucid {

/**
 * List of available parameters used parametrizable objects (e.g., Estimator and Kernel).
 * To check whether an object supports a specific parameter, use the `has(Parameter)` method.
 * This enum behaves as a bitset.
 * It is possible to combine multiple parameters using bitwise OR operations
 * or to check if a parameter is set using the AND operations.
 * @code
 * Parameter::_ // Empty set {}
 * Parameters u = Parameter::SIGMA_F | Parameter::SIGMA_L  // {SIGMA_F} U {SIGMA_L} = {SIGMA_F, SIGMA_L}
 * u | Parameter::DEGREE  // {SIGMA_F, SIGMA_L} U {DEGREE} = {SIGMA_F, SIGMA_L, DEGREE}
 * u & Parameter::SIGMA_F  // Set intersection {SIGMA_F, SIGMA_L} ∩ {SIGMA_F} = {SIGMA_F}
 * u && Parameter::SIGMA_F  // Check if {SIGMA_F, SIGMA_L} ∩ {SIGMA_F} = {SIGMA_F} is non-empty
 * u || Parameter::SIGMA_F  // Check if {SIGMA_F, SIGMA_L} ∪ {SIGMA_F} = {SIGMA_F, SIGMA_L} is non-empty
 * @endcode
 * @note The numerical values are offset in such a way that operating over them is very efficient.
 */
enum class Parameter : std::uint16_t {
  _ = 0,                             ///< No parameters. Used as the empty set placeholder.
  SIGMA_F = 1 << 0,                  ///< Sigma_f parameter
  SIGMA_L = 1 << 1,                  ///< Sigma_l parameter
  REGULARIZATION_CONSTANT = 1 << 2,  ///< Regularization constant parameter
  DEGREE = 1 << 3,                   ///< Degree of the polynomial
  GRADIENT_OPTIMIZABLE = 1 << 4,     ///< Gradient optimizable parameter
};

using HP = Parameter;                                                       ///< Alias for HyperParameter
using Parameters = std::underlying_type_t<Parameter>;                       ///< Efficient set of parameters as bitset
constexpr Parameters NoParameters = static_cast<Parameters>(Parameter::_);  ///< No parameter value placeholder

LUCID_FLAG_ENUMS(Parameter, Parameters, GRADIENT_OPTIMIZABLE)

namespace internal {

struct ParameterTypeInt {
  using type = int;
  using ref_type = int;
};
struct ParameterTypeDouble {
  using type = double;
  using ref_type = double;
};
struct ParameterTypeVector {
  using type = Vector;
  using ref_type = const Vector&;
};

template <Parameter>
struct ParameterType {};
template <>
struct ParameterType<Parameter::SIGMA_F> : ParameterTypeDouble {};
template <>
struct ParameterType<Parameter::SIGMA_L> : ParameterTypeVector {};
template <>
struct ParameterType<Parameter::REGULARIZATION_CONSTANT> : ParameterTypeDouble {};
template <>
struct ParameterType<Parameter::DEGREE> : ParameterTypeInt {};
template <>
struct ParameterType<Parameter::GRADIENT_OPTIMIZABLE> : ParameterTypeVector {};

}  // namespace internal

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
    case Parameter::GRADIENT_OPTIMIZABLE:
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
