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
#include "lucid/util/exception.h"

namespace lucid {

enum class Parameter {
  // Mean parameter
  // MEAN = 0,     ///< Mean parameter (@see HyperParameter::SIGMA_F)
  SIGMA_F = 0,  ///< Sigma_f parameter
  // Length scale parameter
  // LENGTH_SCALE = 1,  ///< Length scale parameter (@see HyperParameter::SIGMA_L, @see HyperParameter::COVARIANCE)
  // COVARIANCE = 1,    ///< Covariance parameter (@see HyperParameter::SIGMA_L, @see HyperParameter::LENGTH_SCALE)
  SIGMA_L = 1,  ///< Sigma_l parameter
  // Regularization parameter
  REGULARIZATION_CONSTANT = 2,
  // Polynomial degree parameter
  DEGREE = 3,  ///< Degree of the polynomial
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

template <class T, Parameter P>
T dispatch(const std::function<T()>& fun_int, const std::function<T()>& fun_double,
           const std::function<T()>& fun_vector) {
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

template <class T>
T dispatch(const Parameter parameter, const std::function<T()>& fun_int, const std::function<T()>& fun_double,
           const std::function<T()>& fun_vector) {
  switch (parameter) {
    case Parameter::DEGREE:
      return dispatch<T, Parameter::DEGREE>(fun_int, fun_double, fun_vector);
    case Parameter::SIGMA_F:
    case Parameter::REGULARIZATION_CONSTANT:
      return dispatch<T, Parameter::SIGMA_F>(fun_int, fun_double, fun_vector);
    case Parameter::SIGMA_L:
      return dispatch<T, Parameter::SIGMA_L>(fun_int, fun_double, fun_vector);
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
