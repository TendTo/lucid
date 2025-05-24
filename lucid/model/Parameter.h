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

template <class T>
T dispatch(const Parameter parameter, [[maybe_unused]] const std::function<T()>& fun_int,
           const std::function<T()>& fun_double, const std::function<T()>& fun_vector) {
  switch (parameter) {
    case Parameter::DEGREE:
      return fun_int();
    case Parameter::SIGMA_F:
    case Parameter::REGULARIZATION_CONSTANT:
      return fun_double();
    case Parameter::SIGMA_L:
      return fun_vector();
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
