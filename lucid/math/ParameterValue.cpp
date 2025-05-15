/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HyperParameterValue class.
 */
#include "lucid/math/ParameterValue.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const ParameterValue& parameter_value) {
  os << " Parameter(";
  return dispatch<std::ostream&>(
      parameter_value.parameter(),
      [&os, &parameter_value]() -> std::ostream& { return os << parameter_value.get<int>(); },
      [&os, &parameter_value]() -> std::ostream& { return os << parameter_value.get<double>(); },
      [&os, &parameter_value]() -> std::ostream& { return os << parameter_value.get<Vector>(); });
}

}  // namespace lucid