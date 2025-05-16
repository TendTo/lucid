/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ParameterValues class.
 */
#include "lucid/model/ParameterValues.h"

#include "lucid/util/logging.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const ParameterValues& parameter_values) {
  os << " Parameter(";
  return dispatch<std::ostream&>(
      parameter_values.parameter(),
      [&os, &parameter_values]() -> std::ostream& { return os << fmt::format("{}", parameter_values.get<int>()); },
      [&os, &parameter_values]() -> std::ostream& { return os << fmt::format("{}", parameter_values.get<double>()); },
      [&os, &parameter_values]() -> std::ostream& { return os << fmt::format("{}", parameter_values.get<Vector>()); });
}

}  // namespace lucid
