/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * ParameterValues class.
 */
#include "lucid/model/ParameterValues.h"

#include <ranges>

#include "lucid/util/logging.h"

namespace lucid {

bool ParameterValues::operator==(const ParameterValues& o) const {
  if (parameter_ != o.parameter_) return false;
  return dispatch<bool>(
      parameter_, [this, &o]() -> bool { return get<int>() == o.get<int>(); },
      [this, &o]() -> bool { return get<double>() == o.get<double>(); },
      [this, &o]() -> bool { return get<Vector>() == o.get<Vector>(); });
}

std::string ParameterValues::to_string() const {
  std::string values_str = dispatch<std::string>(
      parameter_, [this]() -> std::string { return fmt::format("{}", get<int>()); },
      [this]() -> std::string { return fmt::format("{}", get<double>()); },
      [this]() -> std::string { return fmt::format("{}", get<Vector>()); });
  return fmt::format("ParameterValues( {} values( {} )", parameter_, values_str);
}

std::ostream& operator<<(std::ostream& os, const ParameterValues& parameter_values) {
  return os << parameter_values.to_string();
}

}  // namespace lucid
