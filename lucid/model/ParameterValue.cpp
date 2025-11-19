/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HyperParameterValue class.
 */
#include "lucid/model/ParameterValue.h"

namespace lucid {

bool ParameterValue::operator==(const ParameterValue& o) const {
  if (parameter_ != o.parameter_) return false;
  return dispatch<bool>(
      parameter_, [this, &o]() -> bool { return std::get<int>(o.value_) == std::get<int>(value_); },
      [this, &o]() -> bool { return std::get<double>(o.value_) == std::get<double>(value_); },
      [this, &o]() -> bool { return std::get<Vector>(o.value_) == std::get<Vector>(value_); });
}

std::string ParameterValue::to_string() const {
  std::string value_str = dispatch<std::string>(
      parameter_, [this]() -> std::string { return fmt::format("{}", get<int>()); },
      [this]() -> std::string { return fmt::format("{}", get<double>()); },
      [this]() -> std::string { return fmt::format("{}", get<Vector>()); });
  return fmt::format("ParameterValue( {} value( {} )", parameter_, value_str);
}

std::ostream& operator<<(std::ostream& os, const ParameterValue& parameter_value) {
  return os << parameter_value.to_string();
}

}  // namespace lucid
