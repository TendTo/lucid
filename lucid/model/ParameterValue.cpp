/**
 * @author c3054737
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
std::ostream& operator<<(std::ostream& os, const ParameterValue& parameter_value) {
  os << "ParameterValue( " << parameter_value.parameter() << " value( ";
  dispatch<std::ostream&>(
      parameter_value.parameter(),
      [&os, &parameter_value]() -> std::ostream& { return os << parameter_value.get<int>(); },
      [&os, &parameter_value]() -> std::ostream& { return os << parameter_value.get<double>(); },
      [&os, &parameter_value]() -> std::ostream& { return os << parameter_value.get<Vector>(); });
  return os << " )";
}

}  // namespace lucid
