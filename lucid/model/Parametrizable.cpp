/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/Parametrizable.h"

#include "lucid/util/error.h"

namespace lucid {

template <IsAnyOf<int, double, const Vector&> T>
T Parametrizable::get(const Parameter parameter) const {
  if constexpr (std::is_same_v<T, int>) {
    return get_i(parameter);
  } else if constexpr (std::is_same_v<T, double>) {
    return get_d(parameter);
  } else if constexpr (std::is_same_v<T, const Vector&>) {
    return get_v(parameter);
  } else {
    LUCID_UNREACHABLE();
  }
}

void Parametrizable::set(const Parameter parameter, const std::variant<int, double, Vector>& value) {
  if (std::holds_alternative<int>(value)) {
    set(parameter, std::get<int>(value));
  } else if (std::holds_alternative<double>(value)) {
    set(parameter, std::get<double>(value));
  } else if (std::holds_alternative<Vector>(value)) {
    set(parameter, std::get<Vector>(value));
  } else {
    LUCID_UNREACHABLE();
  }
}
void Parametrizable::set(const Parameter parameter, const std::size_t idx,
                         const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>>& values) {
  if (std::holds_alternative<std::vector<int>>(values)) {
    set(parameter, std::get<std::vector<int>>(values).at(idx));
  } else if (std::holds_alternative<std::vector<double>>(values)) {
    set(parameter, std::get<std::vector<double>>(values).at(idx));
  } else if (std::holds_alternative<std::vector<Vector>>(values)) {
    set(parameter, std::get<std::vector<Vector>>(values).at(idx));
  } else {
    LUCID_UNREACHABLE();
  }
}
void Parametrizable::set(Parameter parameter, int) { LUCID_INVALID_HYPER_PARAMETER(parameter, "int"); }
void Parametrizable::set(Parameter parameter, double) { LUCID_INVALID_HYPER_PARAMETER(parameter, "double"); }
void Parametrizable::set(Parameter parameter, const Vector&) { LUCID_INVALID_HYPER_PARAMETER(parameter, "Vector"); }
int Parametrizable::get_i(Parameter parameter) const { LUCID_INVALID_HYPER_PARAMETER(parameter, "int"); }
double Parametrizable::get_d(Parameter parameter) const { LUCID_INVALID_HYPER_PARAMETER(parameter, "double"); }
const Vector& Parametrizable::get_v(Parameter parameter) const { LUCID_INVALID_HYPER_PARAMETER(parameter, "Vector"); }

template int Parametrizable::get<int>(Parameter parameter) const;
template double Parametrizable::get<double>(Parameter parameter) const;
template const Vector& Parametrizable::get<const Vector&>(Parameter parameter) const;

}  // namespace lucid
