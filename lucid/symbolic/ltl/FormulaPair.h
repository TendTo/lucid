/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Operator enum.
 */
#pragma once

#include <iosfwd>

namespace lucid::ltl {

// Forward declaration
class Formula;

struct FormulaPair {
  const Formula& lhs;
  const Formula& rhs;
};

std::ostream& operator<<(std::ostream& os, const FormulaPair& pair);

}  // namespace lucid::ltl
