/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/symbolic/ltl/FormulaPair.h"

#include <ostream>

#include "lucid/symbolic/ltl/Formula.h"

namespace lucid::ltl {

inline std::ostream& operator<<(std::ostream& os, const FormulaPair& pair) { return os << pair.lhs << " " << pair.rhs; }

}  // namespace lucid::ltl
