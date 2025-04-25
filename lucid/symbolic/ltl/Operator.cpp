#include "lucid/symbolic/ltl/Operator.h"

#include <ostream>

#include "lucid/util/error.h"

namespace lucid::ltl {

std::ostream& operator<<(std::ostream& os, const Operator& op) {
  switch (op) {
    case Operator::NOT:
      return os << "NOT";
    case Operator::AND:
      return os << "AND";
    case Operator::OR:
      return os << "OR";
    case Operator::UNTIL:
      return os << "UNTIL";
    case Operator::NEXT:
      return os << "NEXT";
    case Operator::ALWAYS:
      return os << "ALWAYS";
    case Operator::EVENTUALLY:
      return os << "EVENTUALLY";
    case Operator::IMPLIES:
      return os << "IMPLIES";
    default:
      LUCID_UNREACHABLE();
  }
}

}  // namespace lucid::ltl
