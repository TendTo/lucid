#include "lucid/symbolic/ltl/Operator.h"

#include <ostream>

#include "lucid/util/error.h"

namespace lucid::ltl {

std::ostream& operator<<(std::ostream& os, const Operator& op) {
  switch (op) {
    case Operator::NOT:
      return os << "!";
    case Operator::AND:
      return os << "&";
    case Operator::OR:
      return os << "|";
    case Operator::UNTIL:
      return os << "U";
    case Operator::NEXT:
      return os << "X";
    case Operator::ALWAYS:
      return os << "G";
    case Operator::IMPLIES:
      return os << "->";
    case Operator::FINALLY:
      return os << "F";
    default:
      LUCID_UNREACHABLE();
  }
}

}  // namespace lucid::ltl
