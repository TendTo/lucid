/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/symbolic/ltl/AtomicProposition.h"

#include <atomic>
#include <limits>
#include <ostream>

#include "lucid/util/error.h"

namespace lucid::ltl {
std::vector<std::string> AtomicProposition::names_{{"dummy"}};
const AtomicProposition::Id AtomicProposition::dummy_id{std::numeric_limits<Id>::max()};

AtomicProposition::Id AtomicProposition::GetNextId() {
  static std::atomic<Id> next_id{0};
  return next_id.fetch_add(1);
}
AtomicProposition::AtomicProposition(const Id id) : id_{id} {
  LUCID_CHECK_ARGUMENT_EXPECTED(id < names_.size(), "id", id, fmt::format("< {}", names_.size()));
}

AtomicProposition::AtomicProposition(std::string name) : id_{GetNextId()} {
  LUCID_ASSERT(id_ < std::numeric_limits<Id>::max(), "The ID of the variable has reached the maximum value.");
  names_.push_back(std::move(name));
  LUCID_ASSERT(names_.size() == id_ + 2u, "The size of the names vector is not equal to the ID + dummy string.");
}

std::ostream &operator<<(std::ostream &os, const AtomicProposition &var) { return os << var.name(); }

}  // namespace lucid::ltl