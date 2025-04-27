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

enum class Operator {
  VARIABLE,
  NOT,
  AND,
  OR,
  UNTIL,
  NEXT,
  ALWAYS,
  IMPLIES,
  FINALLY,
};

std::ostream& operator<<(std::ostream& os, const Operator& op);

}  // namespace lucid::ltl

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ltl::Operator);

#endif
