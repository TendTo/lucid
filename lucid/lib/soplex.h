/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * SoPlex wrapper.
 * This header includes the SoPlex library and provides various helpers.
 * Other files in the library should depend on this header instead of the SoPlex library directly.
 * Instead of including <Highs.h>, include "lucid/lib/highs.h".
 */
#pragma once

#ifdef LUCID_SOPLEX_BUILD

#pragma GCC system_header

#include <soplex.h>  // IWYU pragma: export

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(soplex::VectorRational)
OSTREAM_FORMATTER(soplex::Rational)
OSTREAM_FORMATTER(soplex::SPxSolver::Status)
OSTREAM_FORMATTER(soplex::DSVectorRational)
OSTREAM_FORMATTER(soplex::SVectorRational)

#endif

#endif
