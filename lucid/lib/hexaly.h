/**
* @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Hexaly wrapper.
 * This header includes the hexaly library and provides a various helpers.
 * Other files in the library should depend on this header instead of the hexaly library directly.
 * Instead of including <optimizer/hexalyoptimizer.h>, include "lucid/lib/hexaly.h".
 */
#pragma once

#ifdef LUCID_HEXALY_BUILD

#include <optimizer/hexalyoptimizer.h>

namespace lucid {}  // namespace lucid

#endif
