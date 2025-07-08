/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HiGHS wrapper.
 * This header includes the gurobi library and provides various helpers.
 * Other files in the library should depend on this header instead of the gurobi library directly.
 * Instead of including <Highs.h>, include "lucid/lib/highs.h".
 */
#pragma once

#ifdef LUCID_HIGHS_BUILD

#include <Highs.h>

#endif