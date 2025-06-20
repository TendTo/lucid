/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * alglib wrapper.
 * This header includes the alglib library and provides a various helpers.
 * Other files in the library should depend on this header instead of the alglib library directly.
 * Instead of including <alglib/optimization.h>, include "lucid/lib/alglib.h".
 */
#pragma once

#ifndef LUCID_ALGIB_BUILD
#include <alglib/optimization.h>
#include <alglib/stdafx.h>
#include <alglib/linalg.h>
#endif

namespace lucid {}  // namespace lucid
