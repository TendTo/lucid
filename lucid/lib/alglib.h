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

#ifdef LUCID_ALGLIB_BUILD

#include <alglib/linalg.h>
#include <alglib/optimization.h>
#include <alglib/stdafx.h>

#include <iosfwd>

namespace lucid {

std::ostream& operator<<(std::ostream& os, const alglib::sparsematrix& matrix);
std::ostream& operator<<(std::ostream& os, const alglib::real_1d_array& array);
void print_lp(const alglib::sparsematrix& matrix, const alglib::real_1d_array& lb, const alglib::real_1d_array& ub);

}  // namespace lucid

#endif
