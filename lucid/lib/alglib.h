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

/**
 * Output operator for ALGLIB sparse matrix.
 * @param os output stream
 * @param matrix ALGLIB sparse matrix to output
 * @return reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const alglib::sparsematrix& matrix);

/**
 * Output operator for ALGLIB real array.
 * @param os output stream
 * @param array ALGLIB real array to output
 * @return reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const alglib::real_1d_array& array);

/**
 * Print a linear programming problem in human-readable format.
 * @param matrix constraint matrix
 * @param lb lower bounds vector
 * @param ub upper bounds vector
 */
void print_lp(const alglib::sparsematrix& matrix, const alglib::real_1d_array& lb, const alglib::real_1d_array& ub);

}  // namespace lucid

#endif
