/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#ifdef LUCID_ALGLIB_BUILD

#include "lucid/lib/alglib.h"

#include <cassert>
#include <iostream>
#include <ostream>
#include <span>

namespace lucid {

std::ostream& operator<<(std::ostream& os, const std::span<const double>& data) {
  os << "[";
  for (std::size_t i = 0; i < data.size(); ++i) {
    os << data[i];
    if (i < data.size() - 1) {
      os << ", ";
    }
  }
  return os << "]";
}
std::ostream& operator<<(std::ostream& os, const alglib::sparsematrix& matrix) {
  const alglib::ae_int_t nrows = alglib::sparsegetnrows(matrix);
  const alglib::ae_int_t ncols = alglib::sparsegetncols(matrix);
  os << "Sparse matrix (" << nrows << "x" << ncols << "): [" << std::endl;
  for (alglib::ae_int_t i = 0; i < nrows; ++i) {
    for (alglib::ae_int_t j = 0; j < ncols; ++j) {
      os << alglib::sparseget(matrix, i, j) << "\t";
    }
    os << std::endl;
  }
  return os << "]";
}
std::ostream& operator<<(std::ostream& os, const alglib::real_1d_array& array) {
  const std::span<const double> data(array.getcontent(), array.length());
  return os << "Real 1D array (length=" << data.size() << "): " << data;
}
void print_lp(const alglib::sparsematrix& matrix, const alglib::real_1d_array& lb, const alglib::real_1d_array& ub) {
  assert(lb.length() == ub.length());
  const alglib::ae_int_t nrows = alglib::sparsegetnrows(matrix);
  const alglib::ae_int_t ncols = alglib::sparsegetncols(matrix);
  assert(nrows == lb.length());
  for (alglib::ae_int_t i = 0; i < nrows; ++i) {
    std::cout << "Row " << i << ": ";
    for (alglib::ae_int_t j = 0; j < ncols; ++j) {
      std::cout << alglib::sparseget(matrix, i, j) << "\t";
    }
    if (lb[i] == alglib::fp_neginf && ub[i] == alglib::fp_posinf) {
      std::cout << " (no bounds)";
    } else if (lb[i] == alglib::fp_neginf) {
      std::cout << " <= " << ub[i];
    } else if (ub[i] == alglib::fp_posinf) {
      std::cout << " >= " << lb[i];
    } else {
      std::cout << " = " << lb[i];
    }
    std::cout << std::endl;
  }
}

}  // namespace lucid

#endif
