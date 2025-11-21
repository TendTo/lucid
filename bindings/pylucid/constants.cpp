/**
 * @file pylucid.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include "lucid/util/constants.h"

#include <pybind11/pybind11.h>

#include "lucid/version.h"

namespace py = pybind11;

PYBIND11_MODULE(_constants, m) {
#ifdef LUCID_DESCRIPTION
  m.doc() = "Constants for the lucid library: " LUCID_DESCRIPTION;
#else
#error "LUCID_DESCRIPTION is not defined"
#endif
#ifdef LUCID_VERSION_STRING
  m.attr("__version__") = LUCID_VERSION_STRING;
#else
#error "LUCID_VERSION_STRING is not defined"
#endif
  m.attr("MATPLOTLIB_BUILD") = lucid::constants::MATPLOTLIB_BUILD;
  m.attr("GUROBI_BUILD") = lucid::constants::GUROBI_BUILD;
  m.attr("ALGLIB_BUILD") = lucid::constants::ALGLIB_BUILD;
  m.attr("HIGHS_BUILD") = lucid::constants::HIGHS_BUILD;
  m.attr("SOPLEX_BUILD") = lucid::constants::SOPLEX_BUILD;
  m.attr("PSOCPP_BUILD") = lucid::constants::PSOCPP_BUILD;
  m.attr("OMP_BUILD") = lucid::constants::OMP_BUILD;
  m.attr("CUDA_BUILD") = lucid::constants::CUDA_BUILD;
  m.attr("DEBUG_BUILD") = lucid::constants::DEBUG_BUILD;
  m.attr("RELEASE_BUILD") = lucid::constants::RELEASE_BUILD;
  m.attr("ASSERT_CHECKS_ENABLED") = lucid::constants::ASSERT_CHECKS_ENABLED;
  m.attr("RUNTIME_CHECKS_ENABLED") = lucid::constants::RUNTIME_CHECKS_ENABLED;
  m.attr("LOG_ENABLED") = lucid::constants::LOG_ENABLED;
}
