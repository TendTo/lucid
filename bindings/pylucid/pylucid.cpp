/**
 * @file pylucid.cpp
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include "bindings/pylucid/pylucid.h"

#include "lucid/util/constants.h"
#include "lucid/version.h"

namespace py = pybind11;

PYBIND11_MODULE(_pylucid, m) {
  init_model(m);
  init_util(m);
  init_verification(m);

#ifdef LUCID_DESCRIPTION
  m.doc() = LUCID_DESCRIPTION;
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
}
