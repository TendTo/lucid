/**
 * @file pylucid.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include "bindings/pylucid/pylucid.h"

#include "lucid/version.h"

namespace py = pybind11;

PYBIND11_MODULE(_pylucid, m) {
  init_math(m);
  init_util(m);

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
  m.attr("MATPLOTLIB_BUILD")
#ifdef LUCID_MATPLOTLIB_BUILD
      = true;
#else
      = false;
#endif
  m.attr("GUROBI_BUILD")
#ifdef LUCID_GUROBI_BUILD
      = true;
#else
      = false;
#endif
}
