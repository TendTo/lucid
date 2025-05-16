/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Math module.
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include "lucid/verification/verification.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace lucid;

void init_verification(py::module_ &m) {
  /**************************** Optimiser ****************************/
  py::class_<GurobiLinearOptimiser>(m, "GurobiLinearOptimiser")
      .def(py::init<int, double, double, double, double, double>(), py::arg("T"), py::arg("gamma"), py::arg("epsilon"),
           py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"))
      .def("solve", &GurobiLinearOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));
}