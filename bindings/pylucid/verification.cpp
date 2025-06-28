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

void init_verification(py::module_& m) {
  /**************************** Optimiser ****************************/
  py::class_<Optimiser>(m, "Optimiser").def(py::init<>());
  py::class_<AlglibOptimiser, Optimiser>(m, "AlglibOptimiser")
      .def(py::init<int, double, double, double, double, double, double>(), py::arg("T"), py::arg("gamma"),
           py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"), py::arg("C_coeff") = 1.0)
      .def("solve", &AlglibOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"))
      .def_property_readonly("T", &AlglibOptimiser::T)
      .def_property_readonly("gamma", &AlglibOptimiser::gamma)
      .def_property_readonly("epsilon", &AlglibOptimiser::epsilon)
      .def_property_readonly("b_norm", &AlglibOptimiser::b_norm)
      .def_property_readonly("b_kappa", &AlglibOptimiser::b_kappa)
      .def_property_readonly("sigma_f", &AlglibOptimiser::sigma_f)
      .def_property_readonly("C_coeff", &AlglibOptimiser::C_coeff);
  py::class_<GurobiOptimiser, Optimiser>(m, "GurobiOptimiser")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &GurobiOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"))
      .def_property_readonly("T", &GurobiOptimiser::T)
      .def_property_readonly("gamma", &GurobiOptimiser::gamma)
      .def_property_readonly("epsilon", &GurobiOptimiser::epsilon)
      .def_property_readonly("b_norm", &GurobiOptimiser::b_norm)
      .def_property_readonly("b_kappa", &GurobiOptimiser::b_kappa)
      .def_property_readonly("sigma_f", &GurobiOptimiser::sigma_f)
      .def_property_readonly("C_coeff", &GurobiOptimiser::C_coeff)
      .def_property("problem_log_file", &GurobiOptimiser::problem_log_file,
                    [](GurobiOptimiser& self, std::string file) { self.m_problem_log_file() = std::move(file); })
      .def_property("iis_log_file", &GurobiOptimiser::iis_log_file,
                    [](GurobiOptimiser& self, std::string file) { self.m_iis_log_file() = std::move(file); });
}