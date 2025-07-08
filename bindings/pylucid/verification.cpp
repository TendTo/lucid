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
  py::class_<Optimiser>(m, "Optimiser")
      .def_property_readonly("T", &Optimiser::T)
      .def_property_readonly("gamma", &Optimiser::gamma)
      .def_property_readonly("epsilon", &Optimiser::epsilon)
      .def_property_readonly("b_norm", &Optimiser::b_norm)
      .def_property_readonly("b_kappa", &Optimiser::b_kappa)
      .def_property_readonly("sigma_f", &Optimiser::sigma_f)
      .def_property_readonly("C_coeff", &Optimiser::C_coeff)
      .def_property("problem_log_file", &Optimiser::problem_log_file,
                    [](Optimiser& self, std::string file) { self.m_problem_log_file() = std::move(file); })
      .def_property("iis_log_file", &Optimiser::iis_log_file,
                    [](Optimiser& self, std::string file) { self.m_iis_log_file() = std::move(file); });
  py::class_<AlglibOptimiser, Optimiser>(m, "AlglibOptimiser")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &AlglibOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));
  py::class_<GurobiOptimiser, Optimiser>(m, "GurobiOptimiser")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &GurobiOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));
  py::class_<HighsOptimiser, Optimiser>(m, "HighsOptimiser")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &HighsOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));
}
