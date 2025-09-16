/**
 * @author Room 6.030
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

#include <iostream>

#include "bindings/pylucid/pylucid.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace py = pybind11;
using namespace lucid;

class DummyOptimiser : public Optimiser {
 public:
  DummyOptimiser(std::string original_solver, std::string missing_dependency)
      : Optimiser(1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) {
    LUCID_NOT_SUPPORTED_MISSING_RUNTIME_DEPENDENCY(original_solver, missing_dependency);
  }
};

class PyBarrierCertificate : public BarrierCertificate {
 public:
  using BarrierCertificate::BarrierCertificate;
  [[nodiscard]] double apply_impl(ConstVectorRef x) const override {
    PYBIND11_OVERRIDE_PURE(double, BarrierCertificate, apply_impl, x);
  }
  [[nodiscard]] std::unique_ptr<BarrierCertificate> clone() const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"BarrierCertificate::clone\"");
  }
};

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
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &AlglibOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));
  py::class_<HighsOptimiser, Optimiser>(m, "HighsOptimiser")
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &HighsOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));
  py::class_<GurobiOptimiser, Optimiser>(m, "GurobiOptimiser")
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &GurobiOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));

  py::class_<BarrierCertificate, PyBarrierCertificate>(m, "BarrierCertificate")
      .def_property_readonly("T", &BarrierCertificate::T)
      .def_property_readonly("gamma", &BarrierCertificate::gamma)
      .def_property_readonly("eta", &BarrierCertificate::eta)
      .def_property_readonly("c", &BarrierCertificate::c)
      .def_property_readonly("norm", &BarrierCertificate::norm)
      .def_property_readonly("is_synthesized", &BarrierCertificate::is_synthesized)
      .def_property_readonly("safety", &BarrierCertificate::safety)
      .def_property_readonly("__call__", &BarrierCertificate::operator())
      .def("__str__", STRING_LAMBDA(BarrierCertificate));
  py::class_<FourierBarrierCertificate, BarrierCertificate>(m, "FourierBarrierCertificate")
      .def(py::init<int, double, double, double>(), py::arg("T"), py::arg("gamma"), py::arg("eta") = 0.0,
           py::arg("c") = 0.0)
      .def("synthesize",
           py::overload_cast<ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef,
                             const TruncatedFourierFeatureMap&, Dimension, double, double, double, double>(
               &FourierBarrierCertificate::synthesize),
           py::arg("fx_lattice"), py::arg("fxp_lattice"), py::arg("fx0_lattice"), py::arg("fxu_lattice"),
           py::arg("feature_map"), py::arg("num_frequency_samples_per_dim"), py::arg("c_coeff") = 1.0,
           py::arg("epsilon") = 0.0, py::arg("target_norm") = 1.0, py::arg("b_kappa") = 1.0)
      .def("synthesize",
           py::overload_cast<const Optimiser&, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef,
                             const TruncatedFourierFeatureMap&, Dimension, double, double, double, double>(
               &FourierBarrierCertificate::synthesize),
           py::arg("optimiser"), py::arg("fx_lattice"), py::arg("fxp_lattice"), py::arg("fx0_lattice"),
           py::arg("fxu_lattice"), py::arg("feature_map"), py::arg("num_frequency_samples_per_dim"),
           py::arg("c_coeff") = 1.0, py::arg("epsilon") = 0.0, py::arg("target_norm") = 1.0, py::arg("b_kappa") = 1.0)
      .def_property_readonly("coefficients", &FourierBarrierCertificate::coefficients)
      .def("__str__", STRING_LAMBDA(FourierBarrierCertificate));
}
