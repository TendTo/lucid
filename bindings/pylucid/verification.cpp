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

#include "bindings/pylucid/doxygen_docstrings.h"
#include "bindings/pylucid/pylucid.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"
#include "lucid/verification/SoplexOptimiser.h"

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
  py::class_<Optimiser>(m, "Optimiser", Optimiser_)
      .def_property_readonly("T", &Optimiser::T, Optimiser_T)
      .def_property_readonly("gamma", &Optimiser::gamma, Optimiser_gamma)
      .def_property_readonly("epsilon", &Optimiser::epsilon, Optimiser_epsilon)
      .def_property_readonly("b_norm", &Optimiser::b_norm, Optimiser_b_norm)
      .def_property_readonly("b_kappa", &Optimiser::b_kappa, Optimiser_b_kappa)
      .def_property_readonly("sigma_f", &Optimiser::sigma_f, Optimiser_sigma_f)
      .def_property_readonly("C_coeff", &Optimiser::C_coeff, Optimiser_C_coeff)
      .def_property(
          "problem_log_file", &Optimiser::problem_log_file,
          [](Optimiser& self, std::string file) { self.m_problem_log_file() = std::move(file); },
          Optimiser_problem_log_file)
      .def_property(
          "iis_log_file", &Optimiser::iis_log_file,
          [](Optimiser& self, std::string file) { self.m_iis_log_file() = std::move(file); }, Optimiser_iis_log_file);
  py::class_<AlglibOptimiser, Optimiser>(m, "AlglibOptimiser", AlglibOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &AlglibOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"),
           AlglibOptimiser_solve);
  py::class_<HighsOptimiser, Optimiser>(m, "HighsOptimiser", HighsOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<std::map<std::string, std::string>, std::string, std::string>(), py::arg("options"),
           py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &HighsOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"),
           HighsOptimiser_solve);
  py::class_<GurobiOptimiser, Optimiser>(m, "GurobiOptimiser", GurobiOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<int, double, double, double, double, double, double, std::string, std::string>(), py::arg("T"),
           py::arg("gamma"), py::arg("epsilon"), py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"),
           py::arg("C_coeff") = 1.0, py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def("solve", &GurobiOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"),
           GurobiOptimiser_solve);
  py::class_<SoplexOptimiser, Optimiser>(m, "SoplexOptimiser", SoplexOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "");

  py::class_<FourierBarrierCertificateParameters>(m, "FourierBarrierCertificateParameters",
                                                  FourierBarrierCertificateParameters_)
      .def(py::init<>())

      .def(py::init([](const double increase, const int num_particles, const double phi_local, const double phi_global,
                       const double weight, const int max_iter, const double max_vel, const double ftol,
                       const double xtol, const double C_coeff, const double epsilon, const double target_norm,
                       const double kappa, const int threads) {
             return FourierBarrierCertificateParameters{.increase = increase,
                                                        .num_particles = num_particles,
                                                        .phi_local = phi_local,
                                                        .phi_global = phi_global,
                                                        .weight = weight,
                                                        .max_iter = max_iter,
                                                        .max_vel = max_vel,
                                                        .ftol = ftol,
                                                        .xtol = xtol,
                                                        .C_coeff = C_coeff,
                                                        .epsilon = epsilon,
                                                        .target_norm = target_norm,
                                                        .kappa = kappa,
                                                        .threads = threads};
           }),
           py::arg("increase") = 0.1, py::arg("num_particles") = 40, py::arg("phi_local") = 0.5,
           py::arg("phi_global") = 0.3, py::arg("weight") = 0.9, py::arg("max_iter") = 150, py::arg("max_vel") = 0.0,
           py::arg("ftol") = 1e-8, py::arg("xtol") = 1e-8, py::arg("C_coeff") = 1.0, py::arg("epsilon") = 1.0,
           py::arg("target_norm") = 0.0, py::arg("kappa") = 1.0, py::arg("threads") = 0,
           FourierBarrierCertificateParameters_)
      .def_readwrite("increase", &FourierBarrierCertificateParameters::increase,
                     FourierBarrierCertificateParameters_increase)
      .def_readwrite("num_particles", &FourierBarrierCertificateParameters::num_particles,
                     FourierBarrierCertificateParameters_num_particles)
      .def_readwrite("phi_local", &FourierBarrierCertificateParameters::phi_local,
                     FourierBarrierCertificateParameters_phi_local)
      .def_readwrite("phi_global", &FourierBarrierCertificateParameters::phi_global,
                     FourierBarrierCertificateParameters_phi_global)
      .def_readwrite("weight", &FourierBarrierCertificateParameters::weight, FourierBarrierCertificateParameters_weight)
      .def_readwrite("max_iter", &FourierBarrierCertificateParameters::max_iter,
                     FourierBarrierCertificateParameters_max_iter)
      .def_readwrite("max_vel", &FourierBarrierCertificateParameters::max_vel,
                     FourierBarrierCertificateParameters_max_vel)
      .def_readwrite("ftol", &FourierBarrierCertificateParameters::ftol, FourierBarrierCertificateParameters_ftol)
      .def_readwrite("xtol", &FourierBarrierCertificateParameters::xtol, FourierBarrierCertificateParameters_xtol)
      .def_readwrite("C_coeff", &FourierBarrierCertificateParameters::C_coeff,
                     FourierBarrierCertificateParameters_C_coeff)
      .def_readwrite("epsilon", &FourierBarrierCertificateParameters::epsilon,
                     FourierBarrierCertificateParameters_epsilon)
      .def_readwrite("target_norm", &FourierBarrierCertificateParameters::target_norm,
                     FourierBarrierCertificateParameters_target_norm)
      .def_readwrite("kappa", &FourierBarrierCertificateParameters::kappa, FourierBarrierCertificateParameters_kappa)
      .def_readwrite("threads", &FourierBarrierCertificateParameters::threads,
                     FourierBarrierCertificateParameters_threads)
      .def("__str__", STRING_LAMBDA(FourierBarrierCertificateParameters));

  /**************************** BarrierCertificate ****************************/
  py::class_<BarrierCertificate, PyBarrierCertificate>(m, "BarrierCertificate", BarrierCertificate_)
      .def_property_readonly("T", &BarrierCertificate::T, BarrierCertificate_T)
      .def_property_readonly("gamma", &BarrierCertificate::gamma, BarrierCertificate_gamma)
      .def_property_readonly("eta", &BarrierCertificate::eta, BarrierCertificate_eta)
      .def_property_readonly("c", &BarrierCertificate::c, BarrierCertificate_c)
      .def_property_readonly("norm", &BarrierCertificate::norm, BarrierCertificate_norm)
      .def_property_readonly("is_synthesized", &BarrierCertificate::is_synthesized, BarrierCertificate_is_synthesized)
      .def_property_readonly("safety", &BarrierCertificate::safety, BarrierCertificate_safety)
      .def("__call__", &BarrierCertificate::operator(), py::arg("x"), BarrierCertificate_operator_apply)
      .def("__str__", STRING_LAMBDA(BarrierCertificate));
  py::class_<FourierBarrierCertificate, BarrierCertificate>(m, "FourierBarrierCertificate", FourierBarrierCertificate_)
      .def(py::init<int, double, double, double>(), py::arg("T"), py::arg("gamma"), py::arg("eta") = 0.0,
           py::arg("c") = 0.0)
      .def("compute_A_periodic_minus_x", &FourierBarrierCertificate::compute_A_periodic_minus_x, py::arg("Q_tilde"),
           py::arg("f_max"), py::arg("X_tilde"), py::arg("X"),
           py::arg("parameters") = FourierBarrierCertificateParameters{})
      .def("synthesize",
           py::overload_cast<int, const Estimator&, const TruncatedFourierFeatureMap&, const RectSet&, const Set&,
                             const Set&, const FourierBarrierCertificateParameters&>(
               &FourierBarrierCertificate::synthesize),
           py::arg("Q_tilde"), py::arg("estimator"), py::arg("feature_map"), py::arg("X_bounds"), py::arg("X_init"),
           py::arg("X_unsafe"), py::arg("parameters") = FourierBarrierCertificateParameters{},
           FourierBarrierCertificate_synthesize)
      .def("synthesize",
           py::overload_cast<const Optimiser&, int, const Estimator&, const TruncatedFourierFeatureMap&, const RectSet&,
                             const Set&, const Set&, const FourierBarrierCertificateParameters&>(
               &FourierBarrierCertificate::synthesize),
           py::arg("optimiser"), py::arg("Q_tilde"), py::arg("estimator"), py::arg("feature_map"), py::arg("X_bounds"),
           py::arg("X_init"), py::arg("X_unsafe"), py::arg("parameters") = FourierBarrierCertificateParameters{},
           FourierBarrierCertificate_synthesize)
      .def("synthesize",
           py::overload_cast<ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef,
                             const TruncatedFourierFeatureMap&, Dimension, double, double, double, double>(
               &FourierBarrierCertificate::synthesize),
           py::arg("fx_lattice"), py::arg("fxp_lattice"), py::arg("fx0_lattice"), py::arg("fxu_lattice"),
           py::arg("feature_map"), py::arg("num_frequency_samples_per_dim"), py::arg("c_coeff") = 1.0,
           py::arg("epsilon") = 0.0, py::arg("target_norm") = 1.0, py::arg("b_kappa") = 1.0,
           FourierBarrierCertificate_synthesize)
      .def("synthesize",
           py::overload_cast<const Optimiser&, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef,
                             const TruncatedFourierFeatureMap&, Dimension, double, double, double, double>(
               &FourierBarrierCertificate::synthesize),
           py::arg("optimiser"), py::arg("fx_lattice"), py::arg("fxp_lattice"), py::arg("fx0_lattice"),
           py::arg("fxu_lattice"), py::arg("feature_map"), py::arg("num_frequency_samples_per_dim"),
           py::arg("c_coeff") = 1.0, py::arg("epsilon") = 0.0, py::arg("target_norm") = 1.0, py::arg("b_kappa") = 1.0,
           FourierBarrierCertificate_synthesize)
      .def_property_readonly("coefficients", &FourierBarrierCertificate::coefficients,
                             FourierBarrierCertificate_coefficients)
      .def("__str__", STRING_LAMBDA(FourierBarrierCertificate));

  /**************************** MontecarloSimulation ****************************/
  py::class_<MontecarloSimulation>(m, "MontecarloSimulation", MontecarloSimulation_)
      .def(py::init<>(), MontecarloSimulation_MontecarloSimulation)
      .def("safety_probability", &MontecarloSimulation::safety_probability, py::arg("X_bounds"), py::arg("X_init"),
           py::arg("X_unsafe"), py::arg("system_dynamics"), py::arg("time_horizon"), py::arg("confidence_level") = 0.9,
           py::arg("num_samples") = 1000, MontecarloSimulation_safety_probability);
}
