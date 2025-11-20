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

#include "bindings/pylucid/doxygen_docstrings.h"
#include "bindings/pylucid/pylucid.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"
#include "lucid/verification/SoplexOptimiser.h"

namespace py = pybind11;
using namespace lucid;

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

const auto default_optimiser_cb = [](bool, double, const Vector&, double, double, double) {};

void init_verification(py::module_& m) {
  /**************************** Optimiser ****************************/
  py::class_<Optimiser>(m, "Optimiser", Optimiser_)
      .def(
          "solve_fourier_barrier_synthesis",
          [](const Optimiser& o, const int num_constraints, const ConstMatrixRefCopy& fxn_lattice,
             const ConstMatrixRefCopy& dn_lattice, const std::vector<Index>& x_include_mask,
             const std::vector<Index>& x_exclude_mask, const std::vector<Index>& x0_include_mask,
             const std::vector<Index>& x0_exclude_mask, const std::vector<Index>& xu_include_mask,
             const std::vector<Index>& xu_exclude_mask, const int T, const double gamma, const double eta_coeff,
             const double min_x0_coeff, const double diff_sx0_coeff, const double gamma_coeff,
             const double max_xu_coeff, const double diff_sxu_coeff, const double ebk, const double c_ebk_coeff,
             const double min_d_coeff, const double diff_d_sx_coeff, const double max_x_coeff,
             const double diff_sx_coeff, const std::optional<Optimiser::SolutionCallback>& cb) {
            // Check cb if none
            return o.solve_fourier_barrier_synthesis(FourierBarrierSynthesisProblem{.num_constraints = num_constraints,
                                                                                    .fxn_lattice = fxn_lattice,
                                                                                    .dn_lattice = dn_lattice,
                                                                                    .x_include_mask = x_include_mask,
                                                                                    .x_exclude_mask = x_exclude_mask,
                                                                                    .x0_include_mask = x0_include_mask,
                                                                                    .x0_exclude_mask = x0_exclude_mask,
                                                                                    .xu_include_mask = xu_include_mask,
                                                                                    .xu_exclude_mask = xu_exclude_mask,
                                                                                    .T = T,
                                                                                    .gamma = gamma,
                                                                                    .eta_coeff = eta_coeff,
                                                                                    .min_x0_coeff = min_x0_coeff,
                                                                                    .diff_sx0_coeff = diff_sx0_coeff,
                                                                                    .gamma_coeff = gamma_coeff,
                                                                                    .max_xu_coeff = max_xu_coeff,
                                                                                    .diff_sxu_coeff = diff_sxu_coeff,
                                                                                    .ebk = ebk,
                                                                                    .c_ebk_coeff = c_ebk_coeff,
                                                                                    .min_d_coeff = min_d_coeff,
                                                                                    .diff_d_sx_coeff = diff_d_sx_coeff,
                                                                                    .max_x_coeff = max_x_coeff,
                                                                                    .diff_sx_coeff = diff_sx_coeff},
                                                     cb.value_or(default_optimiser_cb));
          },
          py::arg("num_constraints") = 1, py::arg("fxn_lattice") = Matrix{}, py::arg("dn_lattice") = Matrix{},
          py::arg("x_include_mask") = std::vector<Index>{}, py::arg("x_exclude_mask") = std::vector<Index>{},
          py::arg("x0_include_mask") = std::vector<Index>{}, py::arg("x0_exclude_mask") = std::vector<Index>{},
          py::arg("xu_include_mask") = std::vector<Index>{}, py::arg("xu_exclude_mask") = std::vector<Index>{},
          py::arg("T") = 1, py::arg("gamma") = 1.0, py::arg("eta_coeff") = 0.0, py::arg("min_x0_coeff") = 0.0,
          py::arg("diff_sx0_coeff") = 0.0, py::arg("gamma_coeff") = 0.0, py::arg("max_xu_coeff") = 0.0,
          py::arg("diff_sxu_coeff") = 0.0, py::arg("ebk") = 0.0, py::arg("c_ebk_coeff") = 0.0,
          py::arg("min_d_coeff") = 0.0, py::arg("diff_d_sx_coeff") = 0.0, py::arg("max_x_coeff") = 0.0,
          py::arg("diff_sx_coeff") = 0.0, py::arg("cb") = py::none{}, Optimiser_solve_fourier_barrier_synthesis)
      .def_property("problem_log_file", &Optimiser::problem_log_file,
                    SETTER(Optimiser, std::string, m_problem_log_file), Optimiser_problem_log_file)
      .def_property("iis_log_file", &Optimiser::iis_log_file, SETTER(Optimiser, std::string, m_iis_log_file),
                    Optimiser_iis_log_file);
  py::class_<AlglibOptimiser, Optimiser>(m, "AlglibOptimiser", AlglibOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "");
  py::class_<HighsOptimiser, Optimiser>(m, "HighsOptimiser", HighsOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "")
      .def(py::init<std::map<std::string, std::string>, std::string, std::string>(), py::arg("options"),
           py::arg("problem_log_file") = "", py::arg("iis_log_file") = "");
  py::class_<GurobiOptimiser, Optimiser>(m, "GurobiOptimiser", GurobiOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "");
  py::class_<SoplexOptimiser, Optimiser>(m, "SoplexOptimiser", SoplexOptimiser_)
      .def(py::init<std::string, std::string>(), py::arg("problem_log_file") = "", py::arg("iis_log_file") = "");

  py::class_<FourierBarrierCertificateParameters>(m, "FourierBarrierCertificateParameters",
                                                  FourierBarrierCertificateParameters_)
      .def(py::init<>())

      .def(py::init([](const double set_scaling, const int num_particles, const double phi_local,
                       const double phi_global, const double weight, const int max_iter, const double max_vel,
                       const double ftol, const double xtol, const double C_coeff, const double epsilon,
                       const double b_norm, const double kappa, const int threads) {
             return FourierBarrierCertificateParameters{.set_scaling = set_scaling,
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
                                                        .b_norm = b_norm,
                                                        .kappa = kappa,
                                                        .threads = threads};
           }),
           py::arg("set_scaling") = 0.1, py::arg("num_particles") = 40, py::arg("phi_local") = 0.5,
           py::arg("phi_global") = 0.3, py::arg("weight") = 0.9, py::arg("max_iter") = 150, py::arg("max_vel") = 0.0,
           py::arg("ftol") = 1e-8, py::arg("xtol") = 1e-8, py::arg("C_coeff") = 1.0, py::arg("epsilon") = 1.0,
           py::arg("b_norm") = 0.0, py::arg("kappa") = 1.0, py::arg("threads") = 0,
           FourierBarrierCertificateParameters_)
      .def_readwrite("set_scaling", &FourierBarrierCertificateParameters::set_scaling,
                     FourierBarrierCertificateParameters_set_scaling)
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
      .def_readwrite("b_norm", &FourierBarrierCertificateParameters::b_norm, FourierBarrierCertificateParameters_b_norm)
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
      .def("synthesize",
           py::overload_cast<int, const Estimator&, const TruncatedFourierFeatureMap&, const RectSet&, const Set&,
                             const Set&, const FourierBarrierCertificateParameters&>(
               &FourierBarrierCertificate::synthesize),
           py::arg("lattice_resolution"), py::arg("estimator"), py::arg("feature_map"), py::arg("X_bounds"),
           py::arg("X_init"), py::arg("X_unsafe"), py::arg("parameters") = FourierBarrierCertificateParameters{},
           FourierBarrierCertificate_synthesize)
      .def("synthesize",
           py::overload_cast<const Optimiser&, int, const Estimator&, const TruncatedFourierFeatureMap&, const RectSet&,
                             const Set&, const Set&, const FourierBarrierCertificateParameters&>(
               &FourierBarrierCertificate::synthesize),
           py::arg("optimiser"), py::arg("lattice_resolution"), py::arg("estimator"), py::arg("feature_map"),
           py::arg("X_bounds"), py::arg("X_init"), py::arg("X_unsafe"),
           py::arg("parameters") = FourierBarrierCertificateParameters{}, FourierBarrierCertificate_synthesize)
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
