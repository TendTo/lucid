/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Math module.
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=python'
#endif

#include "lucid/math/math.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "bindings/pylucid/pylucid.h"

namespace py = pybind11;
using namespace lucid;

class PyKernel : public Kernel {
 public:
  /* Inherit the constructors */
  using Kernel::Kernel;

  Scalar operator()(const Vector &x1, const Vector &x2) const override {
    PYBIND11_OVERRIDE_PURE_NAME(Scalar, Kernel, "__call__", operator(), x1, x2);
  }

  [[nodiscard]] std::unique_ptr<Kernel> clone() const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"Kernel::clone\"");
  }
  [[nodiscard]] std::unique_ptr<Kernel> clone(const Vector &) const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"Kernel::clone\"");
  }
};

void init_math(py::module_ &m) {
  py::class_<Kernel, PyKernel>(m, "Kernel")
      .def(py::init<Dimension>(), py::arg("num_params") = 0)
      .def(py::init<Vector>(), py::arg("params"))
      .def_property_readonly("parameters", &Kernel::parameters)
      .def("__call__", &Kernel::operator())
      .def("clone", py::overload_cast<>(&Kernel::clone, py::const_))
      .def("clone", py::overload_cast<const Vector &>(&Kernel::clone, py::const_), py::arg("params"));

  py::class_<GaussianKernel, Kernel>(m, "GaussianKernel")
      .def(py::init<Vector>(), py::arg("params"))
      .def(py::init<double, const Vector &>(), py::arg("sigma_f"), py::arg("sigma_l"))
      .def_property_readonly("sigma_f", &GaussianKernel::sigma_f)
      .def_property_readonly("sigma_l", &GaussianKernel::sigma_l)
      .def("__str__", STRING_LAMBDA(GaussianKernel));
}