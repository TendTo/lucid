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

#include "lucid/math/math.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "bindings/pylucid/pylucid.h"
#include "lucid/util/error.h"

namespace py = pybind11;
using namespace lucid;

class PyKernel : public Kernel {
 public:
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

class PyRegression : public Regression {
 public:
  using Regression::Regression;
  [[nodiscard]] Matrix operator()(ConstMatrixRef x) const override {
    PYBIND11_OVERRIDE_PURE_NAME(Matrix, Regression, "__call__", operator(), x);
  }
  [[nodiscard]] Matrix operator()(ConstMatrixRef x, const FeatureMap &feature_map) const override {
    PYBIND11_OVERRIDE_PURE_NAME(Matrix, Regression, "__call__", operator(), x, feature_map);
  }
};

class PySet : public Set {
 public:
  using Set::Set;
  [[nodiscard]] Dimension dimension() const override { PYBIND11_OVERRIDE_PURE(Dimension, Set, dimension); }
  [[nodiscard]] Matrix sample_element(Index num_samples) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Set, sample_element, num_samples);
  }
  [[nodiscard]] bool operator()(ConstMatrixRef x) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, Set, "__call__", operator(), x);
  }
  [[nodiscard]] Matrix lattice(const Eigen::VectorX<Index> &points_per_dim, bool include_endpoints) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Set, lattice, points_per_dim, include_endpoints);
  }
  void plot(const std::string &color) const override { PYBIND11_OVERRIDE_PURE(void, Set, plot, color); }
  void plot3d(const std::string &color) const override { PYBIND11_OVERRIDE_PURE(void, Set, plot3d, color); }
};

class MultiSetIterator {
 public:
  explicit MultiSetIterator(const MultiSet &multi_set, std::size_t index = 0) : multi_set_{multi_set}, index_{index} {}
  const Set &operator*() const { return *multi_set_.sets().at(index_); }
  MultiSetIterator &operator++() {
    ++index_;
    return *this;
  }
  bool operator==(const MultiSetIterator &o) const { return &multi_set_ == &o.multi_set_ && index_ == o.index_; }

 private:
  const MultiSet &multi_set_;
  std::size_t index_;
};

void init_math(py::module_ &m) {
  /**************************** Kernel ****************************/
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
      .def(py::init<double, double, Dimension>(), py::arg("sigma_f"), py::arg("sigma_l"), py::arg("dimension"))
      .def_property_readonly("sigma_f", &GaussianKernel::sigma_f)
      .def_property_readonly("sigma_l", &GaussianKernel::sigma_l)
      .def("__str__", STRING_LAMBDA(GaussianKernel));

  /**************************** FeatureMap ****************************/
  py::class_<FeatureMap>(m, "FeatureMap");
  py::class_<TruncatedFourierFeatureMap, FeatureMap>(m, "TruncatedFourierFeatureMap", py::is_final())
      .def(py::init<long, Dimension, ConstVectorRef, Scalar, Matrix>(), py::arg("num_frequencies"),
           py::arg("input_dimension"), py::arg("sigma_l"), py::arg("sigma_f"), py::arg("x_limits"))
      .def(py::init<long, Dimension, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"),
           py::arg("input_dimension"), py::arg("sigma_l"), py::arg("sigma_f"), py::arg("x_limits"))
      .def("map_vector", &TruncatedFourierFeatureMap::map_vector, py::arg("x"))
      .def("map_matrix", &TruncatedFourierFeatureMap::map_matrix, py::arg("x"))
      .def("__call__", &TruncatedFourierFeatureMap::operator(), py::arg("x"))
      .def_property_readonly("dimension", &TruncatedFourierFeatureMap::dimension)
      .def_property_readonly("omega", &TruncatedFourierFeatureMap::omega)
      .def_property_readonly("weights", &TruncatedFourierFeatureMap::weights)
      .def_property_readonly("num_frequencies", &TruncatedFourierFeatureMap::num_frequencies);

  /**************************** Regression ****************************/
  py::class_<Regression, PyRegression>(m, "Regression")
      .def("__call__", py::overload_cast<ConstMatrixRef>(&Regression::operator(), py::const_), py::arg("x"))
      .def("__call__", py::overload_cast<ConstMatrixRef, const FeatureMap &>(&Regression::operator(), py::const_),
           py::arg("x"), py::arg("feature_map"));
  py::class_<KernelRidgeRegression<GaussianKernel>, Regression>(m, "GaussianKernelRidgeRegression")
      .def(py::init<GaussianKernel, Matrix, ConstMatrixRef, Scalar>(), py::arg("kernel"), py::arg("training_inputs"),
           py::arg("training_outputs"), py::arg("regularization_constant") = 0)
      .def_property_readonly("kernel", &KernelRidgeRegression<GaussianKernel>::kernel)
      .def_property_readonly("training_inputs", &KernelRidgeRegression<GaussianKernel>::training_inputs)
      .def_property_readonly("coefficients", &KernelRidgeRegression<GaussianKernel>::coefficients);

  /**************************** Optimiser ****************************/
  py::class_<GurobiLinearOptimiser>(m, "GurobiLinearOptimiser")
      .def(py::init<int, double, double, double, double, double>(), py::arg("T"), py::arg("gamma"), py::arg("epsilon"),
           py::arg("b_norm"), py::arg("b_kappa"), py::arg("sigma_f"))
      .def("solve", &GurobiLinearOptimiser::solve, py::arg("f0_lattice"), py::arg("fu_lattice"), py::arg("phi_mat"),
           py::arg("w_mat"), py::arg("rkhs_dim"), py::arg("num_frequencies_per_dim"),
           py::arg("num_frequency_samples_per_dim"), py::arg("original_dim"), py::arg("callback"));

  /**************************** Set ****************************/
  py::class_<Set, PySet>(m, "Set")
      .def(py::init<>())
      .def_property_readonly("dimension", &Set::dimension)
      .def("sample_element", py::overload_cast<Index>(&Set::sample_element, py::const_), py::arg("num_samples"))
      .def("sample_element", py::overload_cast<>(&Set::sample_element, py::const_))
      .def("lattice", py::overload_cast<Index, bool>(&Set::lattice, py::const_), py::arg("points_per_dim"),
           py::arg("include_endpoints") = false)
      .def("lattice", py::overload_cast<const Eigen::VectorX<Index> &, bool>(&Set::lattice, py::const_),
           py::arg("points_per_dim"), py::arg("include_endpoints"))
      .def("plot", &Set::plot, py::arg("color"))
      .def("plot3d", &Set::plot3d, py::arg("color"))
      .def("__contains__", &Set::contains, py::arg("x"))
      .def("__call__", &Set::operator(), py::arg("x"))
      .def("__str__", STRING_LAMBDA(Set));
  py::class_<RectSet, Set>(m, "RectSet")
      .def(py::init<Vector, Vector, int>(), py::arg("lb"), py::arg("ub"), py::arg("seed") = -1)
      .def(py::init<std::vector<std::pair<Scalar, Scalar>>, int>(), py::arg("bounds"), py::arg("seed") = -1)
      .def_property_readonly("lower_bound", &RectSet::lower_bound)
      .def_property_readonly("upper_bound", &RectSet::upper_bound)
      .def("__str__", STRING_LAMBDA(RectSet));
  py::class_<MultiSet, Set>(m, "MultiSet")
      .def(py::init([](const py::args &sets) {
        std::vector<std::unique_ptr<Set>> unique_sets;
        for (const auto &set : sets) {
          if (py::isinstance<RectSet>(set)) {
            unique_sets.emplace_back(std::make_unique<RectSet>(set.cast<RectSet>()));
          } else {
            throw std::runtime_error("Unsupported set type");
          }
        }
        return MultiSet(std::move(unique_sets));
      }))
      .def(
          "__iter__",
          [](const MultiSet &self) {
            return py::make_iterator(MultiSetIterator{self}, MultiSetIterator{self, self.sets().size()});
          },
          py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
      .def("__len__", [](const MultiSet &self) { return self.sets().size(); })
      .def(
          "__getitem__", [](const MultiSet &self, const Index index) -> const Set & { return *self.sets()[index]; },
          py::return_value_policy::reference_internal)
      .def("__str__", STRING_LAMBDA(MultiSet));

  /**************************** Project ****************************/
  // TODO(tend): it would be nice to encapsulate this in a class
  m.def(
      "fft_upsample",
      [](ConstMatrixRef f, const Index from_num_samples, const Index to_num_samples, const Index dimension) {
        LUCID_CHECK_ARGUMENT_EXPECTED(dimension > 0, "dimension", dimension, "must be greater than 0");
        LUCID_CHECK_ARGUMENT_EXPECTED(to_num_samples > from_num_samples, "to_num_samples > from_num_samples",
                                      to_num_samples, from_num_samples);

        const int n_pad = static_cast<int>(std::floor((to_num_samples - from_num_samples) / 2.0));
        // Get a view of the input data
        const TensorView<double> in_view{std::span<const double>{f.data(), static_cast<std::size_t>(f.size())},
                                         std::vector<std::size_t>(dimension, from_num_samples)};
        // Permute the last two axes and create a complex tensor
        Tensor<std::complex<double>> fft_in{in_view.dimensions()};
        if (dimension > 1) {  // If the dimension is greater than 1, swap the last two axes
          std::vector<std::size_t> axes{fft_in.axes()};
          std::swap(axes[axes.size() - 2], axes[axes.size() - 1]);
          in_view.permute(fft_in.m_view(), axes);
        } else {
          in_view.copy(fft_in.m_view());
        }
        // Perform FFT upsampling on the data and return the result
        return Vector{static_cast<Eigen::Map<const Vector>>(
            fft_in.fft_upsample(std::vector<std::size_t>(dimension, from_num_samples + 2 * n_pad)))};
      },
      py::arg("f"), py::arg("from_num_samples"), py::arg("to_num_samples"), py::arg("dimension"));

  /**************************** Misc ****************************/
  m.def("read_matrix", &read_matrix<double>, py::arg("filename"));
}