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

#include "lucid/model/model.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "bindings/pylucid/pylucid.h"
#include "lucid/model/Scorer.h"
#include "lucid/util/Tensor.h"
#include "lucid/util/TensorView.h"
#include "lucid/util/error.h"
#include "lucid/verification/verification.h"

namespace py = pybind11;
using namespace lucid;

#ifndef NCONVERT
#define ARG_NONCONVERT(name) py::arg(name)
#else
#define ARG_NONCONVERT(name) py::arg(name).noconvert()
#endif

/**
 * Get the value of the specified parameter from a Parametrizable object.
 * This function uses dispatch to call the appropriate getter based on the type of the parameter.
 * It ensures that the python object returned is of the correct type: an integer, a float, or a numpy array.
 * For the latter, it creates a non-owning numpy array from the Eigen vector
 * and ensures it is read-only by clearing the writeable flag.
 * @param self The Parametrizable object
 * @param parameter The parameter to retrieve
 * @return The value of the parameter as a Python object
 */
py::object get_parametrizable(const Parametrizable &self, const Parameter parameter) {
  return dispatch<py::object>(
      parameter, [&self, parameter]() { return py::int_{self.get<int>(parameter)}; },
      [&self, parameter]() { return py::float_{self.get<double>(parameter)}; },
      [&self, parameter]() {
        const Vector &v = self.get<const Vector &>(parameter);
        py::array array{v.size(), v.data(), py::none{}};
        // https://github.com/pybind/pybind11/issues/481
        reinterpret_cast<py::detail::PyArray_Proxy *>(array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return array;
      });
}
/**
 * Get the value of the ParameterValue object.
 * This function uses dispatch to call the appropriate getter based on the type of the parameter.
 * It ensures that the python object returned is of the correct type: an integer, a float, or a numpy array.
 * For the latter, it creates a non-owning numpy array from the Eigen vector
 * and ensures it is read-only by clearing the writeable flag.
 * @param self The ParameterValue object
 * @return The value of the parameter as a Python object
 */
py::object get_parameter_value(const ParameterValue &self) {
  return dispatch<py::object>(
      self.parameter(), [&self]() { return py::int_{self.get<int>()}; },
      [&self]() { return py::float_{self.get<double>()}; },
      [&self]() {
        const Vector &v = self.get<Vector>();
        py::array array{v.size(), v.data(), py::none{}};
        // https://github.com/pybind/pybind11/issues/481
        reinterpret_cast<py::detail::PyArray_Proxy *>(array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return array;
      });
}
/**
 * Get the value of the ParameterValues object.
 * This function uses dispatch to call the appropriate getter based on the type of the parameter.
 * It ensures that the python object returned is of the correct type: a tuple of integers, floats, or numpy arrays.
 * For the latter, it creates a non-owning numpy array from the Eigen vector
 * and ensures it is read-only by clearing the writeable flag.
 * @param self The ParameterValue object
 * @return The value of the parameter as a Python object
 */
py::object get_parameter_values(const ParameterValues &self) {
  return dispatch<py::object>(
      self.parameter(), [&self]() { return py::tuple{py::cast(self.get<int>())}; },
      [&self]() { return py::tuple{py::cast(self.get<double>())}; },
      [&self]() {
        return py::tuple(py::cast(self.get<Vector>()));
        // TODO(tend): this is not efficient, we should return a tuple of non-owning, non-writable numpy arrays
        // const std::vector<Vector> &vectors = self.get<Vector>();
        // py::list l{vectors.size()};
        // for (const Vector &v : vectors) {
        //   py::array array{v.size(), v.data()};
        //   reinterpret_cast<py::detail::PyArray_Proxy *>(array.ptr())->flags &=
        //       ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        //   l.append(array);
        // }
        // return l;
      });
}

class PyKernel final : public Kernel {
 public:
  using Kernel::Kernel;
  Matrix operator()(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix> *gradient) const override {
    PYBIND11_OVERRIDE_PURE_NAME(Matrix, Kernel, "__call__", operator(), x1, x2, gradient);
  }
  [[nodiscard]] std::unique_ptr<Kernel> clone() const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"Kernel::clone\"");
  }
  [[nodiscard]] bool is_stationary() const override { PYBIND11_OVERRIDE_PURE(Scalar, Kernel, is_stationary); }
  [[nodiscard]] bool is_isotropic() const override { PYBIND11_OVERRIDE_PURE(Scalar, Kernel, is_isotropic); }
};

class PyEstimator final : public Estimator {
 public:
  using Estimator::Estimator;
  [[nodiscard]] Matrix predict(ConstMatrixRef x) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Estimator, predict, x);
  }
  Estimator &consolidate(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs, Requests requests) override {
    PYBIND11_OVERRIDE_PURE(Estimator &, Estimator, consolidate, training_inputs, training_outputs, requests);
  }
  [[nodiscard]] double score(ConstMatrixRef evaluation_inputs, ConstMatrixRef evaluation_outputs) const override {
    PYBIND11_OVERRIDE_PURE(double, Estimator, score, evaluation_inputs, evaluation_outputs);
  }
  [[nodiscard]] std::unique_ptr<Estimator> clone() const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"Estimator::clone\"");
  }
};

class PyTuner final : public Tuner {
 public:
  using Tuner::Tuner;

  void tune_impl(Estimator &estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const override {
    PYBIND11_OVERRIDE_PURE(void, Tuner, tune_impl, estimator, training_inputs, training_outputs);
  }
};

class PySet final : public Set {
 public:
  using Set::Set;
  [[nodiscard]] Dimension dimension() const override { PYBIND11_OVERRIDE_PURE(Dimension, Set, dimension); }
  [[nodiscard]] Matrix sample(Index num_samples) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Set, sample, num_samples);
  }
  [[nodiscard]] bool operator()(ConstMatrixRef x) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, Set, "__call__", operator(), x);
  }
  [[nodiscard]] Matrix lattice(const VectorI &points_per_dim, bool include_endpoints) const override {
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
  operator bool() const { return index_ < multi_set_.sets().size(); }

 private:
  const MultiSet &multi_set_;
  std::size_t index_;
};

void init_model(py::module_ &m) {
  /**************************** Parameters ****************************/
  py::enum_<Parameter>(m, "Parameter")
      .value("DEGREE", Parameter::DEGREE)
      .value("SIGMA_L", Parameter::SIGMA_L)
      .value("SIGMA_F", Parameter::SIGMA_F)
      .value("REGULARIZATION_CONSTANT", Parameter::REGULARIZATION_CONSTANT);
  py::class_<ParameterValue>(m, "ParameterValue")
      .def(py::init<Parameter, int>(), py::arg("parameter"), py::arg("value"))
      .def(py::init<Parameter, double>(), py::arg("parameter"), py::arg("value"))
      .def(py::init<Parameter, const Vector &>(), py::arg("parameter"), py::arg("value"))
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", STRING_LAMBDA(ParameterValue))
      .def_property_readonly("parameter", &ParameterValue::parameter)
      .def_property_readonly("value", get_parameter_value);
  py::class_<ParameterValues>(m, "ParameterValues")
      .def(py::init<Parameter, std::vector<int>>(), py::arg("parameter"), py::arg("values"))
      .def(py::init<Parameter, std::vector<double>>(), py::arg("parameter"), py::arg("values"))
      .def(py::init<Parameter, std::vector<Vector>>(), py::arg("parameter"), py::arg("values"))
      .def("__len__", &ParameterValues::size)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", STRING_LAMBDA(ParameterValues))
      .def_property_readonly("size", &ParameterValues::size)
      .def_property_readonly("parameter", &ParameterValues::parameter)
      .def_property_readonly("values", get_parameter_values);
  py::class_<Parametrizable>(m, "Parametrizable")
      .def("get", get_parametrizable, py::arg("parameter"), py::return_value_policy::reference_internal)
      .def(
          "set",
          [](Parametrizable &self, const Parameter parameter, const int value) { return self.set(parameter, value); },
          py::arg("parameter"), py::arg("value"))
      .def(
          "set",
          [](Parametrizable &self, const Parameter parameter, const double value) {
            return self.set(parameter, value);
          },
          py::arg("parameter"), py::arg("value"))
      .def(
          "set",
          [](Parametrizable &self, const Parameter parameter, const Vector &value) {
            return self.set(parameter, value);
          },
          py::arg("parameter"), py::arg("value"))
      .def("parameters", &Parametrizable::parameters_list)
      .def("has", &Parametrizable::has, py::arg("parameter"))
      .def("__contains__", &Parametrizable::has, py::arg("parameter"));

  /**************************** Set ****************************/
  py::class_<Set, PySet>(m, "Set")
      .def(py::init<>())
      .def_property_readonly("dimension", &Set::dimension)
      .def("sample", py::overload_cast<Index>(&Set::sample, py::const_), py::arg("num_samples"))
      .def("sample", py::overload_cast<>(&Set::sample, py::const_))
      .def("lattice", py::overload_cast<Index, bool>(&Set::lattice, py::const_), py::arg("points_per_dim"),
           py::arg("include_endpoints") = false)
      .def("lattice", py::overload_cast<const VectorI &, bool>(&Set::lattice, py::const_), py::arg("points_per_dim"),
           py::arg("include_endpoints"))
      .def("plot", &Set::plot, py::arg("color"))
      .def("plot3d", &Set::plot3d, py::arg("color"))
      .def("__contains__", &Set::contains, ARG_NONCONVERT("x"))
      .def("__call__", &Set::operator(), ARG_NONCONVERT("x"))
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
        unique_sets.reserve(sets.size());
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

  /**************************** Forward declarations ****************************/
  py::class_<Estimator, PyEstimator, Parametrizable> estimator(m, "Estimator");

  /**************************** Scorer ****************************/
  m.def("r2_score", &scorer::r2_score, py::arg("estimator"), ARG_NONCONVERT("evaluation_inputs"),
        ARG_NONCONVERT("evaluation_outputs"));
  m.def("mse_score", &scorer::mse_score, py::arg("estimator"), ARG_NONCONVERT("evaluation_inputs"),
        ARG_NONCONVERT("evaluation_outputs"));

  /**************************** Tuner ****************************/
  py::class_<Tuner, PyTuner, std::shared_ptr<Tuner>>(m, "Tuner")
      .def(py::init<>())
      .def("tune", &Tuner::tune, py::arg("estimator"), ARG_NONCONVERT("training_inputs"),
           ARG_NONCONVERT("training_outputs"))
      .def("__str__", STRING_LAMBDA(Tuner));
  py::class_<MedianHeuristicTuner, Tuner, std::shared_ptr<MedianHeuristicTuner>>(m, "MedianHeuristicTuner",
                                                                                 py::is_final())
      .def(py::init<>());
  py::class_<GridSearchTuner, Tuner, std::shared_ptr<GridSearchTuner>>(m, "GridSearchTuner", py::is_final())
      .def(py::init<const std::vector<ParameterValues> &, std::size_t>(), py::arg("parameters"), py::arg("n_jobs") = 0)
      .def(py::init([](const py::dict &parameters, const std::size_t n_jobs) {
             std::vector<ParameterValues> parameter_values;
             parameter_values.reserve(parameters.size());
             for (const auto &[key, value] : parameters) {
               const Parameter parameter = key.cast<Parameter>();
               parameter_values.emplace_back(dispatch<ParameterValues>(
                   parameter,
                   [parameter, &value]() { return ParameterValues(parameter, value.cast<std::vector<int>>()); },
                   [parameter, &value]() { return ParameterValues(parameter, value.cast<std::vector<double>>()); },
                   [parameter, &value]() { return ParameterValues(parameter, value.cast<std::vector<Vector>>()); }));
             }
             return GridSearchTuner(std::move(parameter_values), n_jobs);
           }),
           py::arg("parameters"), py::arg("n_jobs") = 0)
      .def(py::init([](const py::args &parameters, const std::size_t n_jobs) {
             std::vector<ParameterValues> parameter_values;
             parameter_values.reserve(parameters.size());
             for (const auto &param : parameters) parameter_values.emplace_back(param.cast<ParameterValues>());
             return GridSearchTuner(std::move(parameter_values), n_jobs);
           }),
           py::kw_only(), py::arg("n_jobs") = 0)
      .def_property_readonly("n_jobs", &GridSearchTuner::n_jobs)
      .def_property_readonly("parameters", &GridSearchTuner::parameters);

  /**************************** Kernel ****************************/
  py::class_<Kernel, PyKernel, Parametrizable>(m, "Kernel")
      .def(
          "__call__", [](const Kernel &self, ConstMatrixRef x1, ConstMatrixRef x2) { return self(x1, x2); },
          ARG_NONCONVERT("x1"), ARG_NONCONVERT("x2"))
      .def(
          "__call__", [](const Kernel &self, ConstMatrixRef x1) { return self(x1, x1); }, ARG_NONCONVERT("x1"))
      .def("clone", &Kernel::clone)
      .def("__str__", STRING_LAMBDA(Kernel));
  py::class_<GaussianKernel, Kernel>(m, "GaussianKernel")
      .def(py::init<const Vector &, double>(), py::arg("sigma_l"), py::arg("sigma_f") = 1.0)
      .def(py::init<Dimension, double, double>(), py::arg("dimension"), py::arg("sigma_l") = 1.0,
           py::arg("sigma_f") = 1.0)
      .def_property_readonly("sigma_f", &GaussianKernel::sigma_f)
      .def_property_readonly("sigma_l", &GaussianKernel::sigma_l);

  /**************************** FeatureMap ****************************/
  py::class_<FeatureMap>(m, "FeatureMap");
  py::class_<TruncatedFourierFeatureMap, FeatureMap>(m, "TruncatedFourierFeatureMap")
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("prob_dim_wise"),
           py::arg("sigma_f"), py::arg("x_limits"))
      .def("map_vector", &TruncatedFourierFeatureMap::map_vector, ARG_NONCONVERT("x"))
      .def("map_matrix", &TruncatedFourierFeatureMap::map_matrix, ARG_NONCONVERT("x"))
      .def("__call__", &TruncatedFourierFeatureMap::operator(), ARG_NONCONVERT("x"))
      .def_property_readonly("dimension", &TruncatedFourierFeatureMap::dimension)
      .def_property_readonly("omega", &TruncatedFourierFeatureMap::omega)
      .def_property_readonly("weights", &TruncatedFourierFeatureMap::weights)
      .def_property_readonly("num_frequencies", &TruncatedFourierFeatureMap::num_frequencies);
  py::class_<ConstantTruncatedFourierFeatureMap, TruncatedFourierFeatureMap>(m, "ConstantTruncatedFourierFeatureMap",
                                                                             py::is_final())
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("x_limits"))
      .def(py::init<long, Scalar, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("x_limits"));
  py::class_<LinearTruncatedFourierFeatureMap, TruncatedFourierFeatureMap>(m, "LinearTruncatedFourierFeatureMap",
                                                                           py::is_final())
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("x_limits"))
      .def(py::init<long, Scalar, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("x_limits"));
  py::class_<LogTruncatedFourierFeatureMap, TruncatedFourierFeatureMap>(m, "LogTruncatedFourierFeatureMap",
                                                                        py::is_final())
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("x_limits"))
      .def(py::init<long, Scalar, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("x_limits"));

  /**************************** Estimator ****************************/
  estimator.def("__call__", &Estimator::operator(), ARG_NONCONVERT("x"))
      .def("predict", &Estimator::predict, ARG_NONCONVERT("x"))
      .def("fit", py::overload_cast<ConstMatrixRef, ConstMatrixRef>(&Estimator::fit), ARG_NONCONVERT("x"),
           ARG_NONCONVERT("y"))
      .def("fit", py::overload_cast<ConstMatrixRef, ConstMatrixRef, const Tuner &>(&Estimator::fit),
           ARG_NONCONVERT("x"), ARG_NONCONVERT("y"), py::arg("tuner"))
      .def("score", &Estimator::score, py::arg("x"), py::arg("y"))
      .def_property("tuner", &Estimator::tuner,
                    [](Estimator &self, const std::shared_ptr<Tuner> &tuner) { self.m_tuner() = tuner; })
      .def("consolidate", &Estimator::consolidate, ARG_NONCONVERT("x"), ARG_NONCONVERT("y"))
      .def("clone", &Estimator::clone)
      .def("__str__", STRING_LAMBDA(Estimator));
  py::class_<KernelRidgeRegressor, Estimator>(m, "KernelRidgeRegressor")
      .def(py::init<const Kernel &, double, const std::shared_ptr<Tuner> &>(), py::arg("kernel"),
           py::arg("regularization_constant") = 1.0, py::arg("tuner") = nullptr)
      .def("__call__",
           py::overload_cast<ConstMatrixRef, const FeatureMap &>(&KernelRidgeRegressor::operator(), py::const_),
           ARG_NONCONVERT("x"), ARG_NONCONVERT("feature_map"))
      .def("__call__", py::overload_cast<ConstMatrixRef>(&KernelRidgeRegressor::operator(), py::const_),
           ARG_NONCONVERT("x"))
      .def("predict", py::overload_cast<ConstMatrixRef, const FeatureMap &>(&KernelRidgeRegressor::predict, py::const_),
           ARG_NONCONVERT("x"), ARG_NONCONVERT("feature_map"))
      .def("predict", py::overload_cast<ConstMatrixRef>(&KernelRidgeRegressor::predict, py::const_),
           ARG_NONCONVERT("x"))
      .def_property_readonly("kernel",
                             [](const KernelRidgeRegressor &self) -> const Kernel & { return *self.kernel(); })
      .def_property_readonly("training_inputs", &KernelRidgeRegressor::training_inputs)
      .def_property_readonly("coefficients", &KernelRidgeRegressor::coefficients)
      .def_property_readonly("regularization_constant", &KernelRidgeRegressor::regularization_constant);

  /**************************** Misc ****************************/
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

  m.def("read_matrix", &read_matrix<double>, py::arg("filename"));
}