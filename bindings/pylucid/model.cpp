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

#include "lucid/model/model.h"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "bindings/pylucid/doxygen_docstrings.h"
#include "bindings/pylucid/pylucid.h"
#include "lucid/model/Scorer.h"
#include "lucid/model/SphereSet.h"
#include "lucid/util/Tensor.h"
#include "lucid/util/TensorView.h"
#include "lucid/util/error.h"

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
  Matrix apply_impl(ConstMatrixRef x1, ConstMatrixRef x2, std::vector<Matrix> *gradient) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Kernel, apply_impl, x1, x2, gradient);
  }
  [[nodiscard]] std::unique_ptr<Kernel> clone() const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"Kernel::clone\"");
  }
  [[nodiscard]] bool is_stationary() const override { PYBIND11_OVERRIDE_PURE(Scalar, Kernel, is_stationary); }
};

class PyEstimator final : public Estimator {
 public:
  using Estimator::Estimator;
  [[nodiscard]] Matrix predict(ConstMatrixRef x) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Estimator, predict, x);
  }
  Estimator &consolidate_impl(ConstMatrixRef training_inputs, ConstMatrixRef training_outputs,
                              Requests requests) override {
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

  void tune_impl(Estimator &estimator, ConstMatrixRef training_inputs,
                 const OutputComputer &training_outputs) const override {
    PYBIND11_OVERRIDE_PURE(void, Tuner, tune_impl, estimator, training_inputs, training_outputs);
  }
};

class PyCrossValidator final : public CrossValidator {
 public:
  using CrossValidator::CrossValidator;

  [[nodiscard]] Dimension num_folds(ConstMatrixRef training_inputs) const override {
    PYBIND11_OVERRIDE_PURE(Dimension, CrossValidator, num_folds, training_inputs);
  }

 private:
  [[nodiscard]] std::pair<SliceSelector, SliceSelector> compute_folds(ConstMatrixRef training_inputs) const override {
    using SliceSelectorPair = std::pair<SliceSelector, SliceSelector>;
    PYBIND11_OVERRIDE_PURE(SliceSelectorPair, CrossValidator, compute_folds, training_inputs);
  }
};

class PyFeatureMap : public FeatureMap {
 public:
  using FeatureMap::FeatureMap;
  [[nodiscard]] Matrix apply_impl(ConstMatrixRef x) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, FeatureMap, apply_impl, x);
  }
  [[nodiscard]] Matrix invert_impl(ConstMatrixRef y) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, FeatureMap, invert_impl, y);
  }
  [[nodiscard]] std::unique_ptr<FeatureMap> clone() const override {
    pybind11::pybind11_fail("Tried to call pure virtual function \"FeatureMap::clone\"");
  }
};

class PySet final : public Set {
 public:
  using Set::Set;
  [[nodiscard]] Dimension dimension() const override { PYBIND11_OVERRIDE_PURE(Dimension, Set, dimension); }
  void change_size(ConstVectorRef delta_size) override { PYBIND11_OVERRIDE(void, Set, change_size, delta_size); }
  [[nodiscard]] Matrix sample(Index num_samples) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Set, sample, num_samples);
  }
  [[nodiscard]] bool operator()(ConstVectorRef x) const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, Set, "__call__", operator(), x);
  }
  [[nodiscard]] Matrix lattice(const VectorI &points_per_dim, bool endpoint) const override {
    PYBIND11_OVERRIDE_PURE(Matrix, Set, lattice, points_per_dim, endpoint);
  }
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
  py::class_<LbfgsParameters>(m, "LbfgsParameters", LbfgsParameters_)
      .def(py::init<>())
      .def(py::init([](const int m_, const Scalar epsilon_, const Scalar epsilon_rel_, const int past_,
                       const Scalar delta_, const int max_iterations_, const int linesearch_, const int max_submin_,
                       const int max_linesearch_, const Scalar min_step_, const Scalar max_step_, const Scalar ftol_,
                       const Scalar wolfe_) {
             return LbfgsParameters{.m = m_,
                                    .epsilon = epsilon_,
                                    .epsilon_rel = epsilon_rel_,
                                    .past = past_,
                                    .delta = delta_,
                                    .max_iterations = max_iterations_,
                                    .linesearch = linesearch_,
                                    .max_submin = max_submin_,
                                    .max_linesearch = max_linesearch_,
                                    .min_step = min_step_,
                                    .max_step = max_step_,
                                    .ftol = ftol_,
                                    .wolfe = wolfe_};
           }),
           py::arg("m") = 6, py::arg("epsilon") = 1e-5, py::arg("epsilon_rel") = 1e-5, py::arg("past") = 0,
           py::arg("delta") = 0, py::arg("max_iterations") = 0, py::arg("linesearch") = 3, py::arg("max_submin") = 10,
           py::arg("max_linesearch") = 20, py::arg("min_step") = 1e-20, py::arg("max_step") = 1e20,
           py::arg("ftol") = 1e-4, py::arg("wolfe") = 0.9, py::doc(""))
      .def_readwrite("m", &LbfgsParameters::m, LbfgsParameters_m)
      .def_readwrite("epsilon", &LbfgsParameters::epsilon, LbfgsParameters_epsilon)
      .def_readwrite("epsilon_rel", &LbfgsParameters::epsilon_rel, LbfgsParameters_epsilon_rel)
      .def_readwrite("past", &LbfgsParameters::past, LbfgsParameters_past)
      .def_readwrite("delta", &LbfgsParameters::delta, LbfgsParameters_delta)
      .def_readwrite("max_iterations", &LbfgsParameters::max_iterations, LbfgsParameters_max_iterations)
      .def_readwrite("max_submin", &LbfgsParameters::max_submin, LbfgsParameters_max_submin)
      .def_readwrite("linesearch", &LbfgsParameters::linesearch, LbfgsParameters_linesearch)
      .def_readwrite("max_linesearch", &LbfgsParameters::max_linesearch, LbfgsParameters_max_linesearch)
      .def_readwrite("min_step", &LbfgsParameters::min_step, LbfgsParameters_min_step)
      .def_readwrite("max_step", &LbfgsParameters::max_step, LbfgsParameters_max_step)
      .def_readwrite("ftol", &LbfgsParameters::ftol, LbfgsParameters_ftol)
      .def_readwrite("wolfe", &LbfgsParameters::wolfe, LbfgsParameters_wolfe)
      .def("__str__", STRING_LAMBDA(LbfgsParameters));

  /**************************** Requests ****************************/
  py::enum_<Request>(m, "Request", _Request)
      .value("OBJECTIVE_VALUE", Request::OBJECTIVE_VALUE, Request_OBJECTIVE_VALUE)
      .value("GRADIENT", Request::GRADIENT, Request_GRADIENT);

  /**************************** Parameters ****************************/
  py::enum_<Parameter>(m, "Parameter", _Parameter)
      .value("A", Parameter::A, Parameter_A)
      .value("B", Parameter::B, Parameter_B)
      .value("DEGREE", Parameter::DEGREE, Parameter_DEGREE)
      .value("SIGMA_L", Parameter::SIGMA_L, Parameter_SIGMA_L)
      .value("SIGMA_F", Parameter::SIGMA_F, Parameter_SIGMA_F)
      .value("REGULARIZATION_CONSTANT", Parameter::REGULARIZATION_CONSTANT, Parameter_REGULARIZATION_CONSTANT)
      .value("GRADIENT_OPTIMIZABLE", Parameter::GRADIENT_OPTIMIZABLE, Parameter_GRADIENT_OPTIMIZABLE);
  py::class_<ParameterValue>(m, "ParameterValue", ParameterValue_)
      .def(py::init<Parameter, int>(), py::arg("parameter"), py::arg("value"), ParameterValue_ParameterValue)
      .def(py::init<Parameter, double>(), py::arg("parameter"), py::arg("value"), ParameterValue_ParameterValue)
      .def(py::init<Parameter, const Vector &>(), py::arg("parameter"), py::arg("value"), ParameterValue_ParameterValue)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", STRING_LAMBDA(ParameterValue))
      .def_property_readonly("parameter", &ParameterValue::parameter, ParameterValue_parameter)
      .def_property_readonly("value", get_parameter_value, ParameterValue_value);
  py::class_<ParameterValues>(m, "ParameterValues", ParameterValues_)
      .def(py::init<Parameter, std::vector<int>>(), py::arg("parameter"), py::arg("values"),
           ParameterValues_ParameterValues)
      .def(py::init<Parameter, std::vector<double>>(), py::arg("parameter"), py::arg("values"),
           ParameterValues_ParameterValues)
      .def(py::init<Parameter, std::vector<Vector>>(), py::arg("parameter"), py::arg("values"),
           ParameterValues_ParameterValues)
      .def("__len__", &ParameterValues::size)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", STRING_LAMBDA(ParameterValues))
      .def_property_readonly("size", &ParameterValues::size, ParameterValues_size)
      .def_property_readonly("parameter", &ParameterValues::parameter, ParameterValues_parameter)
      .def_property_readonly("values", get_parameter_values, ParameterValues_values);
  py::class_<Parametrizable>(m, "Parametrizable", Parametrizable_)
      .def("get", get_parametrizable, py::arg("parameter"), py::return_value_policy::reference_internal,
           Parametrizable_get)
      .def(
          "set",
          [](Parametrizable &self, const Parameter parameter, const int value) { return self.set(parameter, value); },
          py::arg("parameter"), py::arg("value"), Parametrizable_set)
      .def(
          "set",
          [](Parametrizable &self, const Parameter parameter, const double value) {
            return self.set(parameter, value);
          },
          py::arg("parameter"), py::arg("value"), Parametrizable_set)
      .def(
          "set",
          [](Parametrizable &self, const Parameter parameter, const Vector &value) {
            return self.set(parameter, value);
          },
          py::arg("parameter"), py::arg("value"), Parametrizable_set)
      .def("has", &Parametrizable::has, py::arg("parameter"), Parametrizable_has)
      .def_property_readonly("parameters", &Parametrizable::parameters_list, Parametrizable_parameters_list)
      .def("__contains__", &Parametrizable::has, py::arg("parameter"));

  /**************************** Set ****************************/
  py::class_<Set, PySet>(m, "Set", Set_)
      .def(py::init<>())
      .def_property_readonly("dimension", &Set::dimension, Set_dimension)
      .def("sample", py::overload_cast<Index>(&Set::sample, py::const_), py::arg("num_samples"), Set_sample)
      .def("sample", py::overload_cast<>(&Set::sample, py::const_), Set_sample)
      .def("include", &Set::include, ARG_NONCONVERT("xs"), Set_include)
      .def("include_mask", &Set::include_mask, ARG_NONCONVERT("xs"), Set_include_mask)
      .def("exclude", &Set::exclude, ARG_NONCONVERT("xs"), Set_exclude)
      .def("exclude_mask", &Set::exclude_mask, ARG_NONCONVERT("xs"), Set_exclude_mask)
      .def("include_exclude_masks", &Set::include_exclude_masks, ARG_NONCONVERT("xs"), Set_include_exclude_masks)
      .def("change_size", py::overload_cast<double>(&Set::change_size), py::arg("delta_size"), Set_change_size)
      .def("change_size", py::overload_cast<ConstVectorRef>(&Set::change_size), ARG_NONCONVERT("delta_size"),
           Set_change_size)
      .def("lattice", py::overload_cast<Index, bool>(&Set::lattice, py::const_), py::arg("points_per_dim"),
           py::arg("endpoint") = false, Set_lattice)
      .def("lattice", py::overload_cast<const VectorI &, bool>(&Set::lattice, py::const_), py::arg("points_per_dim"),
           py::arg("endpoint"), Set_lattice)
      .def("contains", &Set::contains, ARG_NONCONVERT("x"), Set_contains)
      .def("to_rect_set", &Set::to_rect_set, Set_to_rect_set)
      .def("__contains__", &Set::contains, ARG_NONCONVERT("x"), Set_contains)
      .def("__call__", &Set::operator(), ARG_NONCONVERT("x"), Set_operator_apply)
      .def("__str__", STRING_LAMBDA(Set));
  py::class_<RectSet, Set>(m, "RectSet", RectSet_)
      .def(py::init<Vector, Vector>(), py::arg("lb"), py::arg("ub"), RectSet_RectSet)
      .def(py::init<std::vector<std::pair<Scalar, Scalar>>>(), py::arg("bounds"), RectSet_RectSet)
      .def("relative_to", py::overload_cast<const RectSet &>(&RectSet::relative_to, py::const_), py::arg("set"),
           RectSet_relative_to)
      .def("relative_to", py::overload_cast<ConstVectorRef>(&RectSet::relative_to, py::const_), ARG_NONCONVERT("point"),
           RectSet_relative_to)
      .def("scale", py::overload_cast<ConstVectorRef>(&RectSet::scale, py::const_), ARG_NONCONVERT("scale"),
           RectSet_scale)
      .def("scale", py::overload_cast<double>(&RectSet::scale, py::const_), py::arg("scale"), RectSet_scale)
      .def("scale", py::overload_cast<ConstVectorRef, const RectSet &, bool>(&RectSet::scale, py::const_),
           ARG_NONCONVERT("scale"), py::arg("bounds"), py::arg("relative_to_bounds") = false, RectSet_scale)
      .def("scale", py::overload_cast<double, const RectSet &, bool>(&RectSet::scale, py::const_), py::arg("scale"),
           py::arg("bounds"), py::arg("relative_to_bounds") = false, RectSet_scale)
      .def(py::self += double())
      .def(py::self += ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("offset"))
      .def(py::self + ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("offset"))
      .def(py::self + double())
      .def(py::self -= double())
      .def(py::self -= ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("offset"))
      .def(py::self - ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("offset"))
      .def(py::self - double())
      .def(py::self *= double())
      .def(py::self *= ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("scale"))
      .def(py::self * ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("scale"))
      .def(py::self * double())
      .def(py::self /= double())
      .def(py::self /= ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("scale"))
      .def(py::self / ConstMatrixRefCopy(Matrix()), ARG_NONCONVERT("scale"))
      .def(py::self / double())
      .def_property_readonly("sizes", &RectSet::sizes, RectSet_sizes)
      .def_property_readonly("lower_bound", &RectSet::lower_bound, RectSet_lower_bound)
      .def_property_readonly("upper_bound", &RectSet::upper_bound, RectSet_upper_bound);
  py::class_<SphereSet, Set>(m, "SphereSet", SphereSet_)
      .def(py::init<Vector, double>(), py::arg("center"), py::arg("radius"), SphereSet_SphereSet)
      .def_property_readonly("center", &SphereSet::center, SphereSet_center)
      .def_property_readonly("radius", &SphereSet::radius, SphereSet_radius);
  py::class_<PolytopeSet, Set>(m, "PolytopeSet", PolytopeSet_)
      .def(py::init<Matrix, Vector>(), py::arg("A"), py::arg("b"), PolytopeSet_PolytopeSet)
      .def("scale", &PolytopeSet::scale, py::arg("factor"), PolytopeSet_scale)
      .def_property_readonly("A", &PolytopeSet::A, PolytopeSet_A)
      .def_property_readonly("b", &PolytopeSet::b, PolytopeSet_b)
      .def_property_readonly("bounding_box", &PolytopeSet::bounding_box);
  py::class_<MultiSet, Set>(m, "MultiSet", MultiSet_)
      .def(py::init([](const py::args &sets) {
        std::vector<std::unique_ptr<Set>> unique_sets;
        unique_sets.reserve(sets.size());
        for (const auto &set : sets) {
          if (py::isinstance<RectSet>(set)) {
            unique_sets.emplace_back(std::make_unique<RectSet>(set.cast<RectSet>()));
          } else if (py::isinstance<SphereSet>(set)) {
            unique_sets.emplace_back(std::make_unique<SphereSet>(set.cast<SphereSet>()));
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
          py::return_value_policy::reference_internal);

  /**************************** Forward declarations ****************************/
  py::class_<Estimator, PyEstimator, Parametrizable> estimator(m, "Estimator");

  /**************************** Scorer ****************************/
  m.def("r2_score", py::overload_cast<const Estimator &, ConstMatrixRef, ConstMatrixRef>(&scorer::r2_score),
        py::arg("estimator"), ARG_NONCONVERT("evaluation_inputs"), ARG_NONCONVERT("evaluation_outputs"), _r2_score);
  m.def("r2_score", py::overload_cast<ConstMatrixRef, ConstMatrixRef>(&scorer::r2_score), ARG_NONCONVERT("x"),
        ARG_NONCONVERT("y"), _r2_score);
  m.def("mse_score", py::overload_cast<const Estimator &, ConstMatrixRef, ConstMatrixRef>(&scorer::mse_score),
        py::arg("estimator"), ARG_NONCONVERT("evaluation_inputs"), ARG_NONCONVERT("evaluation_outputs"), _mse_score);
  m.def("mse_score", py::overload_cast<ConstMatrixRef, ConstMatrixRef>(&scorer::mse_score), ARG_NONCONVERT("x"),
        ARG_NONCONVERT("y"), _mse_score);
  m.def("rmse_score", py::overload_cast<const Estimator &, ConstMatrixRef, ConstMatrixRef>(&scorer::rmse_score),
        py::arg("estimator"), ARG_NONCONVERT("evaluation_inputs"), ARG_NONCONVERT("evaluation_outputs"), _rmse_score);
  m.def("rmse_score", py::overload_cast<ConstMatrixRef, ConstMatrixRef>(&scorer::rmse_score), ARG_NONCONVERT("x"),
        ARG_NONCONVERT("y"), _rmse_score);
  m.def("mape_score", py::overload_cast<const Estimator &, ConstMatrixRef, ConstMatrixRef>(&scorer::mape_score),
        py::arg("estimator"), ARG_NONCONVERT("evaluation_inputs"), ARG_NONCONVERT("evaluation_outputs"), _mape_score);
  m.def("mape_score", py::overload_cast<ConstMatrixRef, ConstMatrixRef>(&scorer::mape_score), ARG_NONCONVERT("x"),
        ARG_NONCONVERT("y"), _mape_score);

  /**************************** Tuner ****************************/
  py::class_<Tuner, PyTuner, std::shared_ptr<Tuner>>(m, "Tuner", Tuner_)
      .def(py::init<>())
      .def("tune", &Tuner::tune, py::arg("estimator"), ARG_NONCONVERT("training_inputs"),
           ARG_NONCONVERT("training_outputs"), Tuner_tune)
      .def("__str__", STRING_LAMBDA(Tuner));
  py::class_<MedianHeuristicTuner, Tuner, std::shared_ptr<MedianHeuristicTuner>>(m, "MedianHeuristicTuner",
                                                                                 MedianHeuristicTuner_, py::is_final())
      .def(py::init<>());
  py::class_<LbfgsTuner, Tuner, std::shared_ptr<LbfgsTuner>>(m, "LbfgsTuner", LbfgsTuner_, py::is_final())
      .def(py::init<const LbfgsParameters &>(), py::arg("parameters") = LbfgsParameters{}, LbfgsTuner_LbfgsTuner)
      .def(py::init<const Eigen::VectorXd &, const Eigen::VectorXd &, const LbfgsParameters &>(), py::arg("lb"),
           py::arg("ub"), py::arg("parameters") = LbfgsParameters{}, LbfgsTuner_LbfgsTuner)
      .def(py::init<std::vector<std::pair<Scalar, Scalar>>, const LbfgsParameters &>(), py::arg("bounds"),
           py::arg("parameters") = LbfgsParameters{}, LbfgsTuner_LbfgsTuner);
  py::class_<GridSearchTuner, Tuner, std::shared_ptr<GridSearchTuner>>(m, "GridSearchTuner", GridSearchTuner_,
                                                                       py::is_final())
      .def(py::init<const std::vector<ParameterValues> &, std::size_t>(), py::arg("parameters"), py::arg("n_jobs") = 0,
           GridSearchTuner_GridSearchTuner)
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
           py::arg("parameters"), py::arg("n_jobs") = 0, GridSearchTuner_GridSearchTuner)
      .def(py::init([](const py::args &parameters, const std::size_t n_jobs) {
             std::vector<ParameterValues> parameter_values;
             parameter_values.reserve(parameters.size());
             for (const auto &param : parameters) parameter_values.emplace_back(param.cast<ParameterValues>());
             return GridSearchTuner(std::move(parameter_values), n_jobs);
           }),
           py::kw_only(), py::arg("n_jobs") = 0, GridSearchTuner_GridSearchTuner)
      .def("tune", py::overload_cast<Estimator &, ConstMatrixRef, ConstMatrixRef>(&Tuner::tune, py::const_),
           py::arg("estimator"), ARG_NONCONVERT("training_inputs"), ARG_NONCONVERT("training_outputs"))
      .def(
          "tune",
          [](const GridSearchTuner &self, Estimator &estimator_, ConstMatrixRef training_inputs,
             ConstMatrixRef training_outputs, const py::type &feature_map_type, const int num_frequencies,
             const RectSet &X_bounds) {
            if (feature_map_type.is(py::type::of<ConstantTruncatedFourierFeatureMap>())) {
              return self.tune<ConstantTruncatedFourierFeatureMap>(estimator_, training_inputs, training_outputs,
                                                                   num_frequencies, X_bounds);
            }
            if (feature_map_type.is(py::type::of<LinearTruncatedFourierFeatureMap>())) {
              return self.tune<LinearTruncatedFourierFeatureMap>(estimator_, training_inputs, training_outputs,
                                                                 num_frequencies, X_bounds);
            }
            if (feature_map_type.is(py::type::of<LogTruncatedFourierFeatureMap>())) {
              return self.tune<LogTruncatedFourierFeatureMap>(estimator_, training_inputs, training_outputs,
                                                              num_frequencies, X_bounds);
            }
            throw std::runtime_error("Unsupported feature map type");
          },
          py::arg("estimator"), ARG_NONCONVERT("training_inputs"), ARG_NONCONVERT("training_outputs"),
          py::arg("feature_map_type"), py::arg("num_frequencies"), py::arg("X_bounds"))
      .def_property_readonly("n_jobs", &GridSearchTuner::n_jobs, GridSearchTuner_n_jobs)
      .def_property_readonly("parameters", &GridSearchTuner::parameters, GridSearchTuner_parameters);

  /**************************** Kernel ****************************/
  py::class_<Kernel, PyKernel, Parametrizable>(m, "Kernel", Kernel_)
      .def(
          "__call__", [](const Kernel &self, ConstMatrixRef x1, ConstMatrixRef x2) { return self(x1, x2); },
          ARG_NONCONVERT("x1"), ARG_NONCONVERT("x2"), Kernel_operator_apply)
      .def(
          "__call__", [](const Kernel &self, ConstMatrixRef x1) { return self(x1, x1); }, ARG_NONCONVERT("x1"),
          Kernel_operator_apply)
      .def("clone", &Kernel::clone, Kernel_clone)
      .def_property_readonly("is_stationary", &Kernel::is_stationary, Kernel_is_stationary)
      .def("__str__", STRING_LAMBDA(Kernel));
  py::class_<GaussianKernel, Kernel>(m, "GaussianKernel", GaussianKernel_)
      .def(py::init<const Vector &, double>(), py::arg("sigma_l"), py::arg("sigma_f") = 1.0,
           GaussianKernel_GaussianKernel)
      .def(py::init<double, double>(), py::arg("sigma_l") = 1.0, py::arg("sigma_f") = 1.0,
           GaussianKernel_GaussianKernel)
      .def_property_readonly("is_isotropic", &GaussianKernel::is_isotropic, GaussianKernel_is_isotropic)
      .def_property_readonly("sigma_f", &GaussianKernel::sigma_f, GaussianKernel_sigma_f)
      .def_property_readonly("sigma_l", &GaussianKernel::sigma_l, GaussianKernel_sigma_l);
  py::class_<ValleePoussinKernel, Kernel>(m, "ValleePoussinKernel", ValleePoussinKernel_)
      .def(py::init<double, double>(), py::arg("a") = 1.0, py::arg("b") = 1.0, ValleePoussinKernel_ValleePoussinKernel)
      .def_property_readonly("a", &ValleePoussinKernel::a, ValleePoussinKernel_a)
      .def_property_readonly("b", &ValleePoussinKernel::b, ValleePoussinKernel_b);

  /**************************** FeatureMap ****************************/
  py::class_<FeatureMap, PyFeatureMap>(m, "FeatureMap", FeatureMap_)
      .def("clone", &FeatureMap::clone, FeatureMap_clone)
      .def("invert", &FeatureMap::invert, ARG_NONCONVERT("y"), FeatureMap_invert)
      .def("__call__", &FeatureMap::operator(), ARG_NONCONVERT("x"), FeatureMap_operator_apply)
      .def("__str__", STRING_LAMBDA(FeatureMap));
  py::class_<TruncatedFourierFeatureMap, FeatureMap>(m, "TruncatedFourierFeatureMap", TruncatedFourierFeatureMap_)
      .def(py::init<long, ConstVectorRef, ConstVectorRef, ConstVectorRef, Scalar, RectSet>(),
           py::arg("num_frequencies"), py::arg("prob_per_dim"), py::arg("omega_per_dim"), ARG_NONCONVERT("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"))
      .def(py::init<long, ConstVectorRef, ConstVectorRef, Scalar, RectSet, bool>(), py::arg("num_frequencies"),
           py::arg("prob_per_dim"), py::arg("omega_per_dim"), py::arg("sigma_f"), py::arg("X_bounds"),
           py::arg("unused"))
      .def("get_periodic_set", &TruncatedFourierFeatureMap::get_periodic_set,
           TruncatedFourierFeatureMap_get_periodic_set)
      .def("map_vector", &TruncatedFourierFeatureMap::map_vector, ARG_NONCONVERT("x"),
           TruncatedFourierFeatureMap_map_vector)
      .def("map_matrix", &TruncatedFourierFeatureMap::map_matrix, ARG_NONCONVERT("x"),
           TruncatedFourierFeatureMap_map_matrix)
      .def_property_readonly("sigma_f", &TruncatedFourierFeatureMap::sigma_f, TruncatedFourierFeatureMap_sigma_f)
      .def_property_readonly("periodic_coefficients", &TruncatedFourierFeatureMap::periodic_coefficients,
                             TruncatedFourierFeatureMap_periodic_coefficients)
      .def_property_readonly("X_bounds", &TruncatedFourierFeatureMap::X_bounds, TruncatedFourierFeatureMap_X_bounds)
      .def_property_readonly("dimension", &TruncatedFourierFeatureMap::dimension, TruncatedFourierFeatureMap_dimension)
      .def_property_readonly("omega", &TruncatedFourierFeatureMap::omega, TruncatedFourierFeatureMap_omega)
      .def_property_readonly("weights", &TruncatedFourierFeatureMap::weights, TruncatedFourierFeatureMap_weights)
      .def_property_readonly("num_frequencies", &TruncatedFourierFeatureMap::num_frequencies,
                             TruncatedFourierFeatureMap_num_frequencies);
  py::class_<ConstantTruncatedFourierFeatureMap, TruncatedFourierFeatureMap>(
      m, "ConstantTruncatedFourierFeatureMap", ConstantTruncatedFourierFeatureMap_, py::is_final())
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"))
      .def(py::init<long, Scalar, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"));
  py::class_<LinearTruncatedFourierFeatureMap, TruncatedFourierFeatureMap>(
      m, "LinearTruncatedFourierFeatureMap", py::is_final(), LinearTruncatedFourierFeatureMap_)
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"))
      .def(py::init<long, Scalar, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"))
      .def(py::init<long, Scalar, Scalar, RectSet, bool>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"), py::arg("unused"))
      .def(py::init<long, Scalar, Scalar, RectSet, bool>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"), py::arg("unused"));
  py::class_<LogTruncatedFourierFeatureMap, TruncatedFourierFeatureMap>(m, "LogTruncatedFourierFeatureMap",
                                                                        py::is_final(), LogTruncatedFourierFeatureMap_)
      .def(py::init<long, ConstVectorRef, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"))
      .def(py::init<long, Scalar, Scalar, RectSet>(), py::arg("num_frequencies"), py::arg("sigma_l"),
           py::arg("sigma_f"), py::arg("X_bounds"));

  /**************************** Estimator ****************************/
  estimator.def(py::init<>(), Estimator_Estimator)
      .def("predict", &Estimator::predict, ARG_NONCONVERT("x"), Estimator_predict)
      .def("fit", py::overload_cast<ConstMatrixRef, ConstMatrixRef>(&Estimator::fit), ARG_NONCONVERT("x"),
           ARG_NONCONVERT("y"), Estimator_fit)
      .def("fit", py::overload_cast<ConstMatrixRef, ConstMatrixRef, const Tuner &>(&Estimator::fit),
           ARG_NONCONVERT("x"), ARG_NONCONVERT("y"), py::arg("tuner"), Estimator_fit)
      .def("fit", py::overload_cast<ConstMatrixRef, const OutputComputer &>(&Estimator::fit_online),
           ARG_NONCONVERT("x"), py::arg("y"), Estimator_fit_online)
      .def("fit", py::overload_cast<ConstMatrixRef, const OutputComputer &, const Tuner &>(&Estimator::fit_online),
           ARG_NONCONVERT("x"), py::arg("y"), py::arg("tuner"), Estimator_fit_online)
      .def("score", &Estimator::score, py::arg("x"), py::arg("y"), Estimator_score)
      .def_property(
          "tuner", &Estimator::tuner,
          [](Estimator &self, const std::shared_ptr<Tuner> &tuner) { self.m_tuner() = tuner; }, Estimator_tuner)
      .def("consolidate", py::overload_cast<ConstMatrixRef, ConstMatrixRef, Requests>(&Estimator::consolidate),
           ARG_NONCONVERT("x"), ARG_NONCONVERT("y"), py::arg("requests") = NoRequests, Estimator_consolidate)
      .def("clone", &Estimator::clone)
      .def("__call__", py::overload_cast<ConstMatrixRef>(&Estimator::operator(), py::const_), ARG_NONCONVERT("x"),
           Estimator_operator_apply)
      .def("__str__", STRING_LAMBDA(Estimator));
  py::class_<ModelEstimator, Estimator>(m, "ModelEstimator", ModelEstimator_)
      .def(py::init<const std::function<Matrix(ConstMatrixRef)> &>(), py::arg("model_function"),
           ModelEstimator_ModelEstimator);
  py::class_<KernelRidgeRegressor, Estimator>(m, "KernelRidgeRegressor", KernelRidgeRegressor_)
      .def(py::init([](double regularization_constant, const std::shared_ptr<Tuner> &tuner) {
             return KernelRidgeRegressor{std::make_unique<GaussianKernel>(), regularization_constant, tuner};
           }),
           py::arg("regularization_constant") = 1.0, py::arg("tuner") = nullptr)
      .def(py::init<const Kernel &, double, const std::shared_ptr<Tuner> &>(), py::arg("kernel"),
           py::arg("regularization_constant") = 1.0, py::arg("tuner") = nullptr)
      .def(
          "predict",
          [](const KernelRidgeRegressor &self, ConstMatrixRef x, const FeatureMap *const feature_map) {
            return feature_map == nullptr ? self.predict(x) : self.predict(x, *feature_map);
          },
          ARG_NONCONVERT("x"), py::arg("feature_map").none(true) = nullptr)
      .def(
          "__call__",
          [](const KernelRidgeRegressor &self, ConstMatrixRef x, const FeatureMap *const feature_map) {
            return feature_map == nullptr ? self.predict(x) : self.predict(x, *feature_map);
          },
          ARG_NONCONVERT("x"), py::arg("feature_map").none(true) = nullptr)
      .def_property_readonly(
          "kernel", [](const KernelRidgeRegressor &self) -> const Kernel & { return *self.kernel(); },
          KernelRidgeRegressor_kernel)
      .def_property_readonly("training_inputs", &KernelRidgeRegressor::training_inputs,
                             KernelRidgeRegressor_training_inputs)
      .def_property_readonly("coefficients", &KernelRidgeRegressor::coefficients, KernelRidgeRegressor_coefficients)
      .def_property_readonly("regularization_constant", &KernelRidgeRegressor::regularization_constant,
                             KernelRidgeRegressor_regularization_constant);

  /**************************** CrossValidator ****************************/
  py::class_<CrossValidator, PyCrossValidator>(m, "CrossValidator", CrossValidator_)
      .def(py::init<>())
      .def("num_folds", &CrossValidator::num_folds, ARG_NONCONVERT("training_inputs"), CrossValidator_num_folds)
      .def("fit",
           py::overload_cast<Estimator &, ConstMatrixRef, ConstMatrixRef, const scorer::Scorer &>(&CrossValidator::fit,
                                                                                                  py::const_),
           py::arg("estimator"), ARG_NONCONVERT("training_inputs"), ARG_NONCONVERT("training_outputs"),
           py::arg("scorer"), CrossValidator_compute_folds)
      .def("fit",
           py::overload_cast<Estimator &, ConstMatrixRef, ConstMatrixRef, const Tuner &, const scorer::Scorer &>(
               &CrossValidator::fit, py::const_),
           py::arg("estimator"), ARG_NONCONVERT("training_inputs"), ARG_NONCONVERT("training_outputs"),
           py::arg("tuner"), py::arg("scorer") = nullptr, CrossValidator_compute_folds)
      .def("score",
           py::overload_cast<const Estimator &, ConstMatrixRef, ConstMatrixRef, const scorer::Scorer &>(
               &CrossValidator::score, py::const_),
           py::arg("estimator"), ARG_NONCONVERT("training_inputs"), ARG_NONCONVERT("training_outputs"),
           py::arg("scorer"), CrossValidator_compute_folds);
  py::class_<LeaveOneOut, CrossValidator>(m, "LeaveOneOut", LeaveOneOut_).def(py::init<>());
  py::class_<KFold, CrossValidator>(m, "KFold", KFold_)
      .def(py::init<int, bool>(), py::arg("num_folds") = 5, py::arg("shuffle") = true, KFold_KFold);

  /**************************** Misc ****************************/
  // TODO(tend): it would be nice to encapsulate this in a class
  m.def(
      "fft_upsample",
      [](ConstMatrixRef f, const Index from_num_samples, const Index to_num_samples, const Index dimension) {
        LUCID_CHECK_ARGUMENT_CMP(dimension, >, 0);
        LUCID_CHECK_ARGUMENT_CMP(to_num_samples, >, from_num_samples);

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