/**
 * @author Room 6.030
 * @author Benno Evers
 * @copyright 2014 Benno Evers
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#ifndef LUCID_MATPLOTLIB_BUILD
#error "This file should not be included without LUCID_LUCID_MATPLOTLIB_BUILD"
#endif
#include "lucid/util/matplotlibcpp.h"

#include "lucid/util/error.h"

namespace matplotlibcpp ::detail {

PyObject* _interpreter::py_get_function(PyObject* const module, const std::string& name) {
  PyObject* fn = PyObject_GetAttrString(module, name.c_str());
  if (!fn) LUCID_PY_ERROR_FMT("Couldn't find required function: {}", name);
  if (!PyFunction_Check(fn)) LUCID_PY_ERROR_FMT("{} is not a Python function", name);
  return fn;
}
PyObject* _interpreter::py_import(const std::string& name) {
  if (imports_.contains(name)) return imports_[name];  // Module previously loaded

  PyObject* const _import_name = PyString_FromString(name.c_str());
  if (!_import_name) LUCID_PY_ERROR_FMT("Couldn't create string for module name: {}", name);
  PyObject* const _import = PyImport_Import(_import_name);
  if (!_import) LUCID_PY_ERROR_FMT("Couldn't import {}", name);
  Py_DECREF(_import_name);
  imports_.emplace(name, _import);
  return _import;
}

}  // namespace matplotlibcpp::detail
void matplotlibcpp::ylim(double bottom, double top) {
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(bottom, top);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylim, args);
  if (!res) throw std::runtime_error("Call to ylim() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}
std::array<double, 2> matplotlibcpp::xlim() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallNoArgs(detail::_interpreter::get().s_python_function_xlim);
  if (!res) throw std::runtime_error("Call to xlim() failed.");

  const double left = PyFloat_AsDouble(PyTuple_GetItem(res, 0));
  const double right = PyFloat_AsDouble(PyTuple_GetItem(res, 1));

  Py_DECREF(res);
  return {left, right};
}
std::array<double, 2> matplotlibcpp::ylim() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallNoArgs(detail::_interpreter::get().s_python_function_ylim);
  if (!res) throw std::runtime_error("Call to ylim() failed.");

  const double left = PyFloat_AsDouble(PyTuple_GetItem(res, 0));
  const double right = PyFloat_AsDouble(PyTuple_GetItem(res, 1));

  Py_DECREF(res);
  return {left, right};
}
void matplotlibcpp::xlim(double left, double right) {
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(left, right);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlim, args);
  if (!res) throw std::runtime_error("Call to xlim() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}