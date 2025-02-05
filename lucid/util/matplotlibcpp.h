/**
 * @author Room 6.030
 * @author Benno Evers
 * @copyright 2014 Benno Evers
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * A C++ wrapper for python's matplotlib.
 */
#pragma once
#ifndef LUCID_MATPLOTLIB_BUILD
#error "This file should not be included without LUCID_LUCID_MATPLOTLIB_BUILD"
#endif

// Python headers must be included before any system headers, since
// they define _POSIX_C_SOURCE
#include <Python.h>

#if PY_MAJOR_VERSION < 3
#error "Python 2 is not supported"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/util/concept.h"
#include "lucid/util/exception.h"

#ifndef WITHOUT_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif  // WITHOUT_NUMPY

#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString

#define PyArray_SimpleNewFromDataF(nd, dims, typenum, data) \
  PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, data, 0, NPY_ARRAY_FARRAY, NULL)

namespace matplotlibcpp {
namespace detail {

static std::string s_backend;

template <class T>
concept SafeType = lucid::IsAnyOf<T, int, long, std::size_t, bool, double, float, const char*, std::string>;

class _interpreter {
 public:
  PyObject* s_python_function_arrow;
  PyObject* s_python_function_show;
  PyObject* s_python_function_close;
  PyObject* s_python_function_draw;
  PyObject* s_python_function_pause;
  PyObject* s_python_function_save;
  PyObject* s_python_function_figure;
  PyObject* s_python_function_gcf;
  PyObject* s_python_function_fignum_exists;
  PyObject* s_python_function_plot;
  PyObject* s_python_function_quiver;
  PyObject* s_python_function_contour;
  PyObject* s_python_function_semilogx;
  PyObject* s_python_function_semilogy;
  PyObject* s_python_function_loglog;
  PyObject* s_python_function_fill;
  PyObject* s_python_function_fill_between;
  PyObject* s_python_function_hist;
  PyObject* s_python_function_imshow;
  PyObject* s_python_function_scatter;
  PyObject* s_python_function_boxplot;
  PyObject* s_python_function_subplot;
  PyObject* s_python_function_subplot2grid;
  PyObject* s_python_function_legend;
  PyObject* s_python_function_xlim;
  PyObject* s_python_function_ion;
  PyObject* s_python_function_ginput;
  PyObject* s_python_function_ylim;
  PyObject* s_python_function_title;
  PyObject* s_python_function_axis;
  PyObject* s_python_function_axes;
  PyObject* s_python_function_axhline;
  PyObject* s_python_function_axvline;
  PyObject* s_python_function_axvspan;
  PyObject* s_python_function_xlabel;
  PyObject* s_python_function_ylabel;
  PyObject* s_python_function_gca;
  PyObject* s_python_function_xticks;
  PyObject* s_python_function_yticks;
  PyObject* s_python_function_margins;
  PyObject* s_python_function_tick_params;
  PyObject* s_python_function_grid;
  PyObject* s_python_function_cla;
  PyObject* s_python_function_clf;
  PyObject* s_python_function_errorbar;
  PyObject* s_python_function_annotate;
  PyObject* s_python_function_tight_layout;
  PyObject* s_python_colormap;
  PyObject* s_python_empty_tuple;
  PyObject* s_python_function_stem;
  PyObject* s_python_function_xkcd;
  PyObject* s_python_function_text;
  PyObject* s_python_function_suptitle;
  PyObject* s_python_function_bar;
  PyObject* s_python_function_barh;
  PyObject* s_python_function_colorbar;
  PyObject* s_python_function_subplots_adjust;
  PyObject* s_python_function_rcparams;
  PyObject* s_python_function_spy;

  /* For now, _interpreter is implemented as a singleton since its currently not possible to have
     multiple independent embedded python interpreters without patching the python source code
     or starting a separate process for each. [1]
     Furthermore, many python objects expect that they are destructed in the same thread as they
     were constructed. [2] So for advanced usage, a `kill()` function is provided so that library
     users can manually ensure that the interpreter is constructed and destroyed within the
     same thread.

       1: http://bytes.com/topic/python/answers/793370-multiple-independent-python-interpreters-c-c-program
       2: https://github.com/lava/matplotlib-cpp/pull/202#issue-436220256
     */

  static _interpreter& get() { return interkeeper(false); }
  static _interpreter& kill() { return interkeeper(true); }

  PyObject* py_get_function(PyObject* module, const std::string& name);
  PyObject* py_import(const std::string& name);

#ifndef WITHOUT_NUMPY
  static void* import_numpy() {
    fmt::println("importing numpy");
    std::cout << std::endl;
    import_array();  // initialize C-API
    return nullptr;
  }
#endif

 private:
  static _interpreter& interkeeper(const bool should_kill) {
    static _interpreter ctx;
    if (should_kill) ctx.~_interpreter();
    return ctx;
  }

  std::map<std::string, PyObject*> imports_;

  _interpreter() {
    wchar_t name[] = L"plotting";  // optional but recommended
    Py_SetProgramName(name);
    Py_Initialize();
    wchar_t const* dummy_args[] = {L"Python"};  // const is needed because literals must not be modified
    PySys_SetArgv(std::size(dummy_args), const_cast<wchar_t**>(dummy_args));

#ifndef WITHOUT_NUMPY
    import_numpy();  // initialize numpy C-API
#endif

    PyObject* const matplotlib = py_import("matplotlib");
    // matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    // or matplotlib.backends is imported for the first time
    if (!s_backend.empty()) {
      PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), s_backend.c_str());
    }
    PyObject* const pyplot = py_import("matplotlib.pyplot");
    s_python_colormap = py_import("matplotlib.cm");
    PyObject* const pylabmod = py_import("pylab");

    s_python_function_arrow = py_get_function(pyplot, "arrow");
    s_python_function_show = py_get_function(pyplot, "show");
    s_python_function_close = py_get_function(pyplot, "close");
    s_python_function_draw = py_get_function(pyplot, "draw");
    s_python_function_pause = py_get_function(pyplot, "pause");
    s_python_function_figure = py_get_function(pyplot, "figure");
    s_python_function_gcf = py_get_function(pyplot, "gcf");
    s_python_function_fignum_exists = py_get_function(pyplot, "fignum_exists");
    s_python_function_plot = py_get_function(pyplot, "plot");
    s_python_function_quiver = py_get_function(pyplot, "quiver");
    s_python_function_contour = py_get_function(pyplot, "contour");
    s_python_function_semilogx = py_get_function(pyplot, "semilogx");
    s_python_function_semilogy = py_get_function(pyplot, "semilogy");
    s_python_function_loglog = py_get_function(pyplot, "loglog");
    s_python_function_fill = py_get_function(pyplot, "fill");
    s_python_function_fill_between = py_get_function(pyplot, "fill_between");
    s_python_function_hist = py_get_function(pyplot, "hist");
    s_python_function_scatter = py_get_function(pyplot, "scatter");
    s_python_function_boxplot = py_get_function(pyplot, "boxplot");
    s_python_function_subplot = py_get_function(pyplot, "subplot");
    s_python_function_subplot2grid = py_get_function(pyplot, "subplot2grid");
    s_python_function_legend = py_get_function(pyplot, "legend");
    s_python_function_xlim = py_get_function(pyplot, "xlim");
    s_python_function_ylim = py_get_function(pyplot, "ylim");
    s_python_function_title = py_get_function(pyplot, "title");
    s_python_function_axis = py_get_function(pyplot, "axis");
    s_python_function_axes = py_get_function(pyplot, "axes");
    s_python_function_axhline = py_get_function(pyplot, "axhline");
    s_python_function_axvline = py_get_function(pyplot, "axvline");
    s_python_function_axvspan = py_get_function(pyplot, "axvspan");
    s_python_function_xlabel = py_get_function(pyplot, "xlabel");
    s_python_function_ylabel = py_get_function(pyplot, "ylabel");
    s_python_function_gca = py_get_function(pyplot, "gca");
    s_python_function_xticks = py_get_function(pyplot, "xticks");
    s_python_function_yticks = py_get_function(pyplot, "yticks");
    s_python_function_margins = py_get_function(pyplot, "margins");
    s_python_function_tick_params = py_get_function(pyplot, "tick_params");
    s_python_function_grid = py_get_function(pyplot, "grid");
    s_python_function_ion = py_get_function(pyplot, "ion");
    s_python_function_ginput = py_get_function(pyplot, "ginput");
    s_python_function_save = py_get_function(pylabmod, "savefig");
    s_python_function_annotate = py_get_function(pyplot, "annotate");
    s_python_function_cla = py_get_function(pyplot, "cla");
    s_python_function_clf = py_get_function(pyplot, "clf");
    s_python_function_errorbar = py_get_function(pyplot, "errorbar");
    s_python_function_tight_layout = py_get_function(pyplot, "tight_layout");
    s_python_function_stem = py_get_function(pyplot, "stem");
    s_python_function_xkcd = py_get_function(pyplot, "xkcd");
    s_python_function_text = py_get_function(pyplot, "text");
    s_python_function_suptitle = py_get_function(pyplot, "suptitle");
    s_python_function_bar = py_get_function(pyplot, "bar");
    s_python_function_barh = py_get_function(pyplot, "barh");
    s_python_function_colorbar = PyObject_GetAttrString(pyplot, "colorbar");
    s_python_function_subplots_adjust = py_get_function(pyplot, "subplots_adjust");
    s_python_function_rcparams = PyObject_GetAttrString(pyplot, "rcParams");
    s_python_function_spy = PyObject_GetAttrString(pyplot, "spy");
#ifndef WITHOUT_NUMPY
    s_python_function_imshow = py_get_function(pyplot, "imshow");
#endif
    s_python_empty_tuple = PyTuple_New(0);
  }

  ~_interpreter() { Py_Finalize(); }
};

/**
 * Convert a C++ value to the corresponding python object.
 * @note if a python object is passed, it will be returned as is, but increasing its reference count,
 * effectively creating a new reference
 * @warning the returned object must be decref'd by the caller
 * @tparam T type of the value
 * @param value value to convert to a python object
 * @return pointer to the python object
 * @return nullptr if the value could not be converted to a python object
 */
template <SafeType T>
PyObject* PyObject_FromValue(const T& value) {
  using SimpleT = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<SimpleT, int>) return PyInt_FromLong(static_cast<long>(value));
  if constexpr (std::is_same_v<SimpleT, long>) return PyInt_FromLong(value);
  if constexpr (std::is_same_v<SimpleT, std::size_t>) return PyLong_FromSize_t(value);
  if constexpr (std::is_same_v<SimpleT, bool>) return PyBool_FromLong(static_cast<long>(value));
  if constexpr (std::is_same_v<SimpleT, double>) return PyFloat_FromDouble(value);
  if constexpr (std::is_same_v<SimpleT, float>) return PyFloat_FromDouble(static_cast<double>(value));
  if constexpr (std::is_same_v<SimpleT, const char*>) return PyString_FromString(value);
  if constexpr (std::is_same_v<SimpleT, std::string>) return PyString_FromString(value.c_str());
  throw std::invalid_argument("Unsupported type");
}
/**
 * No-op function returning the input python object.
 * The reference count remains unchanged.
 * @param value python object
 * @return python object
 */
inline PyObject* PyObject_FromValue(PyObject* const value) { return value; }

/**
 * Set an item in a python `dict` object from a C++ `value`.
 * The item will be added to the dictionary with the key `key`.
 * @tparam T type of the value
 * @param dict Python dictionary object
 * @param key key of the item
 * @param value value of the item
 * @return true if the item was successfully added to the dictionary
 * @return false if the item could not be added to the dictionary
 */
template <SafeType T>
inline bool PyDict_SetItem(PyObject* const dict, const char* const key, const T& value) {
  PyObject* _value = PyObject_FromValue(value);
  const bool res = PyDict_SetItemString(dict, key, _value);
  Py_DECREF(_value);
  return res;
}
/**
 * Set an item in a python `dict` object from a C++ `value`.
 * The item will be added to the dictionary with the key `key`.
 * @tparam T type of the value
 * @param dict Python dictionary object
 * @param key key of the item
 * @param value value of the item
 * @return true if the item was successfully added to the dictionary
 * @return false if the item could not be added to the dictionary
 */
template <SafeType T>
inline bool PyDict_SetItem(PyObject* const dict, const std::string& key, const T& value) {
  return PyDict_SetItem(dict, key.c_str(), value);
}
/**
 * Set an item in a python `dict` object from another python object.
 * The item will be added to the dictionary with the key `key`.
 * @warning this method will steal the reference of the value.
 * In reality, the reference count will be increased by the dict and, if the insertion was successful,
 * it will then be decreased by 1 by the function
 * @param dict Python dictionary object
 * @param key key of the item
 * @param value value of the item
 * @return true if the item was successfully added to the dictionary
 * @return false if the item could not be added to the dictionary
 */
inline bool PyDict_SetItem(PyObject* const dict, const char* const key, PyObject* const value) {
  if (PyDict_SetItemString(dict, key, value)) {
    Py_DECREF(value);
    return true;
  }
  PyErr_Print();
  return false;
}
/**
 * Set a series of items in a python `dict` object from a C++ `values`.
 * The item will be added to the dictionary, each with the corresponding key.
 * @tparam T type of the value in the map
 * @param dict Python dictionary object
 * @param values map of key-value pairs
 * @return true if all the items were successfully added to the dictionary
 * @return false if any of the items could not be added to the dictionary
 */
template <SafeType T>
inline bool PyDict_SetItems(PyObject* const dict, const std::map<std::string, T>& values) {
  bool res = true;
  for (const auto& [key, value] : values) {
    res &= PyDict_SetItem(dict, key, value);
  }
  return res;
}

/**
 * Create a python tuple object from a series of C++ values or other python objects.
 * @note if a python object is passed, its reference will be stolen and the tuple will own it
 * @warning the returned tuple must be decref'd by the caller
 * @tparam Ts types of the values
 * @param values values used to create the tuple
 * @return tuple object
 * @return nullptr if the tuple could not be created
 */
template <class... Ts>
inline PyObject* PyTuple_Create(const Ts&... values) {
  if constexpr (sizeof...(Ts) == 0) return PyTuple_New(0);
  PyObject* tuple = PyTuple_New(sizeof...(Ts));
  if (!tuple) return nullptr;
  std::array<PyObject*, sizeof...(Ts)> items{PyObject_FromValue(values)...};
  for (std::size_t i = 0; i < items.size(); ++i) PyTuple_SET_ITEM(tuple, i, items[i]);
  return tuple;
}

#ifdef WITHOUT_NUMPY

/**
 * Convert a C++ type to a numpy type.
 * If the type is not supported, NPY_NOTYPE is returned.
 * @tparam T c++ type
 * @return numpy type
 */
template <class T>
constexpr NPY_TYPES numpy_type() {
  using SimpleT = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<SimpleT, std::int8_t>) return NPY_INT8;
  if constexpr (std::is_same_v<SimpleT, std::int16_t>) return NPY_INT16;
  if constexpr (std::is_same_v<SimpleT, std::int32_t>) return NPY_INT32;
  if constexpr (std::is_same_v<SimpleT, std::int64_t>) return NPY_INT64;
  if constexpr (std::is_same_v<SimpleT, std::uint8_t>) return NPY_UINT8;
  if constexpr (std::is_same_v<SimpleT, std::uint16_t>) return NPY_UINT16;
  if constexpr (std::is_same_v<SimpleT, std::uint32_t>) return NPY_UINT32;
  if constexpr (std::is_same_v<SimpleT, std::uint64_t>) return NPY_UINT64;
  if constexpr (std::is_same_v<SimpleT, float>) return NPY_FLOAT;
  if constexpr (std::is_same_v<SimpleT, double>) return NPY_DOUBLE;
  if constexpr (std::is_same_v<SimpleT, long double>) return NPY_LONGDOUBLE;
  if constexpr (std::is_same_v<SimpleT, bool>) return NPY_BOOL;
  if constexpr (std::is_same_v<SimpleT, std::complex<float>>) return NPY_CFLOAT;
  if constexpr (std::is_same_v<SimpleT, std::complex<double>>) return NPY_CDOUBLE;
  if constexpr (std::is_same_v<SimpleT, std::complex<long double>>) return NPY_CLONGDOUBLE;
  if constexpr (std::is_same_v<SimpleT, std::string>) return NPY_STRING;
  throw std::runtime_error("Unsupported type for numpy array");
}

/**
 * Create a numpy array from a raw pointer `data` and the number of elements `size`.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param data raw pointer to the data
 * @param size number of elements
 * @return numpy array
 */
template <class Numeric>
PyObject* get_array(const Numeric* const data, const std::size_t size) {
  npy_intp vsize = static_cast<npy_intp>(size);
  constexpr NPY_TYPES type = numpy_type<Numeric>();
  return PyArray_SimpleNewFromDataF(1, &vsize, type, const_cast<void*>(reinterpret_cast<const void*>(data)));
}

/**
 * Create a 2D numpy array from a C++ vector of vectors.
 * All inner vectors have the same size.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Container container type
 * @tparam Numeric numeric type
 * @param m 2D vector
 * @param rows number of rows
 * @param cols number of columns
 * @return 2D numpy array
 */
template <template <class Numeric> class Container, class Numeric>
  requires lucid::SizedDataContainer<Container<Numeric>, Numeric>
PyObject* get_2darray(const Container<Numeric>& m, const std::size_t rows, const std::size_t cols) {
  if (m.size() != rows * cols) throw lucid::LucidPyException("missmatched array size");
  const npy_intp vsize[2] = {static_cast<npy_intp>(rows), static_cast<npy_intp>(cols)};
  _interpreter::import_numpy();
  return PyArray_SimpleNewFromDataF(2, vsize, numpy_type<Numeric>(),
                                    const_cast<void*>(reinterpret_cast<const void*>(m.data())));
}
/**
 * Create a 2D numpy array from a C++ vector of vectors.
 * All inner vectors have the same size.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param m 2D vector
 * @return 2D numpy array
 */
template <class Numeric>
PyObject* get_2darray(const std::vector<std::vector<Numeric>>& m) {
  if (m.empty()) throw std::runtime_error("get_2d_array v too small");

  const npy_intp vsize[2] = {static_cast<npy_intp>(m.size()), static_cast<npy_intp>(m[0].size())};
  _interpreter::import_numpy();
  PyArrayObject* varray = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, vsize, numpy_type<Numeric>()));

  Numeric* vd_begin = static_cast<Numeric*>(PyArray_DATA(varray));
  for (const std::vector<Numeric>& v_row : m) {
    if (v_row.size() != static_cast<std::size_t>(vsize[1])) throw std::runtime_error("Missmatched array size");
    std::copy(v_row.begin(), v_row.end(), vd_begin);
    vd_begin += vsize[1];
  }
  return reinterpret_cast<PyObject*>(varray);
}

#else  // fallback if we don't have numpy: copy every element of the given vector

template <class Numeric>
PyObject* get_array(const Numeric* const data, const std::size_t size) {
  PyObject* tuple = PyTuple_New(size);
  for (std::size_t i = 0; i < size; ++i) {
    PyTuple_SET_ITEM(tuple, i, PyObject_FromValue(data[i]));
  }
  return tuple;
}

/**
 * Create a 2D numpy array from a C++ vector of vectors.
 * All inner vectors have the same size.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Container container type
 * @tparam Numeric numeric type
 * @param m 2D vector
 * @param rows number of rows
 * @param cols number of columns
 * @return 2D numpy array
 */
template <template <class Numeric> class Container, class Numeric>
  requires lucid::SizedDataContainer<Container<Numeric>, Numeric>
PyObject* get_2darray(const Container<Numeric>& m, const std::size_t rows, const std::size_t cols) {
  PyObject* tuple = PyTuple_New(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    PyTuple_SET_ITEM(tuple, i, get_array(m.data() + i * cols, cols));
  }
  return tuple;
}
/**
 * Create a 2D numpy array from a C++ vector of vectors.
 * All inner vectors have the same size.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param m 2D vector
 * @return 2D numpy array
 */
template <class Numeric>
PyObject* get_2darray(const std::vector<std::vector<Numeric>>& m) {
  PyObject* tuple = PyTuple_New(m.size());
  for (std::size_t i = 0; i < m.size(); ++i) {
    PyTuple_SET_ITEM(tuple, i, get_array(m[i]));
  }
  return tuple;
}

#endif  // WITHOUT_NUMPY

/**
 * Create a numpy array from a C++ container.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param v container
 * @return numpy array
 */
template <template <class Numeric> class Container, class Numeric>
  requires lucid::SizedDataContainer<Container<Numeric>, Numeric>
PyObject* get_array(const Container<Numeric>& v) {
  return get_array(v.data(), v.size());
}
/**
 * Create a numpy array from a eigen vector.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param v vector
 * @return numpy array
 */
template <class Numeric>
PyObject* get_array(const Eigen::VectorX<Numeric>& v) {
  return get_array(v.data(), v.size());
}

/**
 * Create a Python list from a C++ vector of strings.
 * @warning the returned list must be DECREF'ed when it is no longer needed.
 * @param strings vector of strings
 * @return Python list
 */
template <lucid::SizedDataContainer<std::string> Container>
inline PyObject* get_array(const Container& strings) {
  PyObject* list = PyList_New(strings.size());
  for (std::size_t i = 0; i < strings.size(); ++i) {
    PyList_SetItem(list, i, PyString_FromString(strings[i].c_str()));
  }
  return list;
}

/**
 * Create a Python 2D list from a C++ vector of vectors.
 * @warning the returned list must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param m 2D vector
 * @return Python 2D list
 */
template <class Numeric>
PyObject* get_listlist(const std::vector<std::vector<Numeric>>& m) {
  PyObject* listlist = PyList_New(m.size());
  for (std::size_t i = 0; i < m.size(); ++i) {
    PyList_SET_ITEM(listlist, i, get_array(m[i]));
  }
  return listlist;
}

/**
 * Create a 2D numpy array from an eigen matrix.
 * @warning the returned numpy array must be DECREF'ed when it is no longer needed.
 * @tparam Numeric numeric type
 * @param m matrix
 * @return 2D numpy array
 */
template <class Numeric>
PyObject* get_2darray(const Eigen::MatrixX<Numeric>& m) {
  return get_2darray(std::span<const Numeric>{m.data(), static_cast<std::size_t>(m.size())}, m.rows(), m.cols());
}

inline PyObject* get_3d_axis(const long fig_number) {
  PyObject* fig;
  if (fig_number >= 0) {
    PyObject* fig_arg = PyObject_FromValue(fig_number);
    fig = PyObject_CallOneArg(_interpreter::get().s_python_function_figure, fig_arg);
    Py_DECREF(fig_arg);
  } else {
    fig = PyObject_CallNoArgs(_interpreter::get().s_python_function_gcf);
  }
  if (!fig) throw std::runtime_error("Call to figure() failed.");
  Py_DECREF(fig);
  PyObject* gca_kwargs = PyDict_New();
  PyDict_SetItem(gca_kwargs, "projection", static_cast<const char*>("3d"));
  PyObject* axis =
      PyObject_Call(_interpreter::get().s_python_function_axes, _interpreter::get().s_python_empty_tuple, gca_kwargs);
  Py_DECREF(gca_kwargs);
  return axis;
}

}  // namespace detail
/**
 * Set the backend used by matplotlib.
 * Use 'AGG', 'PDF', 'PS', 'SVG', 'Cairo' in non-interactive mode (i.e. you won't be able to run @ref show()).
 * Use 'WebAgg', 'QtAgg', 'GTK3Agg', 'GTK3Cairo', 'wxAgg', 'TkAgg' for interactive mode.
 * The interactive backend will only work if the required python packages are installed (e.g., `tornado` for 'WebAgg').
 * @note This must be called before the first plot command to have any effect.
 * @param name The name of the backend to use.
 * @see https://matplotlib.org/stable/users/explain/figure/backends.html
 */
inline void backend(const std::string& name) { detail::s_backend = name; }

/**
 * Annotate the point xy with `text`.
 * In the simplest form, the `text` is placed at (`x`, `y`).
 * @param text annotation text
 * @param x x coordinate
 * @param y y coordinate
 * @return true if the annotation was successful, false otherwise
 */
inline bool annotate(const std::string& text, const double x, const double y) {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  detail::PyDict_SetItem(kwargs, "xy", detail::PyTuple_Create(x, y));
  detail::PyDict_SetItem(kwargs, "s", text);

  PyObject* args = PyTuple_New(0);
  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_annotate, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);

  return res;
}

/**
 * Plot `y` versus `x` as lines and/or markers.
 * The coordinates of the points or line nodes are given by `x`, `y`.
 * The optional parameter `fmt` is a convenient way for defining basic formatting like color, marker and linestyle.
 * @tparam Numeric numeric type
 * @param x x data
 * @param y y data
 * @param keywords additional keywords
 * @return true if the plot was successful
 * @return false if an error occurred
 */
template <class Numeric>
bool plot(const std::vector<Numeric>& x, const std::vector<Numeric>& y,
          const std::map<std::string, std::string>& keywords) {
  if (x.size() != y.size()) throw std::invalid_argument("x and y data must have the same size");
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(detail::get_array(x), detail::get_array(y));

  PyObject* kwargs = PyDict_New();
  for (const auto& [key, value] : keywords) {
    detail::PyDict_SetItem(kwargs, key, value);
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
  return res;
}

#ifndef WITHOUT_NUMPY

#if 0
// TODO(tend): make it work with vector of vectors
template <template <class NumericX> class ContainerX, class NumericX, template <class NumericY> class ContainerY,
          class NumericY, template <class NumericZ> class ContainerZ, class NumericZ>
  requires lucid::SizedDataContainer<ContainerX<NumericX>, NumericX> &&
           lucid::SizedDataContainer<ContainerY<NumericY>, NumericY> &&
           lucid::SizedDataContainer<ContainerZ<NumericZ>, NumericZ>
void plot_surface(const std::vector<std::vector<Numeric>>& x, const std::vector<std::vector<Numeric>>& y,
                  const std::vector<std::vector<Numeric>>& z,
                  const std::map<std::string, std::string>& keywords = std::map<std::string, std::string>(),
                  const long fig_number = 0) {
  detail::_interpreter::get();

  // We lazily load the modules here the first time this function is called
  // because I'm not sure that we can assume "matplotlib installed" implies
  // "mpl_toolkits installed" on all platforms, and we don't want to require
  // it for people who don't need 3d plots.
  static PyObject *mpl_toolkitsmod = nullptr, *axis3dmod = nullptr;
  if (!mpl_toolkitsmod) {
    detail::_interpreter::get();

    PyObject* mpl_toolkits = PyString_FromString("mpl_toolkits");
    PyObject* axis3d = PyString_FromString("mpl_toolkits.mplot3d");
    if (!mpl_toolkits || !axis3d) {
      throw std::runtime_error("couldnt create string");
    }

    mpl_toolkitsmod = PyImport_Import(mpl_toolkits);
    Py_DECREF(mpl_toolkits);
    if (!mpl_toolkitsmod) {
      throw std::runtime_error("Error loading module mpl_toolkits!");
    }

    axis3dmod = PyImport_Import(axis3d);
    Py_DECREF(axis3d);
    if (!axis3dmod) {
      throw std::runtime_error("Error loading module mpl_toolkits.mplot3d!");
    }
  }

  assert(x.size() == y.size());
  assert(y.size() == z.size());

  // using numpy arrays
  PyObject* xarray = detail::get_2darray(x);
  PyObject* yarray = detail::get_2darray(y);
  PyObject* zarray = detail::get_2darray(z);

  // construct positional args
  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, xarray);
  PyTuple_SET_ITEM(args, 1, yarray);
  PyTuple_SET_ITEM(args, 2, zarray);

  // Build up the kw args.
  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "rstride", PyInt_FromLong(1));
  PyDict_SetItemString(kwargs, "cstride", PyInt_FromLong(1));

  PyObject* python_colormap_coolwarm =
      PyObject_GetAttrString(detail::_interpreter::get().s_python_colormap, "coolwarm");

  PyDict_SetItemString(kwargs, "cmap", python_colormap_coolwarm);

  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    if (it->first == "linewidth" || it->first == "alpha") {
      PyDict_SetItemString(kwargs, it->first.c_str(), PyFloat_FromDouble(std::stod(it->second)));
    } else {
      PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }
  }

  PyObject* fig_args = PyTuple_New(1);
  PyObject* fig = nullptr;
  PyTuple_SET_ITEM(fig_args, 0, PyLong_FromLong(fig_number));
  PyObject* fig_exists = PyObject_CallObject(detail::_interpreter::get().s_python_function_fignum_exists, fig_args);
  if (!PyObject_IsTrue(fig_exists)) {
    fig = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure,
                              detail::_interpreter::get().s_python_empty_tuple);
  } else {
    fig = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure, fig_args);
  }
  Py_DECREF(fig_exists);
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject* gca_kwargs = PyDict_New();
  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject* gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject* axis = PyObject_Call(gca, detail::_interpreter::get().s_python_empty_tuple, gca_kwargs);

  if (!axis) throw std::runtime_error("No axis");
  Py_INCREF(axis);

  Py_DECREF(gca);
  Py_DECREF(gca_kwargs);

  PyObject* plot_surface = PyObject_GetAttrString(axis, "plot_surface");
  if (!plot_surface) throw std::runtime_error("No surface");
  Py_INCREF(plot_surface);
  PyObject* res = PyObject_Call(plot_surface, args, kwargs);
  if (!res) throw std::runtime_error("failed surface");
  Py_DECREF(plot_surface);

  Py_DECREF(axis);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}
#endif
template <class Numeric>
void plot_surface(const Eigen::MatrixX<Numeric>& x, const Eigen::MatrixX<Numeric>& y, const Eigen::MatrixX<Numeric>& z,
                  const std::map<std::string, std::string>& keywords = {{"rstride", "1"},
                                                                        {"cstride", "1"},
                                                                        {"cmap", "coolwarm"}},
                  const long fig_number = 0) {
  if (x.rows() != y.rows() || y.rows() != z.rows())
    throw std::invalid_argument("x, y, and z data must have the same size");
  if (x.cols() != y.cols() || y.cols() != z.cols())
    throw std::invalid_argument("x, y, and z data must have the same size");
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(detail::get_2darray(x), detail::get_2darray(y), detail::get_2darray(z));
  PyObject* kwargs = PyDict_New();

  for (const auto& [key, value] : keywords) {
    if (key == "linewidth" || value == "alpha") {
      detail::PyDict_SetItem(kwargs, key.c_str(), std::stod(value));
    } else if (key == "rstride" || key == "cstride") {
      detail::PyDict_SetItem(kwargs, key.c_str(), std::stoi(value));
    } else if (key == "cmap") {
      detail::PyDict_SetItem(kwargs, "cmap",
                             PyObject_GetAttrString(detail::_interpreter::get().s_python_colormap, value.c_str()));
    } else {
      detail::PyDict_SetItem(kwargs, key.c_str(), value);
    }
  }

  PyObject* axis = detail::get_3d_axis(fig_number);
  if (!axis) throw std::runtime_error("No axis");
  PyObject* plot_surface = PyObject_GetAttrString(axis, "plot_surface");
  if (!plot_surface) throw std::runtime_error("No surface");
  PyObject* res = PyObject_Call(plot_surface, args, kwargs);
  if (!res) throw std::runtime_error("failed surface");

  Py_DECREF(axis);
  Py_DECREF(plot_surface);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}
template <class Numeric>
void plot_surface(const Eigen::MatrixX<Numeric>& z,
                  const std::map<std::string, std::string>& keywords = {{"rstride", "1"},
                                                                        {"cstride", "1"},
                                                                        {"cmap", "coolwarm"}},
                  const long fig_number = 0) {
  Eigen::MatrixX<Numeric> X, Y;
  lucid::meshgrid(Eigen::VectorX<Numeric>::LinSpaced(z.cols(), 0, z.cols() - 1),
                 Eigen::VectorX<Numeric>::LinSpaced(z.rows(), 0, z.rows() - 1), X, Y);
  std::cout << X << std::endl;
  std::cout << Y << std::endl;
  return plot_surface(X, Y, z, keywords, fig_number);
}

template <class Numeric>
void plot_wireframe(
    const Eigen::MatrixX<Numeric>& x, const Eigen::MatrixX<Numeric>& y, const Eigen::MatrixX<Numeric>& z,
    const std::map<std::string, std::string>& keywords = {{"rstride", "1"}, {"cstride", "1"}, {"cmap", "coolwarm"}},
    const long fig_number = 0) {
  if (x.rows() != y.rows() || y.rows() != z.rows())
    throw std::invalid_argument("x, y, and z data must have the same size");
  if (x.cols() != y.cols() || y.cols() != z.cols())
    throw std::invalid_argument("x, y, and z data must have the same size");
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(detail::get_2darray(x), detail::get_2darray(y), detail::get_2darray(z));
  PyObject* kwargs = PyDict_New();
  for (const auto& [key, value] : keywords) {
    if (key == "linewidth" || value == "alpha") {
      detail::PyDict_SetItem(kwargs, key.c_str(), std::stod(value));
    } else if (key == "rstride" || key == "cstride") {
      detail::PyDict_SetItem(kwargs, key.c_str(), std::stoi(value));
    } else if (key == "cmap") {
      detail::PyDict_SetItem(kwargs, "cmap",
                             PyObject_GetAttrString(detail::_interpreter::get().s_python_colormap, value.c_str()));
    } else {
      detail::PyDict_SetItem(kwargs, key.c_str(), value);
    }
  }

  PyObject* fig_arg = detail::PyObject_FromValue(fig_number);
  const PyObject* fig = PyObject_CallOneArg(detail::_interpreter::get().s_python_function_figure, fig_arg);
  Py_DECREF(fig_arg);
  if (!fig) throw std::runtime_error("Call to figure() failed.");
  Py_DECREF(fig);

  PyObject* gca_kwargs = PyDict_New();
  detail::PyDict_SetItem(gca_kwargs, "projection", static_cast<const char*>("3d"));
  PyObject* axis = PyObject_Call(detail::_interpreter::get().s_python_function_axes,
                                 detail::_interpreter::get().s_python_empty_tuple, gca_kwargs);
  Py_DECREF(gca_kwargs);
  if (!axis) throw std::runtime_error("No axis");
  PyObject* plot_wireframe = PyObject_GetAttrString(axis, "plot_wireframe");
  if (!plot_wireframe) throw std::runtime_error("No surface");
  PyObject* res = PyObject_Call(plot_wireframe, args, kwargs);
  if (!res) throw std::runtime_error("failed surface");

  Py_DECREF(axis);
  Py_DECREF(plot_wireframe);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}
template <class Numeric>
void plot_wireframe(const Eigen::MatrixX<Numeric>& z,
                    const std::map<std::string, std::string>& keywords = {{"rstride", "1"},
                                                                          {"cstride", "1"},
                                                                          {"cmap", "coolwarm"}},
                    const long fig_number = 0) {
  Eigen::MatrixX<Numeric> X, Y;
  lucid::meshgrid(Eigen::VectorX<Numeric>::LinSpaced(z.cols(), 0, z.cols() - 1),
                 Eigen::VectorX<Numeric>::LinSpaced(z.rows(), 0, z.rows() - 1), X, Y);
  return plot_wireframe(X, Y, z, keywords, fig_number);
}

template <class Numeric>
void contour(const std::vector<::std::vector<Numeric>>& x, const std::vector<::std::vector<Numeric>>& y,
             const std::vector<::std::vector<Numeric>>& z, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  // using numpy arrays
  PyObject* xarray = detail::get_2darray(x);
  PyObject* yarray = detail::get_2darray(y);
  PyObject* zarray = detail::get_2darray(z);

  // construct positional args
  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, xarray);
  PyTuple_SET_ITEM(args, 1, yarray);
  PyTuple_SET_ITEM(args, 2, zarray);

  // Build up the kw args.
  PyObject* kwargs = PyDict_New();

  PyObject* python_colormap_coolwarm =
      PyObject_GetAttrString(detail::_interpreter::get().s_python_colormap, "coolwarm");

  PyDict_SetItemString(kwargs, "cmap", python_colormap_coolwarm);

  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_contour, args, kwargs);
  if (!res) throw std::runtime_error("failed contour");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}

template <class Numeric>
void spy(const std::vector<::std::vector<Numeric>>& x,
         const double markersize = -1,  // -1 for default matplotlib size
         const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* xarray = detail::get_2darray(x);

  PyObject* kwargs = PyDict_New();
  if (markersize != -1) {
    PyDict_SetItemString(kwargs, "markersize", PyFloat_FromDouble(markersize));
  }
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* plot_args = PyTuple_New(1);
  PyTuple_SET_ITEM(plot_args, 0, xarray);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_spy, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}
#endif  // WITHOUT_NUMPY

template <class Numeric>
void plot3(const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::vector<Numeric>& z,
           const std::map<std::string, std::string>& keywords = std::map<std::string, std::string>(),
           const long fig_number = 0) {
  detail::_interpreter::get();

  // Same as with plot_surface: We lazily load the modules here the first time
  // this function is called because I'm not sure that we can assume "matplotlib
  // installed" implies "mpl_toolkits installed" on all platforms, and we don't
  // want to require it for people who don't need 3d plots.
  static PyObject *mpl_toolkitsmod = nullptr, *axis3dmod = nullptr;
  if (!mpl_toolkitsmod) {
    detail::_interpreter::get();

    PyObject* mpl_toolkits = PyString_FromString("mpl_toolkits");
    PyObject* axis3d = PyString_FromString("mpl_toolkits.mplot3d");
    if (!mpl_toolkits || !axis3d) {
      throw std::runtime_error("couldnt create string");
    }

    mpl_toolkitsmod = PyImport_Import(mpl_toolkits);
    Py_DECREF(mpl_toolkits);
    if (!mpl_toolkitsmod) {
      throw std::runtime_error("Error loading module mpl_toolkits!");
    }

    axis3dmod = PyImport_Import(axis3d);
    Py_DECREF(axis3d);
    if (!axis3dmod) {
      throw std::runtime_error("Error loading module mpl_toolkits.mplot3d!");
    }
  }

  assert(x.size() == y.size());
  assert(y.size() == z.size());

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);
  PyObject* zarray = detail::get_array(z);

  // construct positional args
  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, xarray);
  PyTuple_SET_ITEM(args, 1, yarray);
  PyTuple_SET_ITEM(args, 2, zarray);

  // Build up the kw args.
  PyObject* kwargs = PyDict_New();

  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* fig_args = PyTuple_New(1);
  PyObject* fig = nullptr;
  PyTuple_SET_ITEM(fig_args, 0, PyLong_FromLong(fig_number));
  PyObject* fig_exists = PyObject_CallObject(detail::_interpreter::get().s_python_function_fignum_exists, fig_args);
  if (!PyObject_IsTrue(fig_exists)) {
    fig = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure,
                              detail::_interpreter::get().s_python_empty_tuple);
  } else {
    fig = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure, fig_args);
  }
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject* gca_kwargs = PyDict_New();
  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject* gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject* axis = PyObject_Call(gca, detail::_interpreter::get().s_python_empty_tuple, gca_kwargs);

  if (!axis) throw std::runtime_error("No axis");
  Py_INCREF(axis);

  Py_DECREF(gca);
  Py_DECREF(gca_kwargs);

  PyObject* plot3 = PyObject_GetAttrString(axis, "plot");
  if (!plot3) throw std::runtime_error("No 3D line plot");
  Py_INCREF(plot3);
  PyObject* res = PyObject_Call(plot3, args, kwargs);
  if (!res) throw std::runtime_error("Failed 3D line plot");
  Py_DECREF(plot3);

  Py_DECREF(axis);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}

template <class Numeric>
bool stem(const std::vector<Numeric>& x, const std::vector<Numeric>& y,
          const std::map<std::string, std::string>& keywords) {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  // using numpy arrays
  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  // construct positional args
  PyObject* args = PyTuple_New(2);
  PyTuple_SET_ITEM(args, 0, xarray);
  PyTuple_SET_ITEM(args, 1, yarray);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_stem, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

template <class Numeric>
bool fill(const std::vector<Numeric>& x, const std::vector<Numeric>& y,
          const std::map<std::string, std::string>& keywords) {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  // using numpy arrays
  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  // construct positional args
  PyObject* args = PyTuple_New(2);
  PyTuple_SET_ITEM(args, 0, xarray);
  PyTuple_SET_ITEM(args, 1, yarray);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_fill, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);

  return res;
}

/**
 * Fill the area between two horizontal curves.
 * The curves are defined by the points (`x`, `y1`) and (`x`, `y2`).
 * This creates one or multiple polygons describing the filled area.
 * You may exclude some horizontal sections from filling using where.
 * By default, the edges connect the given points directly. Use step if the filling should be a step function,
 * i.e. constant in between `x`.
 * @tparam ContainerX container type for x
 * @tparam NumericX numeric type for x
 * @tparam ContainerY1 container type for y1
 * @tparam NumericY1 numeric type for y1
 * @tparam ContainerY2 container type for y2
 * @tparam NumericY2 numeric type for y2
 * @param x x coordinates of the nodes defining the curves.
 * @param y1 y coordinates of the nodes defining the first curve.
 * If it contains a single value, it is replicated for every element of `x`
 * @param y2 y coordinates of the nodes defining the second curve.
 * If it contains a single value, it is replicated for every element of `x`
 * @param alpha the alpha blending value, between 0 (transparent) and 1 (opaque)
 * @param keywords additional keywords
 * @return true if the fill was successful
 * @return false if an error occurred
 */
template <template <class NumericX> class ContainerX, class NumericX, template <class NumericY1> class ContainerY1,
          class NumericY1, template <class NumericY2> class ContainerY2, class NumericY2>
  requires lucid::SizedDataContainer<ContainerX<NumericX>, NumericX> &&
           lucid::SizedDataContainer<ContainerY1<NumericY1>, NumericY1> &&
           lucid::SizedDataContainer<ContainerY2<NumericY2>, NumericY2>
bool fill_between(const ContainerX<NumericX>& x, const ContainerY1<NumericY1>& y1, const ContainerY2<NumericY2>& y2,
                  const double alpha = 1, const std::map<std::string, std::string>& keywords = {}) {
  if (y1.size() != 1 && x.size() != y1.size()) throw lucid::LucidPyException("missmatched array size x and y1");
  if (y2.size() != 1 && x.size() != y2.size()) throw lucid::LucidPyException("missmatched array size x and y2");

  detail::_interpreter::get();

  PyObject* _y1 = y1.size() > 1 ? detail::get_array(y1) : detail::PyObject_FromValue(y1[0]);
  PyObject* _y2 = y2.size() > 1 ? detail::get_array(y2) : detail::PyObject_FromValue(y2[0]);

  // construct positional args
  PyObject* args = detail::PyTuple_Create(detail::get_array(x), _y1, _y2);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  detail::PyDict_SetItems(kwargs, keywords);
  detail::PyDict_SetItem(kwargs, "alpha", PyFloat_FromDouble(alpha));

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_fill_between, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
  return res;
}
template <class NumericX, class NumericY1, class NumericY2, int SizeX, int SizeY, int SizeZ>
bool fill_between(const Eigen::Vector<NumericX, SizeX>& x, const Eigen::Vector<NumericY1, SizeY>& y1,
                  const Eigen::Vector<NumericY2, SizeZ>& y2, const double alpha = 1,
                  const std::map<std::string, std::string>& keywords = {}) {
  return fill_between(std::span<const NumericX>{x.data(), static_cast<std::size_t>(x.size())},
                      std::span<const NumericY1>{y1.data(), static_cast<std::size_t>(y1.size())},
                      std::span<const NumericY2>{y2.data(), static_cast<std::size_t>(y2.size())}, alpha, keywords);
}

template <class Numeric>
bool arrow(Numeric x, Numeric y, Numeric end_x, Numeric end_y, const std::string& fc = "r", const std::string ec = "k",
           Numeric head_length = 0.25, Numeric head_width = 0.1625) {
  PyObject* obj_x = PyFloat_FromDouble(x);
  PyObject* obj_y = PyFloat_FromDouble(y);
  PyObject* obj_end_x = PyFloat_FromDouble(end_x);
  PyObject* obj_end_y = PyFloat_FromDouble(end_y);

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "fc", PyString_FromString(fc.c_str()));
  PyDict_SetItemString(kwargs, "ec", PyString_FromString(ec.c_str()));
  PyDict_SetItemString(kwargs, "head_width", PyFloat_FromDouble(head_width));
  PyDict_SetItemString(kwargs, "head_length", PyFloat_FromDouble(head_length));

  PyObject* plot_args = PyTuple_New(4);
  PyTuple_SET_ITEM(plot_args, 0, obj_x);
  PyTuple_SET_ITEM(plot_args, 1, obj_y);
  PyTuple_SET_ITEM(plot_args, 2, obj_end_x);
  PyTuple_SET_ITEM(plot_args, 3, obj_end_y);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_arrow, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

template <class Numeric>
bool hist(const std::vector<Numeric>& y, long bins = 10, std::string color = "b", double alpha = 1.0,
          bool cumulative = false) {
  detail::_interpreter::get();

  PyObject* yarray = detail::get_array(y);

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
  PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
  PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
  PyDict_SetItemString(kwargs, "cumulative", cumulative ? Py_True : Py_False);

  PyObject* plot_args = PyTuple_New(1);

  PyTuple_SET_ITEM(plot_args, 0, yarray);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_hist, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

#ifndef WITHOUT_NUMPY
namespace detail {

inline void imshow(void* ptr, const NPY_TYPES type, const int rows, const int columns, const int colors,
                   const std::map<std::string, std::string>& keywords, PyObject** out) {
  assert(type == NPY_UINT8 || type == NPY_FLOAT);
  assert(colors == 1 || colors == 3 || colors == 4);

  detail::_interpreter::get();

  // construct args
  npy_intp dims[3] = {rows, columns, colors};
  PyObject* args = PyTuple_New(1);
  _interpreter::import_numpy();
  PyTuple_SET_ITEM(args, 0, PyArray_SimpleNewFromDataF(colors == 1 ? 2 : 3, dims, type, ptr));

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_imshow, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (!res) throw std::runtime_error("Call to imshow() failed");
  if (out)
    *out = res;
  else
    Py_DECREF(res);
}

}  // namespace detail

inline void imshow(const unsigned char* ptr, const int rows, const int columns, const int colors,
                   const std::map<std::string, std::string>& keywords = {}, PyObject** out = nullptr) {
  detail::imshow(const_cast<void*>(reinterpret_cast<const void*>(ptr)), NPY_UINT8, rows, columns, colors, keywords,
                 out);
}

inline void imshow(const float* ptr, const int rows, const int columns, const int colors,
                   const std::map<std::string, std::string>& keywords = {}, PyObject** out = nullptr) {
  detail::imshow(const_cast<void*>(reinterpret_cast<const void*>(ptr)), NPY_FLOAT, rows, columns, colors, keywords,
                 out);
}

#ifdef WITH_OPENCV
void imshow(const cv::Mat& image, const std::map<std::string, std::string>& keywords = {}) {
  // Convert underlying type of matrix, if needed
  cv::Mat image2;
  NPY_TYPES npy_type = NPY_UINT8;
  switch (image.type() & CV_MAT_DEPTH_MASK) {
    case CV_8U:
      image2 = image;
      break;
    case CV_32F:
      image2 = image;
      npy_type = NPY_FLOAT;
      break;
    default:
      image.convertTo(image2, CV_MAKETYPE(CV_8U, image.channels()));
  }

  // If color image, convert from BGR to RGB
  switch (image2.channels()) {
    case 3:
      cv::cvtColor(image2, image2, CV_BGR2RGB);
      break;
    case 4:
      cv::cvtColor(image2, image2, CV_BGRA2RGBA);
  }

  detail::imshow(image2.data, npy_type, image2.rows, image2.cols, image2.channels(), keywords);
}
#endif  // WITH_OPENCV
#endif  // WITHOUT_NUMPY

/**
 * A scatter plot of `y` vs `x` with varying marker size and/or color.
 * @tparam ContainerX x container type
 * @tparam NumericX x numeric type
 * @tparam ContainerY y container type
 * @tparam NumericY y numeric type
 * @param x data points for the horizontal axis
 * @param y data points for the vertical axis
 * @param s marker size in points squared
 * @param keywords additional keywords
 * @return true if the plot was successful
 * @return false if an error occurred
 */
template <template <class NumericX> class ContainerX, class NumericX, template <class NumericY> class ContainerY,
          class NumericY>
  requires lucid::SizedDataContainer<ContainerX<NumericX>, NumericX> &&
           lucid::SizedDataContainer<ContainerY<NumericY>, NumericY>
bool scatter(const ContainerX<NumericX>& x, const ContainerY<NumericY>& y, const double s = 1.0,
             const std::map<std::string, std::string>& keywords = {}) {
  if (x.size() != y.size()) throw std::invalid_argument("x and y must have the same size!");

  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(detail::get_array(x), detail::get_array(y));

  PyObject* kwargs = PyDict_New();
  detail::PyDict_SetItem(kwargs, "s", s);
  detail::PyDict_SetItems(kwargs, keywords);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_scatter, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
  return res;
}
#ifndef WITHOUT_EIGEN
/**
 * A scatter plot of `y` vs `x` with varying marker size and/or color.
 * @tparam NumericX x numeric type
 * @tparam NumericY y numeric type
 * @param x data points for the horizontal axis
 * @param y data points for the vertical axis
 * @param s marker size in points squared
 * @param keywords additional keywords
 * @return true if the plot was successful
 * @return false if an error occurred
 */
template <class NumericX, class NumericY, int SizeX, int SizeY>
bool scatter(const Eigen::Vector<NumericX, SizeX>& x, const Eigen::Vector<NumericY, SizeY>& y, const double s = 1.0,
             const std::map<std::string, std::string>& keywords = {}) {
  return scatter(std::span<const NumericX>{x.data(), static_cast<std::size_t>(x.size())},
                 std::span<const NumericY>{y.data(), static_cast<std::size_t>(y.size())}, s, keywords);
}
#endif  // WITHOUT_EIGEN

template <class NumericX, typename NumericY, typename NumericColors>
bool scatter_colored(const std::vector<NumericX>& x, const std::vector<NumericY>& y,
                     const std::vector<NumericColors>& colors,
                     const double s = 1.0,  // The marker size in points**2
                     const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  assert(x.size() == y.size());

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);
  PyObject* colors_array = detail::get_array(colors);

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "s", PyLong_FromLong(s));
  PyDict_SetItemString(kwargs, "c", colors_array);

  for (const auto& it : keywords) {
    PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
  }

  PyObject* plot_args = PyTuple_New(2);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_scatter, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

template <template <class NumericX> class ContainerX, class NumericX, template <class NumericY> class ContainerY,
          class NumericY, template <class NumericZ> class ContainerZ, class NumericZ>
  requires lucid::SizedDataContainer<ContainerX<NumericX>, NumericX> &&
           lucid::SizedDataContainer<ContainerY<NumericY>, NumericY> &&
           lucid::SizedDataContainer<ContainerZ<NumericZ>, NumericZ>
bool scatter(const ContainerX<NumericX>& x, const ContainerY<NumericY>& y, const ContainerZ<NumericZ>& z,
             const std::string zdir = "z", const double s = 1.0, const std::string& c = "b",
             const std::map<std::string, std::string>& keywords = {}, const long fig_number = -1) {
  if (x.size() != y.size() || y.size() != z.size()) throw lucid::LucidPyException("x, y, and z must have the same size!");
  detail::_interpreter::get();

  // construct positional args
  PyObject* args = detail::PyTuple_Create(detail::get_array(x), detail::get_array(y), detail::get_array(z), zdir, s, c);

  // Build up the kw args.
  PyObject* kwargs = PyDict_New();
  detail::PyDict_SetItems(kwargs, keywords);

  PyObject* axis = detail::get_3d_axis(fig_number);
  if (!axis) throw std::runtime_error("No axis");
  PyObject* scatter = PyObject_GetAttrString(axis, "scatter");
  if (!scatter) throw std::runtime_error("No 3D line plot");
  Py_INCREF(scatter);
  PyObject* res = PyObject_Call(scatter, args, kwargs);
  if (!res) throw std::runtime_error("Failed 3D line plot");

  Py_DECREF(scatter);
  Py_DECREF(axis);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
  return res;
}
template <class NumericX, class NumericY, class NumericZ, int SizeX, int SizeY, int SizeZ>
bool scatter(const Eigen::Vector<NumericX, SizeX>& x, const Eigen::Vector<NumericY, SizeY>& y,
             const Eigen::Vector<NumericZ, SizeZ>& z, const std::string& zdir = "z", const double s = 1.0,
             const std::string& c = "b", const std::map<std::string, std::string>& keywords = {},
             const long fig_number = -1) {
  return scatter(std::span<const NumericX>{x.data(), static_cast<std::size_t>(x.size())},
                 std::span<const NumericY>{y.data(), static_cast<std::size_t>(y.size())},
                 std::span<const NumericZ>{z.data(), static_cast<std::size_t>(z.size())}, zdir, s, c, keywords,
                 fig_number);
}

template <class Numeric>
bool boxplot(const std::vector<std::vector<Numeric>>& data, const std::vector<std::string>& labels = {},
             const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* listlist = detail::get_listlist(data);
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, listlist);

  PyObject* kwargs = PyDict_New();

  // kwargs needs the labels, if there are (the correct number of) labels
  if (!labels.empty() && labels.size() == data.size()) {
    PyDict_SetItemString(kwargs, "labels", detail::get_array(labels));
  }

  // take care of the remaining keywords
  for (const auto& it : keywords) {
    PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_boxplot, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);

  return res;
}

template <class Numeric>
bool boxplot(const std::vector<Numeric>& data, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* vector = detail::get_array(data);
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, vector);

  PyObject* kwargs = PyDict_New();
  for (const auto& it : keywords) {
    PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_boxplot, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);

  return res;
}

template <template <class NumericX> class ContainerX, class NumericX, template <class NumericY> class ContainerY,
          class NumericY>
  requires lucid::SizedDataContainer<ContainerX<NumericX>, NumericX> &&
           lucid::SizedDataContainer<ContainerY<NumericY>, NumericY>
bool bar(const ContainerX<NumericX> x, const ContainerY<NumericY>& y, const std::string& ec = "black",
         const std::string& ls = "-", const double lw = 1.0, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(detail::get_array(x), detail::get_array(y));
  PyObject* kwargs = PyDict_New();
  detail::PyDict_SetItem(kwargs, "ec", ec);
  detail::PyDict_SetItem(kwargs, "ls", ls);
  detail::PyDict_SetItem(kwargs, "lw", lw);
  detail::PyDict_SetItems(kwargs, keywords);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_bar, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
  return res;
}
template <class Numeric>
bool bar(const Eigen::VectorX<Numeric>& x, const Eigen::VectorX<Numeric>& y, std::string ec = "black",
         std::string ls = "-", double lw = 1.0, const std::map<std::string, std::string>& keywords = {}) {
  return bar(std::span<const Numeric>{x.data(), x.size()}, std::span<const Numeric>{y.data(), y.size()}, ec, ls, lw,
             keywords);
}
template <template <class NumericX> class Container, class Numeric>
  requires lucid::SizedDataContainer<Container<Numeric>, Numeric>
bool bar(const Container<Numeric>& y, const std::string& ec = "black", const std::string& ls = "-",
         const double lw = 1.0, const std::map<std::string, std::string>& keywords = {}) {
  std::vector<std::size_t> x(y.size());
  std::iota(x.begin(), x.end(), 0);
  return bar(x, y, ec, ls, lw, keywords);
}
template <class Numeric>
bool bar(const Eigen::VectorX<Numeric>& y, const std::string& ec = "black", const std::string& ls = "-",
         const double lw = 1.0, const std::map<std::string, std::string>& keywords = {}) {
  return bar(std::span<const Numeric>{y.data(), static_cast<std::size_t>(y.size())}, ec, ls, lw, keywords);
}

template <class Numeric>
bool barh(const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& ec = "black",
          const std::string& ls = "-", const double lw = 1.0, const std::map<std::string, std::string>& keywords = {}) {
  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* kwargs = PyDict_New();

  PyDict_SetItemString(kwargs, "ec", PyString_FromString(ec.c_str()));
  PyDict_SetItemString(kwargs, "ls", PyString_FromString(ls.c_str()));
  PyDict_SetItemString(kwargs, "lw", PyFloat_FromDouble(lw));

  for (const auto& [key, value] : keywords) {
    PyObject* _value = PyUnicode_FromString(value.c_str());
    PyDict_SetItemString(kwargs, key.c_str(), _value);
    Py_DECREF(_value);
  }

  PyObject* plot_args = PyTuple_New(2);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_barh, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

inline bool subplots_adjust(const std::map<std::string, double>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, double>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyFloat_FromDouble(it->second));
  }

  PyObject* plot_args = PyTuple_New(0);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_subplots_adjust, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

template <class Numeric>
bool named_hist(std::string label, const std::vector<Numeric>& y, long bins = 10, std::string color = "b",
                double alpha = 1.0) {
  detail::_interpreter::get();

  PyObject* yarray = detail::get_array(y);

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
  PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
  PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
  PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));

  PyObject* plot_args = PyTuple_New(1);
  PyTuple_SET_ITEM(plot_args, 0, yarray);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_hist, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);

  return res;
}

template <template <class NumericX> class ContainerX, class NumericX, template <class NumericY> class ContainerY,
          class NumericY>
  requires lucid::SizedDataContainer<ContainerX<NumericX>, NumericX> &&
           lucid::SizedDataContainer<ContainerY<NumericY>, NumericY>
bool plot(const ContainerX<NumericX>& x, const ContainerY<NumericY>& y, const std::string& s = "") {
  assert(x.size() == y.size());
  detail::_interpreter::get();

  PyObject* args = detail::PyTuple_Create(detail::get_array(x), detail::get_array(y), s);
  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_plot, args);

  Py_DECREF(args);
  Py_XDECREF(res);
  return res;
}
template <class NumericX, class NumericY>
bool plot(const Eigen::VectorX<NumericX>& x, const Eigen::VectorX<NumericY>& y, const std::string& s = "") {
  return plot(std::span<const NumericX>{x.data(), static_cast<std::size_t>(x.size())},
              std::span<const NumericY>{y.data(), static_cast<std::size_t>(y.size())}, s);
}

template <class NumericX, typename NumericY, typename NumericZ>
bool contour(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericZ>& z,
             const std::map<std::string, std::string>& keywords = {}) {
  assert(x.size() == y.size() && x.size() == z.size());

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);
  PyObject* zarray = detail::get_array(z);

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, zarray);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_contour, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY, typename NumericU, typename NumericW>
bool quiver(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericU>& u,
            const std::vector<NumericW>& w, const std::map<std::string, std::string>& keywords = {}) {
  assert(x.size() == y.size() && x.size() == u.size() && u.size() == w.size());

  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);
  PyObject* uarray = detail::get_array(u);
  PyObject* warray = detail::get_array(w);

  PyObject* plot_args = PyTuple_New(4);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, uarray);
  PyTuple_SET_ITEM(plot_args, 3, warray);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_quiver, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY, typename NumericZ, typename NumericU, typename NumericW, typename NumericV>
bool quiver(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericZ>& z,
            const std::vector<NumericU>& u, const std::vector<NumericW>& w, const std::vector<NumericV>& v,
            const std::map<std::string, std::string>& keywords = {}) {
  // set up 3d axes stuff
  static PyObject *mpl_toolkitsmod = nullptr, *axis3dmod = nullptr;
  if (!mpl_toolkitsmod) {
    detail::_interpreter::get();

    PyObject* mpl_toolkits = PyString_FromString("mpl_toolkits");
    PyObject* axis3d = PyString_FromString("mpl_toolkits.mplot3d");
    if (!mpl_toolkits || !axis3d) {
      throw std::runtime_error("couldnt create string");
    }

    mpl_toolkitsmod = PyImport_Import(mpl_toolkits);
    Py_DECREF(mpl_toolkits);
    if (!mpl_toolkitsmod) {
      throw std::runtime_error("Error loading module mpl_toolkits!");
    }

    axis3dmod = PyImport_Import(axis3d);
    Py_DECREF(axis3d);
    if (!axis3dmod) {
      throw std::runtime_error("Error loading module mpl_toolkits.mplot3d!");
    }
  }

  // assert sizes match up
  assert(x.size() == y.size() && x.size() == u.size() && u.size() == w.size() && x.size() == z.size() &&
         x.size() == v.size() && u.size() == v.size());

  // set up parameters
  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);
  PyObject* zarray = detail::get_array(z);
  PyObject* uarray = detail::get_array(u);
  PyObject* warray = detail::get_array(w);
  PyObject* varray = detail::get_array(v);

  PyObject* plot_args = PyTuple_New(6);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, zarray);
  PyTuple_SET_ITEM(plot_args, 3, uarray);
  PyTuple_SET_ITEM(plot_args, 4, warray);
  PyTuple_SET_ITEM(plot_args, 5, varray);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  // get figure gca to enable 3d projection
  PyObject* fig = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure,
                                      detail::_interpreter::get().s_python_empty_tuple);
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject* gca_kwargs = PyDict_New();
  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject* gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject* axis = PyObject_Call(gca, detail::_interpreter::get().s_python_empty_tuple, gca_kwargs);

  if (!axis) throw std::runtime_error("No axis");
  Py_INCREF(axis);
  Py_DECREF(gca);
  Py_DECREF(gca_kwargs);

  // plot our boys bravely, plot them strongly, plot them with a wink and clap
  PyObject* plot3 = PyObject_GetAttrString(axis, "quiver");
  if (!plot3) throw std::runtime_error("No 3D line plot");
  Py_INCREF(plot3);
  PyObject* res = PyObject_Call(plot3, plot_args, kwargs);
  if (!res) throw std::runtime_error("Failed 3D plot");
  Py_DECREF(plot3);
  Py_DECREF(axis);
  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool stem(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(s.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_stem, plot_args);

  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool semilogx(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(s.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_semilogx, plot_args);

  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool semilogy(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(s.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_semilogy, plot_args);

  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool loglog(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(s.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_loglog, plot_args);

  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool errorbar(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericX>& yerr,
              const std::map<std::string, std::string>& keywords = {}) {
  assert(x.size() == y.size());

  detail::_interpreter::get();

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);
  PyObject* yerrarray = detail::get_array(yerr);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyDict_SetItemString(kwargs, "yerr", yerrarray);

  PyObject* plot_args = PyTuple_New(2);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_errorbar, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);

  if (res)
    Py_DECREF(res);
  else
    throw std::runtime_error("Call to errorbar() failed.");

  return res;
}

template <class Numeric>
bool named_plot(const std::string& name, const std::vector<Numeric>& y, const std::string& format = "") {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(format.c_str());

  PyObject* plot_args = PyTuple_New(2);

  PyTuple_SET_ITEM(plot_args, 0, yarray);
  PyTuple_SET_ITEM(plot_args, 1, pystring);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool named_plot(const std::string& name, const std::vector<NumericX>& x, const std::vector<NumericY>& y,
                const std::string& format = "") {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(format.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool named_semilogx(const std::string& name, const std::vector<NumericX>& x, const std::vector<NumericY>& y,
                    const std::string& format = "") {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(format.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_semilogx, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool named_semilogy(const std::string& name, const std::vector<NumericX>& x, const std::vector<NumericY>& y,
                    const std::string& format = "") {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(format.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_semilogy, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class NumericX, typename NumericY>
bool named_loglog(const std::string& name, const std::vector<NumericX>& x, const std::vector<NumericY>& y,
                  const std::string& format = "") {
  detail::_interpreter::get();

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

  PyObject* xarray = detail::get_array(x);
  PyObject* yarray = detail::get_array(y);

  PyObject* pystring = PyString_FromString(format.c_str());

  PyObject* plot_args = PyTuple_New(3);
  PyTuple_SET_ITEM(plot_args, 0, xarray);
  PyTuple_SET_ITEM(plot_args, 1, yarray);
  PyTuple_SET_ITEM(plot_args, 2, pystring);
  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_loglog, plot_args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(plot_args);
  Py_XDECREF(res);

  return res;
}

template <class Numeric>
bool plot(const std::vector<Numeric>& y, const std::string& format = "") {
  std::vector<Numeric> x(y.size());
  for (std::size_t i = 0; i < x.size(); ++i) x.at(i) = i;
  return plot(x, y, format);
}

template <class Numeric>
bool plot(const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords) {
  std::vector<Numeric> x(y.size());
  for (std::size_t i = 0; i < x.size(); ++i) x.at(i) = i;
  return plot(x, y, keywords);
}

template <class Numeric>
bool stem(const std::vector<Numeric>& y, const std::string& format = "") {
  std::vector<Numeric> x(y.size());
  for (std::size_t i = 0; i < x.size(); ++i) x.at(i) = i;
  return stem(x, y, format);
}

template <class Numeric>
void text(Numeric x, Numeric y, const std::string& s = "") {
  detail::_interpreter::get();

  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(x));
  PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(y));
  PyTuple_SET_ITEM(args, 2, PyString_FromString(s.c_str()));

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_text, args);
  if (!res) throw std::runtime_error("Call to text() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

inline void colorbar(PyObject* mappable = NULL, const std::map<std::string, float>& keywords = {}) {
  if (mappable == NULL)
    throw std::runtime_error("Must call colorbar with PyObject* returned from an image, contour, surface, etc.");

  detail::_interpreter::get();

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, mappable);

  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, float>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyFloat_FromDouble(it->second));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_colorbar, args, kwargs);
  if (!res) throw std::runtime_error("Call to colorbar() failed.");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(res);
}

inline long figure(long number = -1) {
  detail::_interpreter::get();

  PyObject* res;
  if (number == -1)
    res = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure,
                              detail::_interpreter::get().s_python_empty_tuple);
  else {
    assert(number > 0);

    PyObject* args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, PyLong_FromLong(number));
    res = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure, args);
    Py_DECREF(args);
  }

  if (!res) throw std::runtime_error("Call to figure() failed.");

  PyObject* num = PyObject_GetAttrString(res, "number");
  if (!num) throw std::runtime_error("Could not get number attribute of figure object");
  const long figureNumber = PyLong_AsLong(num);

  Py_DECREF(num);
  Py_DECREF(res);

  return figureNumber;
}

inline bool fignum_exists(long number) {
  detail::_interpreter::get();

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyLong_FromLong(number));
  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_fignum_exists, args);
  if (!res) throw std::runtime_error("Call to fignum_exists() failed.");

  bool ret = PyObject_IsTrue(res);
  Py_DECREF(res);
  Py_DECREF(args);

  return ret;
}

inline void figure_size(std::size_t w, std::size_t h) {
  detail::_interpreter::get();

  const std::size_t dpi = 100;
  PyObject* size = PyTuple_New(2);
  PyTuple_SET_ITEM(size, 0, PyFloat_FromDouble(static_cast<double>(w) / dpi));
  PyTuple_SET_ITEM(size, 1, PyFloat_FromDouble(static_cast<double>(h) / dpi));

  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "figsize", size);
  PyDict_SetItemString(kwargs, "dpi", PyLong_FromSize_t(dpi));

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_figure,
                                detail::_interpreter::get().s_python_empty_tuple, kwargs);

  Py_DECREF(kwargs);

  if (!res) throw std::runtime_error("Call to figure_size() failed.");
  Py_DECREF(res);
}

inline void legend() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_legend,
                                      detail::_interpreter::get().s_python_empty_tuple);
  if (!res) throw std::runtime_error("Call to legend() failed.");

  Py_DECREF(res);
}

inline void legend(const std::map<std::string, std::string>& keywords) {
  detail::_interpreter::get();

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_legend,
                                detail::_interpreter::get().s_python_empty_tuple, kwargs);
  if (!res) throw std::runtime_error("Call to legend() failed.");

  Py_DECREF(kwargs);
  Py_DECREF(res);
}

template <class Numeric>
inline void set_aspect(Numeric ratio) {
  detail::_interpreter::get();

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(ratio));
  PyObject* kwargs = PyDict_New();

  PyObject* ax = PyObject_CallObject(detail::_interpreter::get().s_python_function_gca,
                                     detail::_interpreter::get().s_python_empty_tuple);
  if (!ax) throw std::runtime_error("Call to gca() failed.");
  Py_INCREF(ax);

  PyObject* set_aspect = PyObject_GetAttrString(ax, "set_aspect");
  if (!set_aspect) throw std::runtime_error("Attribute set_aspect not found.");
  Py_INCREF(set_aspect);

  PyObject* res = PyObject_Call(set_aspect, args, kwargs);
  if (!res) throw std::runtime_error("Call to set_aspect() failed.");
  Py_DECREF(set_aspect);

  Py_DECREF(ax);
  Py_DECREF(args);
  Py_DECREF(kwargs);
}

inline void set_aspect_equal() {
  // expect ratio == "equal". Leaving error handling to matplotlib.
  detail::_interpreter::get();

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyString_FromString("equal"));
  PyObject* kwargs = PyDict_New();

  PyObject* ax = PyObject_CallObject(detail::_interpreter::get().s_python_function_gca,
                                     detail::_interpreter::get().s_python_empty_tuple);
  if (!ax) throw std::runtime_error("Call to gca() failed.");
  Py_INCREF(ax);

  PyObject* set_aspect = PyObject_GetAttrString(ax, "set_aspect");
  if (!set_aspect) throw std::runtime_error("Attribute set_aspect not found.");
  Py_INCREF(set_aspect);

  PyObject* res = PyObject_Call(set_aspect, args, kwargs);
  if (!res) throw std::runtime_error("Call to set_aspect() failed.");
  Py_DECREF(set_aspect);

  Py_DECREF(ax);
  Py_DECREF(args);
  Py_DECREF(kwargs);
}

/**
 * Set the x-axis view limits.
 * @param left left limit
 * @param right right limit
 */
void xlim(double left, double right);
/**
 * Set the y-axis view limits.
 * @param bottom bottom limit
 * @param top top limit
 */
void ylim(double bottom, double top);
/**
 * Get the x-axis view limits.
 * @return array of two doubles: {left limit, right limit}
 */
std::array<double, 2> xlim();
/**
 * Get the y-axis view limits.
 * @return array of two doubles: {bottom limit, top limit}
 */
std::array<double, 2> ylim();

template <class Numeric>
inline void xticks(const std::vector<Numeric>& ticks, const std::vector<std::string>& labels = {},
                   const std::map<std::string, std::string>& keywords = {}) {
  assert(labels.size() == 0 || ticks.size() == labels.size());

  detail::_interpreter::get();

  // using numpy array
  PyObject* ticksarray = detail::get_array(ticks);

  PyObject* args;
  if (labels.size() == 0) {
    // construct positional args
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, ticksarray);
  } else {
    // make tuple of tick labels
    PyObject* labelstuple = PyTuple_New(labels.size());
    for (std::size_t i = 0; i < labels.size(); i++)
      PyTuple_SET_ITEM(labelstuple, i, PyUnicode_FromString(labels[i].c_str()));

    // construct positional args
    args = PyTuple_New(2);
    PyTuple_SET_ITEM(args, 0, ticksarray);
    PyTuple_SET_ITEM(args, 1, labelstuple);
  }

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_xticks, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (!res) throw std::runtime_error("Call to xticks() failed");

  Py_DECREF(res);
}

template <class Numeric>
inline void xticks(const std::vector<Numeric>& ticks, const std::map<std::string, std::string>& keywords) {
  xticks(ticks, {}, keywords);
}

template <class Numeric>
inline void yticks(const std::vector<Numeric>& ticks, const std::vector<std::string>& labels = {},
                   const std::map<std::string, std::string>& keywords = {}) {
  assert(labels.size() == 0 || ticks.size() == labels.size());

  detail::_interpreter::get();

  // using numpy array
  PyObject* ticksarray = detail::get_array(ticks);

  PyObject* args;
  if (labels.size() == 0) {
    // construct positional args
    args = PyTuple_New(1);
    PyTuple_SET_ITEM(args, 0, ticksarray);
  } else {
    // make tuple of tick labels
    PyObject* labelstuple = PyTuple_New(labels.size());
    for (std::size_t i = 0; i < labels.size(); i++)
      PyTuple_SET_ITEM(labelstuple, i, PyUnicode_FromString(labels[i].c_str()));

    // construct positional args
    args = PyTuple_New(2);
    PyTuple_SET_ITEM(args, 0, ticksarray);
    PyTuple_SET_ITEM(args, 1, labelstuple);
  }

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_yticks, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (!res) throw std::runtime_error("Call to yticks() failed");

  Py_DECREF(res);
}

template <class Numeric>
inline void yticks(const std::vector<Numeric>& ticks, const std::map<std::string, std::string>& keywords) {
  yticks(ticks, {}, keywords);
}

template <class Numeric>
inline void margins(Numeric margin) {
  // construct positional args
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(margin));

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_margins, args);
  if (!res) throw std::runtime_error("Call to margins() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

template <class Numeric>
inline void margins(Numeric margin_x, Numeric margin_y) {
  // construct positional args
  PyObject* args = PyTuple_New(2);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(margin_x));
  PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(margin_y));

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_margins, args);
  if (!res) throw std::runtime_error("Call to margins() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

inline void tick_params(const std::map<std::string, std::string>& keywords, const std::string axis = "both") {
  detail::_interpreter::get();

  // construct positional args
  PyObject* args;
  args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyString_FromString(axis.c_str()));

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_tick_params, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (!res) throw std::runtime_error("Call to tick_params() failed");

  Py_DECREF(res);
}

inline void subplot(long nrows, long ncols, long plot_number) {
  detail::_interpreter::get();

  // construct positional args
  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(nrows));
  PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(ncols));
  PyTuple_SET_ITEM(args, 2, PyFloat_FromDouble(plot_number));

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_subplot, args);
  if (!res) throw std::runtime_error("Call to subplot() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

inline void subplot2grid(long nrows, long ncols, long rowid = 0, long colid = 0, long rowspan = 1, long colspan = 1) {
  detail::_interpreter::get();

  PyObject* shape = PyTuple_New(2);
  PyTuple_SET_ITEM(shape, 0, PyLong_FromLong(nrows));
  PyTuple_SET_ITEM(shape, 1, PyLong_FromLong(ncols));

  PyObject* loc = PyTuple_New(2);
  PyTuple_SET_ITEM(loc, 0, PyLong_FromLong(rowid));
  PyTuple_SET_ITEM(loc, 1, PyLong_FromLong(colid));

  PyObject* args = PyTuple_New(4);
  PyTuple_SET_ITEM(args, 0, shape);
  PyTuple_SET_ITEM(args, 1, loc);
  PyTuple_SET_ITEM(args, 2, PyLong_FromLong(rowspan));
  PyTuple_SET_ITEM(args, 3, PyLong_FromLong(colspan));

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_subplot2grid, args);
  if (!res) throw std::runtime_error("Call to subplot2grid() failed.");

  Py_DECREF(shape);
  Py_DECREF(loc);
  Py_DECREF(args);
  Py_DECREF(res);
}

inline void title(const std::string& titlestr, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* pytitlestr = PyString_FromString(titlestr.c_str());
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pytitlestr);

  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_title, args, kwargs);
  if (!res) throw std::runtime_error("Call to title() failed.");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(res);
}

inline void suptitle(const std::string& suptitlestr, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* pysuptitlestr = PyString_FromString(suptitlestr.c_str());
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pysuptitlestr);

  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_suptitle, args, kwargs);
  if (!res) throw std::runtime_error("Call to suptitle() failed.");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(res);
}

inline void axis(const std::string& axisstr) {
  detail::_interpreter::get();

  PyObject* str = PyString_FromString(axisstr.c_str());
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, str);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_axis, args);
  if (!res) throw std::runtime_error("Call to title() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

inline void axhline(double y, double xmin = 0., double xmax = 1.,
                    const std::map<std::string, std::string>& keywords = std::map<std::string, std::string>()) {
  detail::_interpreter::get();

  // construct positional args
  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(y));
  PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(xmin));
  PyTuple_SET_ITEM(args, 2, PyFloat_FromDouble(xmax));

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_axhline, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);
}

inline void axvline(double x, double ymin = 0., double ymax = 1.,
                    const std::map<std::string, std::string>& keywords = std::map<std::string, std::string>()) {
  detail::_interpreter::get();

  // construct positional args
  PyObject* args = PyTuple_New(3);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(x));
  PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(ymin));
  PyTuple_SET_ITEM(args, 2, PyFloat_FromDouble(ymax));

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_axvline, args, kwargs);

  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);
}

inline void axvspan(double xmin, double xmax, double ymin = 0., double ymax = 1.,
                    const std::map<std::string, std::string>& keywords = std::map<std::string, std::string>()) {
  // construct positional args
  PyObject* args = PyTuple_New(4);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(xmin));
  PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(xmax));
  PyTuple_SET_ITEM(args, 2, PyFloat_FromDouble(ymin));
  PyTuple_SET_ITEM(args, 3, PyFloat_FromDouble(ymax));

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    if (it->first == "linewidth" || it->first == "alpha") {
      PyDict_SetItemString(kwargs, it->first.c_str(), PyFloat_FromDouble(std::stod(it->second)));
    } else {
      PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_axvspan, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);

  Py_XDECREF(res);
}

inline void xlabel(const std::string& str, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* pystr = PyString_FromString(str.c_str());
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pystr);

  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_xlabel, args, kwargs);
  if (!res) throw std::runtime_error("Call to xlabel() failed.");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(res);
}

inline void ylabel(const std::string& str, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* pystr = PyString_FromString(str.c_str());
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pystr);

  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_ylabel, args, kwargs);
  if (!res) throw std::runtime_error("Call to ylabel() failed.");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(res);
}

inline void set_zlabel(const std::string& str, const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  // Same as with plot_surface: We lazily load the modules here the first time
  // this function is called because I'm not sure that we can assume "matplotlib
  // installed" implies "mpl_toolkits installed" on all platforms, and we don't
  // want to require it for people who don't need 3d plots.
  static PyObject *mpl_toolkitsmod = nullptr, *axis3dmod = nullptr;
  if (!mpl_toolkitsmod) {
    PyObject* mpl_toolkits = PyString_FromString("mpl_toolkits");
    PyObject* axis3d = PyString_FromString("mpl_toolkits.mplot3d");
    if (!mpl_toolkits || !axis3d) {
      throw std::runtime_error("couldnt create string");
    }

    mpl_toolkitsmod = PyImport_Import(mpl_toolkits);
    Py_DECREF(mpl_toolkits);
    if (!mpl_toolkitsmod) {
      throw std::runtime_error("Error loading module mpl_toolkits!");
    }

    axis3dmod = PyImport_Import(axis3d);
    Py_DECREF(axis3d);
    if (!axis3dmod) {
      throw std::runtime_error("Error loading module mpl_toolkits.mplot3d!");
    }
  }

  PyObject* pystr = PyString_FromString(str.c_str());
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pystr);

  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* ax = PyObject_CallObject(detail::_interpreter::get().s_python_function_gca,
                                     detail::_interpreter::get().s_python_empty_tuple);
  if (!ax) throw std::runtime_error("Call to gca() failed.");
  Py_INCREF(ax);

  PyObject* zlabel = PyObject_GetAttrString(ax, "set_zlabel");
  if (!zlabel) throw std::runtime_error("Attribute set_zlabel not found.");
  Py_INCREF(zlabel);

  PyObject* res = PyObject_Call(zlabel, args, kwargs);
  if (!res) throw std::runtime_error("Call to set_zlabel() failed.");
  Py_DECREF(zlabel);

  Py_DECREF(ax);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_XDECREF(res);
}

inline void grid(bool flag) {
  detail::_interpreter::get();

  PyObject* pyflag = flag ? Py_True : Py_False;
  Py_INCREF(pyflag);

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pyflag);

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_grid, args);
  if (!res) throw std::runtime_error("Call to grid() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

inline void show(const bool block = true) {
  detail::_interpreter::get();

  PyObject* res;
  if (block) {
    res = PyObject_CallObject(detail::_interpreter::get().s_python_function_show,
                              detail::_interpreter::get().s_python_empty_tuple);
  } else {
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "block", Py_False);
    res = PyObject_Call(detail::_interpreter::get().s_python_function_show,
                        detail::_interpreter::get().s_python_empty_tuple, kwargs);
    Py_DECREF(kwargs);
  }

  if (!res) throw std::runtime_error("Call to show() failed.");

  Py_DECREF(res);
}

inline void close() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_close,
                                      detail::_interpreter::get().s_python_empty_tuple);

  if (!res) throw std::runtime_error("Call to close() failed.");

  Py_DECREF(res);
}

inline void xkcd() {
  detail::_interpreter::get();

  PyObject* res;
  PyObject* kwargs = PyDict_New();

  res = PyObject_Call(detail::_interpreter::get().s_python_function_xkcd,
                      detail::_interpreter::get().s_python_empty_tuple, kwargs);

  Py_DECREF(kwargs);

  if (!res) throw std::runtime_error("Call to show() failed.");

  Py_DECREF(res);
}

inline void draw() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_draw,
                                      detail::_interpreter::get().s_python_empty_tuple);

  if (!res) throw std::runtime_error("Call to draw() failed.");

  Py_DECREF(res);
}

template <class Numeric>
inline void pause(Numeric interval) {
  detail::_interpreter::get();

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyFloat_FromDouble(interval));

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_pause, args);
  if (!res) throw std::runtime_error("Call to pause() failed.");

  Py_DECREF(args);
  Py_DECREF(res);
}

inline void save(const std::string& filename, const int dpi = 0) {
  detail::_interpreter::get();

  PyObject* pyfilename = PyString_FromString(filename.c_str());

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pyfilename);

  PyObject* kwargs = PyDict_New();

  if (dpi > 0) {
    PyDict_SetItemString(kwargs, "dpi", PyLong_FromLong(dpi));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_save, args, kwargs);
  if (!res) throw std::runtime_error("Call to save() failed.");

  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(res);
}

inline void rcparams(const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();
  PyObject* args = PyTuple_New(0);
  PyObject* kwargs = PyDict_New();
  for (auto it = keywords.begin(); it != keywords.end(); ++it) {
    if ("text.usetex" == it->first)
      PyDict_SetItemString(kwargs, it->first.c_str(), PyLong_FromLong(std::stoi(it->second.c_str())));
    else
      PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
  }

  PyObject* update = PyObject_GetAttrString(detail::_interpreter::get().s_python_function_rcparams, "update");
  PyObject* res = PyObject_Call(update, args, kwargs);
  if (!res) throw std::runtime_error("Call to rcParams.update() failed.");
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(update);
  Py_DECREF(res);
}

inline void clf() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_clf,
                                      detail::_interpreter::get().s_python_empty_tuple);

  if (!res) throw std::runtime_error("Call to clf() failed.");

  Py_DECREF(res);
}

inline void cla() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_cla,
                                      detail::_interpreter::get().s_python_empty_tuple);

  if (!res) throw std::runtime_error("Call to cla() failed.");

  Py_DECREF(res);
}

inline void ion() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ion,
                                      detail::_interpreter::get().s_python_empty_tuple);

  if (!res) throw std::runtime_error("Call to ion() failed.");

  Py_DECREF(res);
}

inline std::vector<std::array<double, 2>> ginput(const int numClicks = 1,
                                                 const std::map<std::string, std::string>& keywords = {}) {
  detail::_interpreter::get();

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, PyLong_FromLong(numClicks));

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_ginput, args, kwargs);

  Py_DECREF(kwargs);
  Py_DECREF(args);
  if (!res) throw std::runtime_error("Call to ginput() failed.");

  const std::size_t len = PyList_Size(res);
  std::vector<std::array<double, 2>> out;
  out.reserve(len);
  for (std::size_t i = 0; i < len; i++) {
    PyObject* current = PyList_GetItem(res, i);
    std::array<double, 2> position;
    position[0] = PyFloat_AsDouble(PyTuple_GetItem(current, 0));
    position[1] = PyFloat_AsDouble(PyTuple_GetItem(current, 1));
    out.push_back(position);
  }
  Py_DECREF(res);

  return out;
}

// Actually, is there any reason not to call this automatically for every plot?
inline void tight_layout() {
  detail::_interpreter::get();

  PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_tight_layout,
                                      detail::_interpreter::get().s_python_empty_tuple);

  if (!res) throw std::runtime_error("Call to tight_layout() failed.");

  Py_DECREF(res);
}

// Support for variadic plot() and initializer lists:

namespace detail {

template <class T>
using is_function = typename std::is_function<std::remove_pointer<std::remove_reference<T>>>::type;

template <bool obj, typename T>
struct is_callable_impl;

template <class T>
struct is_callable_impl<false, T> {
  typedef is_function<T> type;
};  // a non-object is callable iff it is a function

template <class T>
struct is_callable_impl<true, T> {
  struct Fallback {
    void operator()();
  };
  struct Derived : T, Fallback {};

  template <class U, U>
  struct Check;

  template <class U>
  static std::true_type test(
      ...);  // use a variadic function to make sure (1) it accepts everything and (2) its always the worst match

  template <class U>
  static std::false_type test(Check<void (Fallback::*)(), &U::operator()>*);

 public:
  typedef decltype(test<Derived>(nullptr)) type;
  typedef decltype(&Fallback::operator()) dtype;
  static constexpr bool value = type::value;
};  // an object is callable iff it defines operator()

template <class T>
struct is_callable {
  // dispatch to is_callable_impl<true, T> or is_callable_impl<false, T> depending on whether T is of class type or not
  typedef typename is_callable_impl<std::is_class<T>::value, T>::type type;
};

template <class IsYDataCallable>
struct plot_impl {};

template <>
struct plot_impl<std::false_type> {
  template <class IterableX, typename IterableY>
  bool operator()(const IterableX& x, const IterableY& y, const std::string& format) {
    detail::_interpreter::get();

    // 2-phase lookup for distance, begin, end
    using std::begin;
    using std::distance;
    using std::end;

    auto xs = distance(begin(x), end(x));
    auto ys = distance(begin(y), end(y));
    assert(xs == ys && "x and y data must have the same number of elements!");

    PyObject* xlist = PyList_New(xs);
    PyObject* ylist = PyList_New(ys);
    PyObject* pystring = PyString_FromString(format.c_str());

    auto itx = begin(x), ity = begin(y);
    for (std::size_t i = 0; i < xs; ++i) {
      PyList_SetItem(xlist, i, PyFloat_FromDouble(*itx++));
      PyList_SetItem(ylist, i, PyFloat_FromDouble(*ity++));
    }

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SET_ITEM(plot_args, 0, xlist);
    PyTuple_SET_ITEM(plot_args, 1, ylist);
    PyTuple_SET_ITEM(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_plot, plot_args);

    Py_DECREF(plot_args);
    Py_XDECREF(res);

    return res;
  }
};

template <>
struct plot_impl<std::true_type> {
  template <class Iterable, typename Callable>
  bool operator()(const Iterable& ticks, const Callable& f, const std::string& format) {
    if (begin(ticks) == end(ticks)) return true;

    // We could use additional meta-programming to deduce the correct element type of y,
    // but all values have to be convertible to double anyways
    std::vector<double> y;
    for (auto x : ticks) y.push_back(f(x));
    return plot_impl<std::false_type>()(ticks, y, format);
  }
};

}  // end namespace detail

// recursion stop for the above
template <class... Args>
bool plot() {
  return true;
}

template <class A, typename B, typename... Args>
bool plot(const A& a, const B& b, const std::string& format, Args... args) {
  return detail::plot_impl<class detail::is_callable<B>::type>()(a, b, format) && plot(args...);
}

/*
 * This group of plot() functions is needed to support initializer lists, i.e. calling
 *    plot( {1,2,3,4} )
 */
inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& format = "") {
  return plot(x, y, format);
}

inline bool plot(const std::vector<double>& y, const std::string& format = "") { return plot<double>(y, format); }

inline bool plot(const std::vector<double>& x, const std::vector<double>& y,
                 const std::map<std::string, std::string>& keywords) {
  return plot(x, y, keywords);
}

/*
 * This class allows dynamic plots, ie changing the plotted data without clearing and re-plotting
 */
class Plot {
 public:
  // default initialization with plot label, some data and format
  template <class Numeric>
  Plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y,
       const std::string& format = "") {
    detail::_interpreter::get();

    assert(x.size() == y.size());

    PyObject* kwargs = PyDict_New();
    if (name != "") PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = detail::get_array(x);
    PyObject* yarray = detail::get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SET_ITEM(plot_args, 0, xarray);
    PyTuple_SET_ITEM(plot_args, 1, yarray);
    PyTuple_SET_ITEM(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);

    if (res) {
      line = PyList_GetItem(res, 0);

      if (line)
        set_data_fct = PyObject_GetAttrString(line, "set_data");
      else
        Py_DECREF(line);
      Py_DECREF(res);
    }
  }

  // shorter initialization with name or format only
  // basically calls line, = plot([], [])
  Plot(const std::string& name = "", const std::string& format = "")
      : Plot(name, std::vector<double>(), std::vector<double>(), format) {}

  template <class Numeric>
  bool update(const std::vector<Numeric>& x, const std::vector<Numeric>& y) {
    assert(x.size() == y.size());
    if (set_data_fct) {
      PyObject* xarray = detail::get_array(x);
      PyObject* yarray = detail::get_array(y);

      PyObject* plot_args = PyTuple_New(2);
      PyTuple_SET_ITEM(plot_args, 0, xarray);
      PyTuple_SET_ITEM(plot_args, 1, yarray);

      PyObject* res = PyObject_CallObject(set_data_fct, plot_args);
      Py_XDECREF(res);
      return res;
    }
    return false;
  }

  // clears the plot but keep it available
  bool clear() { return update(std::vector<double>(), std::vector<double>()); }

  // definitely remove this line
  void remove() {
    if (line) {
      auto remove_fct = PyObject_GetAttrString(line, "remove");
      PyObject* args = PyTuple_New(0);
      PyObject* res = PyObject_CallObject(remove_fct, args);
      Py_XDECREF(res);
    }
    decref();
  }

  ~Plot() { decref(); }

 private:
  void decref() {
    if (line) Py_DECREF(line);
    if (set_data_fct) Py_DECREF(set_data_fct);
  }

  PyObject* line = nullptr;
  PyObject* set_data_fct = nullptr;
};

}  // end namespace matplotlibcpp
