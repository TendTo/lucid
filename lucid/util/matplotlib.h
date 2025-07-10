/**
 * @author Room 6.030
 * @author Benno Evers
 * @copyright 2014 (https://github.com/lava/matplotlib-cpp)
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * A C++ wrapper for python's matplotlib.
 * Inspired by Benno Evers' matplotlib-cpp
 */
#pragma once

#ifndef LUCID_MATPLOTLIB_BUILD
#error "This file should not be included without LUCID_LUCID_MATPLOTLIB_BUILD"
#endif

#ifndef WITHOUT_EIGEN
#include <pybind11/eigen.h>
#endif
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#ifndef WITHOUT_NUMPY
#include <pybind11/numpy.h>
#endif

#if PY_MAJOR_VERSION < 3
#error "Python 2 is not supported"
#endif

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef WITHOUT_EIGEN
#include "lucid/lib/eigen.h"
#endif

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces_literals): standard use of literals

/**
 * @namespace lucid::plt
 * Lucid's matplotlib wrapper.
 * Inspired by Benno Evers' [matplotlib-cpp](https://github.com/lava/matplotlib-cpp).
 */
namespace lucid::plt {
namespace internal {

static std::string backend_;  // NOLINT(runtime/string): the backend string is a global variable

#if __GNUC__ >= 4
#define HIDDEN __attribute__((visibility("hidden")))
#else
#define HIDDEN
#endif

/**
 * Singleton representing the Python interpreter.
 * Allows c++ code to call and use Python functions.
 * Only intended for internal usage.
 */
class HIDDEN Interpreter {
 public:
  static Interpreter& get() { return interkeeper(false); }
  static Interpreter& kill() { return interkeeper(true); }
  static void set_backend(const std::string& name) { backend_ = name; }

  py::function arrow() const { return pyplot_.attr("arrow"); }
  py::function show() const { return pyplot_.attr("show"); }
  py::function close() const { return pyplot_.attr("close"); }
  py::function draw() const { return pyplot_.attr("draw"); }
  py::function pause() const { return pyplot_.attr("pause"); }
  py::function figure() const { return pyplot_.attr("figure"); }
  py::function gcf() const { return pyplot_.attr("gcf"); }
  py::function fignum_exists() const { return pyplot_.attr("fignum_exists"); }
  py::function plot() const { return pyplot_.attr("plot"); }
  py::function quiver() const { return pyplot_.attr("quiver"); }
  py::function contour() const { return pyplot_.attr("contour"); }
  py::function semilogx() const { return pyplot_.attr("semilogx"); }
  py::function semilogy() const { return pyplot_.attr("semilogy"); }
  py::function loglog() const { return pyplot_.attr("loglog"); }
  py::function fill() const { return pyplot_.attr("fill"); }
  py::function fill_between() const { return pyplot_.attr("fill_between"); }
  py::function hist() const { return pyplot_.attr("hist"); }
  py::function scatter() const { return pyplot_.attr("scatter"); }
  py::function boxplot() const { return pyplot_.attr("boxplot"); }
  py::function subplot() const { return pyplot_.attr("subplot"); }
  py::function subplot2grid() const { return pyplot_.attr("subplot2grid"); }
  py::function legend() const { return pyplot_.attr("legend"); }
  py::function xlim() const { return pyplot_.attr("xlim"); }
  py::function ylim() const { return pyplot_.attr("ylim"); }
  py::function title() const { return pyplot_.attr("title"); }
  py::function axis() const { return pyplot_.attr("axis"); }
  py::function axes() const { return pyplot_.attr("axes"); }
  py::function axhline() const { return pyplot_.attr("axhline"); }
  py::function axvline() const { return pyplot_.attr("axvline"); }
  py::function axvspan() const { return pyplot_.attr("axvspan"); }
  py::function xlabel() const { return pyplot_.attr("xlabel"); }
  py::function ylabel() const { return pyplot_.attr("ylabel"); }
  py::function gca() const { return pyplot_.attr("gca"); }
  py::function xticks() const { return pyplot_.attr("xticks"); }
  py::function yticks() const { return pyplot_.attr("yticks"); }
  py::function margins() const { return pyplot_.attr("margins"); }
  py::function tick_params() const { return pyplot_.attr("tick_params"); }
  py::function grid() const { return pyplot_.attr("grid"); }
  py::function ion() const { return pyplot_.attr("ion"); }
  py::function ginput() const { return pyplot_.attr("ginput"); }
  py::function annotate() const { return pyplot_.attr("annotate"); }
  py::function cla() const { return pyplot_.attr("cla"); }
  py::function clf() const { return pyplot_.attr("clf"); }
  py::function errorbar() const { return pyplot_.attr("errorbar"); }
  py::function tight_layout() const { return pyplot_.attr("tight_layout"); }
  py::function stem() const { return pyplot_.attr("stem"); }
  py::function xkcd() const { return pyplot_.attr("xkcd"); }
  py::function text() const { return pyplot_.attr("text"); }
  py::function suptitle() const { return pyplot_.attr("suptitle"); }
  py::function bar() const { return pyplot_.attr("bar"); }
  py::function barh() const { return pyplot_.attr("barh"); }
  py::function subplots_adjust() const { return pyplot_.attr("subplots_adjust"); }
  py::function imshow() const { return pyplot_.attr("imshow"); }

  py::function savefig() const { return pylab_.attr("savefig"); }

  // Attributes
  py::str colorbar() const { return pyplot_.attr("colorbar"); }
  auto rcParams() const { return pyplot_.attr("rcParams"); }
  py::function spy() const { return pyplot_.attr("spy"); }

  py::object get_3d_axis(const int fig_number) {
    const py::object fig = fig_number >= 0 ? figure()(fig_number) : gcf()();
    if (fig.is_none()) throw std::runtime_error("Call to figure() failed.");
    const py::list fig_axes = fig.attr("axes");
    if (fig_axes.empty()) return axes()("projection"_a = "3d");  // There is no axes, create one
    if (fig_axes[0].attr("name").cast<std::string>() == "3d") return fig_axes[0];
    throw std::runtime_error("The figure already contains an axis that is not 3D.");
  }

 private:
  static Interpreter& interkeeper(const bool should_kill) {
    static Interpreter ctx;
    if (should_kill) ctx.~Interpreter();
    return ctx;
  }

  Interpreter()
      : matplotlib_(py::module_::import("matplotlib")),
        pyplot_{py::module_::import("matplotlib.pyplot")},
        pylab_{py::module_::import("pylab")} {
    matplotlib_.attr("use")(backend_);
  }

  py::scoped_interpreter guard_;  // Ensure that the python interpreter is released once the object is destroyed
  py::module_ matplotlib_;        // Matplotlib module
  py::module_ pyplot_;            // Pyplot module
  py::module_ pylab_;             // Pylab module
};

}  // namespace internal

/**
 * Set the backend used by matplotlib.
 * Use 'AGG', 'PDF', 'PS', 'SVG', 'Cairo' in non-interactive mode (i.e. you won't be able to run @ref show()).
 * Use 'WebAgg', 'QtAgg', 'GTK3Agg', 'GTK3Cairo', 'wxAgg', 'TkAgg' for interactive mode.
 * The interactive backend will only work if the required python packages are installed (e.g., `tornado` for 'WebAgg').
 * @note This must be called before the first plot command to have any effect.
 * @param name The name of the backend to use.
 * @see https://matplotlib.org/stable/users/explain/figure/backends.html
 */
inline void backend(const std::string& name) { internal::Interpreter::set_backend(name); }

/**
 * Annotate the point xy with `text`.
 * In the simplest form, the `text` is placed at (`x`, `y`).
 * @param text annotation text
 * @param x x coordinate
 * @param y y coordinate
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
 */
inline bool annotate(const std::string& text, const double x, const double y) {
  return static_cast<bool>(internal::Interpreter::get().annotate()(text, "xy"_a = std::array<double, 2>{x, y}));
}

/**
 * Create a new figure or select an existing figure.
 * @param fig_number number of the figure to select. If negative, a new figure is created.
 * @return the figure number
 */
inline int figure(const int fig_number = -1) {
  const py::object res =
      fig_number >= 0 ? internal::Interpreter::get().figure()(fig_number) : internal::Interpreter::get().figure()();
  if (!res) throw std::runtime_error("Call to figure() failed.");
  return res.attr("number").cast<int>();
}

/** Keyword arguments for the plot function */
struct PlotKwargs {
  std::string fmt{""};  ///< format string
  float alpha{1};       ///< transparency
  bool scalex{true};    ///< scale x-axis
  bool scaley{true};    ///< scale y-axis
};

/**
 * Plot `y` versus `x` as lines and/or markers.
 * The coordinates of the points or line nodes are given by `x`, `y`.
 * The optional parameter `fmt` is a convenient way for defining basic formatting like color, marker and linestyle.
 * @tparam ContainerX container type for x data
 * @tparam ContainerY container type for y data
 * @param x x data
 * @param y y data
 * @param kwargs additional keywords
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
 */
template <class ContainerX, class ContainerY>
bool plot(const ContainerX& x, const ContainerY& y, const PlotKwargs& kwargs = {}) {
  if (x.size() != y.size()) throw std::invalid_argument("x and y data must have the same size");
  return static_cast<bool>(internal::Interpreter::get().plot()(x, y, kwargs.fmt, "alpha"_a = kwargs.alpha,
                                                               "scalex"_a = kwargs.scalex, "scaley"_a = kwargs.scaley));
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
                  const int fig_number = 0) {
  internal::_interpreter::get();

  // We lazily load the modules here the first time this function is called
  // because I'm not sure that we can assume "matplotlib installed" implies
  // "mpl_toolkits installed" on all platforms, and we don't want to require
  // it for people who don't need 3d plots.
  static PyObject *mpl_toolkitsmod = nullptr, *axis3dmod = nullptr;
  if (!mpl_toolkitsmod) {
    internal::_interpreter::get();

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
  PyObject* xarray = internal::get_2darray(x);
  PyObject* yarray = internal::get_2darray(y);
  PyObject* zarray = internal::get_2darray(z);

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
      PyObject_GetAttrString(internal::_interpreter::get().s_python_colormap, "coolwarm");

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
  PyObject* fig_exists = PyObject_CallObject(internal::_interpreter::get().s_python_function_fignum_exists, fig_args);
  if (!PyObject_IsTrue(fig_exists)) {
    fig = PyObject_CallObject(internal::_interpreter::get().s_python_function_figure,
                              internal::_interpreter::get().s_python_empty_tuple);
  } else {
    fig = PyObject_CallObject(internal::_interpreter::get().s_python_function_figure, fig_args);
  }
  Py_DECREF(fig_exists);
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject* gca_kwargs = PyDict_New();
  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject* gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject* axis = PyObject_Call(gca, internal::_interpreter::get().s_python_empty_tuple, gca_kwargs);

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

#endif

/** Keyword arguments for @ref plot_surface. */
struct PlotSurfaceKwargs {
  int rstride{1};                ///< Downsampling stride in each direction.
  int cstride{1};                ///< Downsampling stride in each direction.
  std::string cmap{"coolwarm"};  ///< A colormap for the surface.
  int fig_number{0};             ///< The figure number.
};

// template <class Matrix>
// py::array_t<typename Matrix::Scalar, py::array::f_style> to_numpy(const Eigen::DenseBase<Matrix>& matrix) {
//   if constexpr (Matrix::RowsAtCompileTime == 1 || Matrix::ColsAtCompileTime == 1)
//     return {std::vector<std::size_t>{static_cast<std::size_t>(matrix.size())}, matrix.derived().eval().data()};
//   return pybind11::array_t<typename Matrix::Scalar, py::array::f_style>{
//       std::vector<std::size_t>{static_cast<std::size_t>(matrix.rows()), static_cast<std::size_t>(matrix.cols())},
//       matrix.derived().eval().data()};
// }

/**
 * Create a surface plot.
 * By default, it will be colored in shades of a solid color,
 * but it also supports colormapping by supplying the cmap argument.
 * @tparam ContainerX container type for x
 * @tparam ContainerY container type for y
 * @tparam ContainerZ container type for z
 * @param x data value
 * @param y data value
 * @param z data value
 * @param kwargs additional keywords arguments
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface.html
 */
template <class ContainerX, class ContainerY, class ContainerZ>
bool plot_surface(const ContainerX& x, const ContainerY& y, const ContainerZ& z, const PlotSurfaceKwargs& kwargs = {}) {
  const py::object axis = internal::Interpreter::get().get_3d_axis(kwargs.fig_number);
  if (!axis) throw std::runtime_error("No axis");
  py::object plot_surface = axis.attr("plot_surface");
  if (!plot_surface) throw std::runtime_error("No surface");
  return static_cast<bool>(
      plot_surface(x, y, z, "cmap"_a = kwargs.cmap, "rstride"_a = kwargs.rstride, "cstride"_a = kwargs.cstride));
}

/** Keyword arguments for @ref plot_wireframe. */
struct PlotWireframeKwargs {
  int rstride{1};                ///< Downsampling stride in each direction.
  int cstride{1};                ///< Downsampling stride in each direction.
  std::string cmap{"coolwarm"};  ///< A colormap for the surface.
  int fig_number{0};             ///< The figure number.
};

/**
 *
 * @tparam ContainerX container type for x
 * @tparam ContainerY container type for y
 * @tparam ContainerZ container type for z
 * @param x data value
 * @param y data value
 * @param z data value
 * @param kwargs additional keywords arguments
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe.html
 */
template <class ContainerX, class ContainerY, class ContainerZ>
bool plot_wireframe(const ContainerX& x, const ContainerY& y, const ContainerZ& z,
                    const PlotWireframeKwargs& kwargs = {}) {
  const py::object axis = internal::Interpreter::get().get_3d_axis(kwargs.fig_number);
  if (!axis) throw std::runtime_error("No axis");
  py::object plot_wireframe = axis.attr("plot_wireframe");
  if (!plot_wireframe) throw std::runtime_error("No surface");
  return static_cast<bool>(
      plot_wireframe(x, y, z, "cmap"_a = kwargs.cmap, "rstride"_a = kwargs.rstride, "cstride"_a = kwargs.cstride));
}

/** Keyword arguments for @ref fill_between. */
struct FillBetweenKwargs {
  double alpha{1};                ///< The alpha blending value, between 0 (transparent) and 1 (opaque).
  bool interpolate{false};        ///< Whether to interpolate between the points.
  std::string edgecolor{"none"};  ///< The edge color of the filled area.
  std::string facecolor{"blue"};  ///< The face color of the filled area.
};

/**
 * Fill the area between two horizontal curves.
 * The curves are defined by the points (`x`, `y1`) and (`x`, `y2`).
 * This creates one or multiple polygons describing the filled area.
 * You may exclude some horizontal sections from filling using where.
 * By default, the edges connect the given points directly. Use step if the filling should be a step function,
 * i.e. constant in between `x`.
 * @tparam ContainerX container type for x
 * @tparam ContainerY1 container type for y1
 * @tparam ContainerY2 container type for y2
 * @param x x coordinates of the nodes defining the curves.
 * @param y1 y coordinates of the nodes defining the first curve.
 * If it contains a single value, it is replicated for every element of `x`
 * @param y2 y coordinates of the nodes defining the second curve.
 * If it contains a single value, it is replicated for every element of `x`
 * @param kwargs additional keywords arguments
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
 */
template <class ContainerX, class ContainerY1, class ContainerY2>
bool fill_between(const ContainerX& x, const ContainerY1& y1, const ContainerY2& y2,
                  const FillBetweenKwargs& kwargs = {}) {
  return static_cast<bool>(internal::Interpreter::get().fill_between()(
      x, y1, y2, "alpha"_a = kwargs.alpha, "interpolate"_a = kwargs.interpolate, "edgecolor"_a = kwargs.edgecolor,
      "facecolor"_a = kwargs.facecolor));
}

/** Keyword arguments for @ref savefig. */
struct SaveKwargs {
  bool transparent{false};  ///< If True, the Axes patches will all be transparent, otherwise as no effect and the
                            ///< color of the Axes and Figure patches are unchanged
  int dpi{-1};              ///< The resolution in dots per inch.
  std::optional<std::string> format{};  ///< The file format to use. By default it is inferred from the file name.
  double pad_inches{0.1};               ///< Amount of padding around the figure.
  std::string facecolor{"auto"};        ///< The facecolor of the figure.
  std::string edgecolor{"auto"};        ///< The edgecolor of the figure.
};

/**
 * Save the current figure as an image or vector graphic to a file.
 * @param filename the path of the file to save the figure to
 * @param kwargs additional keywords arguments
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
 */
inline bool savefig(const std::string& filename, const SaveKwargs& kwargs = {}) {
  py::object dpi, format;
  if (kwargs.dpi == -1) {
    dpi = py::str{"figure"};
  } else {
    dpi = py::cast(kwargs.dpi);
  }
  if (kwargs.format.has_value()) {
    format = py::str{kwargs.format.value()};
  } else {
    format = py::none();
  }
  return static_cast<bool>(internal::Interpreter::get().savefig()(
      filename, "transparent"_a = kwargs.transparent, "dpi"_a = dpi, "format"_a = format,
      "pad_inches"_a = kwargs.pad_inches, "facecolor"_a = kwargs.facecolor, "edgecolor"_a = kwargs.edgecolor));
}

/**
 * Set the x-axis view limits.
 * @param left left limit
 * @param right right limit
 * @return true if the function was successful
 * @return false if an error occurred
 */
bool xlim(double left, double right);
/**
 * Set the y-axis view limits.
 * @param bottom bottom limit
 * @param top top limit
 * @return true if the function was successful
 * @return false if an error occurred
 */
bool ylim(double bottom, double top);
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

#if 0
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
  if (y1.size() != 1 && x.size() != y1.size())
    throw lucid::exception::LucidPyException("missmatched array size x and y1");
  if (y2.size() != 1 && x.size() != y2.size())
    throw lucid::exception::LucidPyException("missmatched array size x and y2");

  internal::_interpreter::get();

  PyObject* _y1 = y1.size() > 1 ? internal::get_array(y1) : internal::PyObject_FromValue(y1[0]);
  PyObject* _y2 = y2.size() > 1 ? internal::get_array(y2) : internal::PyObject_FromValue(y2[0]);

  // construct positional args
  PyObject* args = internal::PyTuple_Create(internal::get_array(x), _y1, _y2);

  // construct keyword args
  PyObject* kwargs = PyDict_New();
  internal::PyDict_SetItems(kwargs, keywords);
  internal::PyDict_SetItem(kwargs, "alpha", PyFloat_FromDouble(alpha));

  PyObject* res = PyObject_Call(internal::_interpreter::get().s_python_function_fill_between, args, kwargs);

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
#endif

/**
 * Show the current figure.
 * @param block if true, the function will block until the figure is closed.
 * If false, it will return immediately and the figure will be shown in a separate window.
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
 */
inline void show(const bool block = true) { internal::Interpreter::get().show()("block"_a = block); }

/** Keyword arguments for @ref scatter. */
struct ScatterKwargs {
  double s = 1.0;          ///< The marker size in points**2.
  std::string c = "b";     ///< The marker color.
  std::string zdir = "z";  ///< The direction to use as z (‘x’, ‘y’ or ‘z’).
  int fig_number = -1;     ///< The figure number.
  char marker = 'o';       ///< The marker style.
};

/**
 * A scatter plot of `y` vs `x` with varying marker size and/or color.
 * @tparam ContainerX x container type
 * @tparam ContainerY y container type
 * @param x data points for the horizontal axis
 * @param y data points for the vertical axis
 * @param kwargs additional keywords arguments
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
 */
template <class ContainerX, class ContainerY>
bool scatter(const ContainerX& x, const ContainerY& y, const ScatterKwargs& kwargs = {}) {
  return static_cast<bool>(
      internal::Interpreter::get().scatter()(x, y, "s"_a = kwargs.s, "c"_a = kwargs.c, "marker"_a = kwargs.marker));
}
/**
 * A scatter plot of `y` vs `x` vs `z` with varying marker size and/or color.
 * @tparam ContainerX x container type
 * @tparam ContainerY y container type
 * @tparam ContainerZ y container type
 * @param x data points for the horizontal axis
 * @param y data points for the vertical axis
 * @param z data points for the depth axis
 * @param kwargs additional keywords arguments
 * @return true if the function was successful
 * @return false if an error occurred
 * @see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
 */
template <class ContainerX, class ContainerY, class ContainerZ>
bool scatter(const ContainerX& x, const ContainerY& y, const ContainerZ& z, const ScatterKwargs& kwargs = {}) {
  const py::object axis = internal::Interpreter::get().get_3d_axis(kwargs.fig_number);
  if (!axis) throw std::runtime_error("No axis");
  py::function scatter = axis.attr("scatter");
  if (!scatter) throw std::runtime_error("No 3D line plot");
  return static_cast<bool>(
      scatter(x, y, z, "zdir"_a = kwargs.zdir, "s"_a = kwargs.s, "c"_a = kwargs.c, "marker"_a = kwargs.marker));
}

}  // namespace lucid::plt
