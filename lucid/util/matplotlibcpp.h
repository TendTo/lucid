/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * A C++ wrapper for python's matplotlib.
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
#include <stdexcept>
#include <string>

#ifndef WITHOUT_EIGEN
#include "lucid/lib/eigen.h"
#endif

namespace py = pybind11;
using namespace py::literals;

namespace matplotlibcpp {
namespace detail {

static std::string backend_;

class __attribute__((visibility("hidden"))) _interpreter {
 public:
  static _interpreter& get() { return interkeeper(false); }
  static _interpreter& kill() { return interkeeper(true); }
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

  py::function savefig() const { return pylabmod_.attr("savefig"); }

  // Attributes
  py::str colorbar() const { return pyplot_.attr("colorbar"); }
  auto rcParams() const { return pyplot_.attr("rcParams"); }
  py::function spy() const { return pyplot_.attr("spy"); }

  py::object get_3d_axis(const long fig_number) {
    const py::object fig = fig_number >= 0 ? figure()(fig_number) : gcf()();
    if (fig.is_none()) throw std::runtime_error("Call to figure() failed.");
    const py::list fig_axes = fig.attr("axes");
    if (fig_axes.empty()) return axes()("projection"_a = "3d");  // There is no axes, create one
    if (fig_axes[0].attr("name").cast<std::string>() == "3d") return fig_axes[0];
    throw std::runtime_error("The figure already contains an axis that is not 3D.");
  }

 private:
  static _interpreter& interkeeper(const bool should_kill) {
    static _interpreter ctx;
    if (should_kill) ctx.~_interpreter();
    return ctx;
  }

  _interpreter()
      : matplotlib_(py::module_::import("matplotlib")),
        pyplot_{py::module_::import("matplotlib.pyplot")},
        pylabmod_{py::module_::import("pylab")} {
    matplotlib_.attr("use")(backend_);
  }

  py::scoped_interpreter guard_;
  py::module_ matplotlib_;
  py::module_ pyplot_;
  py::module_ pylabmod_;
};

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
inline void backend(const std::string& name) { detail::backend_ = name; }

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
  return static_cast<bool>(detail::_interpreter::get().annotate()(text, "xy"_a = std::array<double, 2>{x, y}));
}

/**
 * Create a new figure or select an existing figure.
 * @param fig_number number of the figure to select. If negative, a new figure is created.
 * @return the figure number
 */
inline long figure(const long fig_number = -1) {
  const py::object res =
      fig_number >= 0 ? detail::_interpreter::get().figure()(fig_number) : detail::_interpreter::get().figure()();
  if (!res) throw std::runtime_error("Call to figure() failed.");
  return res.attr("number").cast<long>();
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
  return static_cast<bool>(detail::_interpreter::get().plot()(x, y, kwargs.fmt, "alpha"_a = kwargs.alpha,
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

#endif

/** Keyword arguments for @ref plot_surface. */
struct PlotSurfaceKwargs {
  int rstride{1};                ///< Downsampling stride in each direction.
  int cstride{1};                ///< Downsampling stride in each direction.
  std::string cmap{"coolwarm"};  ///< A colormap for the surface.
  int fig_number{0};             ///< The figure number.
};

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
  const py::object axis = detail::_interpreter::get().get_3d_axis(kwargs.fig_number);
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
  const py::object axis = detail::_interpreter::get().get_3d_axis(kwargs.fig_number);
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
  return static_cast<bool>(detail::_interpreter::get().fill_between()(
      x, y1, y2, "alpha"_a = kwargs.alpha, "interpolate"_a = kwargs.interpolate, "edgecolor"_a = kwargs.edgecolor,
      "facecolor"_a = kwargs.facecolor));
}

/** Keyword arguments for @ref save. */
struct SaveKwargs {
  bool transparent{false};  ///< If True, the Axes patches will all be transparent, otherwise as no effect and the
                            ///< color of the Axes and Figure patches are unchanged
  int dpi{-1};              ///< The resolution in dots per inch.
  std::optional<std::string> format{};  ///< The file format to use. By default it is inferred from the file name.
  float pad_inches{0.1};                ///< Amount of padding around the figure.
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
  return static_cast<bool>(detail::_interpreter::get().savefig()(
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

#if 0
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
  if (y1.size() != 1 && x.size() != y1.size())
    throw lucid::exception::LucidPyException("missmatched array size x and y1");
  if (y2.size() != 1 && x.size() != y2.size())
    throw lucid::exception::LucidPyException("missmatched array size x and y2");

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

#endif

inline void show(const bool block = true) { detail::_interpreter::get().show()("block"_a = block); }

/** Keyword arguments for @ref scatter. */
struct ScatterKwargs {
  double s = 1.0;          ///< The marker size in points**2.
  std::string c = "b";     ///< The marker color.
  std::string zdir = "z";  ///< The direction to use as z (‘x’, ‘y’ or ‘z’).
  long fig_number = -1;    ///< The figure number.
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
      detail::_interpreter::get().scatter()(x, y, "s"_a = kwargs.s, "c"_a = kwargs.c, "marker"_a = kwargs.marker));
}
template <class ContainerX, class ContainerY, class ContainerZ>
bool scatter(const ContainerX& x, const ContainerY& y, const ContainerZ& z, const ScatterKwargs& kwargs = {}) {
  const py::object axis = detail::_interpreter::get().get_3d_axis(kwargs.fig_number);
  if (!axis) throw std::runtime_error("No axis");
  py::function scatter = axis.attr("scatter");
  if (!scatter) throw std::runtime_error("No 3D line plot");
  return static_cast<bool>(
      scatter(x, y, z, "zdir"_a = kwargs.zdir, "s"_a = kwargs.s, "c"_a = kwargs.c, "marker"_a = kwargs.marker));
  ;
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
// template <class NumericX, class NumericY, int SizeX, int SizeY>
// bool scatter(const Eigen::Vector<NumericX, SizeX>& x, const Eigen::Vector<NumericY, SizeY>& y, const double s = 1.0,
//              const std::map<std::string, std::string>& keywords = {}) {
//   return scatter(std::span<const NumericX>{x.data(), static_cast<std::size_t>(x.size())},
//                  std::span<const NumericY>{y.data(), static_cast<std::size_t>(y.size())}, s, keywords);
// }
#endif  // WITHOUT_EIGEN

#if 0

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
  if (x.size() != y.size() || y.size() != z.size())
    throw lucid::exception::LucidPyException("x, y, and z must have the same size!");
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
template <class DerivedX, class DerivedY>
bool plot(const Eigen::DenseBase<DerivedX>& x, const Eigen::DenseBase<DerivedY>& y, const std::string& s = "") {
  // Create a span from the Eigen data
  const std::span<const typename DerivedX::Scalar> x_span{x.derived().eval().data(),
                                                          static_cast<std::size_t>(x.size())};
  const std::span<const typename DerivedY::Scalar> y_span{y.derived().eval().data(),
                                                          static_cast<std::size_t>(y.size())};
  return plot(x_span, y_span, s);
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

#endif

}  // end namespace matplotlibcpp
