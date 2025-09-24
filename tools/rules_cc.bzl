"""Provide a centralised place to define rules for building C++ code.
This makes it easy to change the build configuration for all C++ rules in the project at once.
Inspired by Drake's drake.bzl file https://github.com/RobotLocomotion/drake/blob/master/tools/drake.bzl.
"""

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

# The CXX_FLAGS will be enabled for all C++ rules in the project
# building with any linux compiler.
CXX_FLAGS = [
    "-std=c++20",
    "-Wall",
    "-Wattributes",
    "-Wdeprecated",
    "-Wdeprecated-declarations",
    "-Wextra",
    "-Wignored-qualifiers",
    "-Wold-style-cast",
    "-Woverloaded-virtual",
    "-Wpedantic",
    "-Wshadow",
    "-Werror",
]

# The CLANG_FLAGS will be enabled for all C++ rules in the project when
# building with clang.
CLANG_FLAGS = CXX_FLAGS + [
    "-Wabsolute-value",
    "-Winconsistent-missing-override",
    "-Wnon-virtual-dtor",
    "-Wreturn-stack-address",
    "-Wsign-compare",
]

# The GCC_FLAGS will be enabled for all C++ rules in the project when
# building with gcc.
GCC_FLAGS = CXX_FLAGS + [
    "-Wlogical-op",
    "-Wnon-virtual-dtor",
    "-Wreturn-local-addr",
    "-Wunused-but-set-parameter",
]

# The MSVC_CL_FLAGS will be enabled for all C++ rules in the project when
# building with MSVC.
MSVC_CL_FLAGS = [
    "/std:c++20",
    "/utf-8",
    "-W4",
    "-WX",
    # "-wd4068",  # unknown pragma
    "-wd4673",  # the following types will not be considered at the catch site
    "-wd4670",  # the base class is inaccessible
    "-wd4068",  # unknown pragma 'GCC'
    "-wd4702",  # unreachable code
    "-external:anglebrackets",  # Treat angle brackets headers as external headers
    "-external:W0",  # Disable warnings for external headers
]

# The CLANG_CL_FLAGS will be enabled for all C++ rules in the project when
# building with clang-cl.
CLANG_CL_FLAGS = [
    "-Wabsolute-value",
    "-Winconsistent-missing-override",
    "-Wnon-virtual-dtor",
    "-Wreturn-stack-address",
    "-Wsign-compare",
]

# The test configuration flags for each compiler.
GCC_TEST_FLAGS = []
CLANG_TEST_FLAGS = []
MSVC_CL_TEST_FLAGS = []
CLANG_CL_TEST_FLAGS = []

# Default defines for all C++ rules in the project.
LUCID_DEFINES = ["LUCID_INCLUDE_FMT"]

def _get_copts(rule_copts, cc_test = False):
    """Alter the provided rule specific copts, adding the platform-specific ones.

    When cc_test is True, the corresponding test flags will be added.
    It should only be set on cc_test rules or rules that are boil down to cc_test rules.

    Args:
        rule_copts: The copts passed to the rule.
        cc_test: Whether the rule is a cc_test rule.

    Returns:
        A list of copts.
    """
    return rule_copts + select({
        "//tools:gcc_build": GCC_FLAGS + (GCC_TEST_FLAGS if cc_test else []),
        "//tools:clang_build": CLANG_FLAGS + (CLANG_CL_TEST_FLAGS if cc_test else []),
        "//tools:msvc_cl_build": MSVC_CL_FLAGS + (MSVC_CL_TEST_FLAGS if cc_test else []),
        "//tools:clang_cl_build": CLANG_CL_FLAGS + (CLANG_CL_TEST_FLAGS if cc_test else []),
        "//conditions:default": CXX_FLAGS,
    }) + select({
        "//tools:gcc_omp_build": ["-fopenmp"],
        "//tools:clang_omp_build": [],  # ["-fopenmp"],
        "//tools:msvc_cl_omp_build": ["/openmp"],
        "//tools:clang_cl_omp_build": [],  # ["-fopenmp"],
        "//conditions:default": [],
    })

def _get_linkopts(rule_linkopts, cc_test = False):
    """Alter the provided rule specific linkopts, adding the platform-specific ones.

    When cc_test is True, the corresponding test flags will be added.
    It should only be set on cc_test rules or rules that are boil down to cc_test rules.

    Args:
        rule_linkopts: The linkopts passed to the rule.
        cc_test: Whether the rule is a cc_test rule.

    Returns:
        A list of linkopts.
    """
    return rule_linkopts + select({
        "//tools:gcc_omp_build": ["-lgomp"],
        "//conditions:default": [],
    })

def _get_defines(rule_defines):
    """Alter the provided rule specific defines, adding the platform-specific ones.

    Args:
        rule_defines: The defines passed to the rule.

    Returns:
        A list of defines.
    """
    return rule_defines + LUCID_DEFINES + select({
        "//tools:debug_build": [],
        "//tools:release_build": ["NDEBUG"],
        "//conditions:default": [],
    }) + select({
        "//tools:benchmark_build": ["NCHECK", "NCONVERT", "NLOG"],
        "//conditions:default": [],
    }) + select({
        "//tools:python_build": ["LUCID_PYTHON_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:matplotlib_build": ["LUCID_MATPLOTLIB_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:gurobi_build": ["LUCID_GUROBI_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:hexaly_build": ["LUCID_HEXALY_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:alglib_build": ["LUCID_ALGLIB_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:highs_build": ["LUCID_HIGHS_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:soplex_build": ["LUCID_SOPLEX_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:psocpp_build": ["LUCID_PSOCPP_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:verbose_eigen_build": ["LUCID_VERBOSE_EIGEN_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:gui_build": ["LUCID_GUI_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:omp_build": ["LUCID_OMP_BUILD"],
        "//conditions:default": [],
    }) + select({
        "//tools:cuda_build": ["LUCID_CUDA_BUILD"],
        "//conditions:default": [],
    })

def _get_static(rule_linkstatic):
    """Alter the provided linkstatic, by considering the platform-specific one.

    The files are linked statically by default.

    Args:
        rule_linkstatic: The linkstatic passed to the rule.

    Returns:
        The linkstatic value to use
    """
    if rule_linkstatic != None:
        return rule_linkstatic
    return select({
        "//tools:static_build": True,
        "//tools:dynamic_build": False,
        "//conditions:default": True,
    })

def _get_features(rule_features):
    """Alter the provided features, adding the platform-specific ones.

    Args:
        rule_features: The features passed to the rule.

    Returns:
        A list of features.
    """
    return rule_features + select({
        "//tools:dynamic_build": [],
        "//tools:static_build": ["fully_static_link"],
        "//conditions:default": [],
    })

def lucid_cc_library(
        name,
        hdrs = None,
        srcs = None,
        deps = None,
        copts = [],
        linkopts = [],
        linkstatic = None,
        defines = [],
        implementation_deps = [],
        **kwargs):
    """Creates a rule to declare a C++ library.

    Args:
        name: The name of the library.
        hdrs: A list of header files to add. Will be inherited by dependents.
        srcs: A list of source files to compile.
        deps: A list of dependencies. Will be inherited by dependents.
        implementation_deps: A list of dependencies that are only needed for this target.
        copts: A list of compiler options.
        linkopts: A list of linker options.
        linkstatic: Whether to link statically.
        defines: A list of compiler defines used when compiling this target and its dependents.
        **kwargs: Additional arguments to pass to cc_library.
    """
    cc_library(
        name = name,
        hdrs = hdrs,
        srcs = srcs,
        deps = deps,
        implementation_deps = implementation_deps,
        copts = _get_copts(copts),
        linkopts = _get_linkopts(linkopts),
        linkstatic = _get_static(linkstatic),
        defines = _get_defines(defines),
        **kwargs
    )

def lucid_cc_binary(
        name,
        srcs = None,
        deps = None,
        copts = [],
        linkopts = [],
        linkstatic = None,
        defines = [],
        features = [],
        **kwargs):
    """Creates a rule to declare a C++ binary.

    Args:
        name: The name of the binary.
        srcs: A list of source files to compile.
        deps: A list of dependencies.
        copts: A list of compiler options.
        linkstatic: Whether to link statically.
        linkopts: A list of linker options.
        defines: A list of compiler defines used when compiling this target.
        features: A list of features to add to the binary.
        **kwargs: Additional arguments to pass to cc_binary.
    """
    cc_binary(
        name = name,
        srcs = srcs,
        deps = deps,
        copts = _get_copts(copts),
        linkopts = _get_linkopts(linkopts),
        linkstatic = _get_static(linkstatic),
        defines = _get_defines(defines),
        features = _get_features(features),
        **kwargs
    )

def lucid_cc_test(
        name,
        srcs = None,
        data = [],
        deps = None,
        copts = [],
        linkopts = [],
        tags = [],
        defines = [],
        **kwargs):
    """Creates a rule to declare a C++ unit test.

    Note that for almost all cases, lucid_cc_googletest should be used instead of this rule.

    By default, sets size="small" because that indicates a unit test.
    If a list of srcs is not provided, it will be inferred from the name, by capitalizing each _-separated word and appending .cpp.
    For example, lucid_cc_test(name = "test_foo_bar") will look for TestFooBar.cpp.
    Furthermore, a tag will be added for the test, based on the name, by converting the name to lowercase and removing the "test_" prefix.

    Args:
        name: The name of the test.
        srcs: A list of source files to compile.
        data: A list of data files to include in the test. Can be used to provide input files.
        deps: A list of dependencies.
        copts: A list of compiler options.
        linkopts: A list of linker options.
        tags: A list of tags to add to the test. Allows for test filtering.
        defines: A list of compiler defines used when compiling this target.
        **kwargs: Additional arguments to pass to cc_test.
    """
    if srcs == None:
        srcs = ["".join([word.capitalize() for word in name.split("_")]) + ".cpp"]
    if deps == None:
        deps = []
    if data:
        deps.append("@rules_cc//cc/runfiles")
    cc_test(
        name = name,
        srcs = srcs,
        data = data,
        deps = deps,
        copts = _get_copts(copts, cc_test = True),
        linkopts = _get_linkopts(linkopts, cc_test = True),
        linkstatic = True,
        tags = tags + ["lucid", "".join([word.lower() for word in name.split("_")][1:])],
        defines = _get_defines(defines),
        **kwargs
    )

def lucid_cc_googletest(
        name,
        srcs = None,
        deps = None,
        size = "small",
        tags = [],
        use_default_main = True,
        defines = [],
        **kwargs):
    """Creates a rule to declare a C++ unit test using googletest.

    Always adds a deps= entry for googletest main
    (@googletest//:gtest_main).

    By default, it uses size="small" because that indicates a unit test.
    By default, it uses use_default_main=True to use GTest's main, via @googletest//:gtest_main.
    If use_default_main is False, it will depend on @googletest//:gtest instead.
    If a list of srcs is not provided, it will be inferred from the name, by capitalizing each _-separated word and appending .cpp.
    For example, lucid_cc_test(name = "test_foo_bar") will look for TestFooBar.cpp.
    Furthermore, a tag will be added for the test, based on the name, by converting the name to lowercase and removing the "test_" prefix.

    Args:
        name: The name of the test.
        srcs: A list of source files to compile.
        deps: A list of dependencies.
        size: The size of the test.
        tags: A list of tags to add to the test. Allows for test filtering.
        use_default_main: Whether to use googletest's main.
        defines: A list of compiler defines used when compiling this target.
        **kwargs: Additional arguments to pass to lucid_cc_test.
    """
    if deps == None:
        deps = []
    if type(deps) == "select":
        if use_default_main:
            deps += select({"//conditions:default": ["@googletest//:gtest_main"]})
        else:
            deps += select({"//conditions:default": ["@googletest//:gtest"]})
    elif use_default_main:
        deps.append("@googletest//:gtest_main")
    else:
        deps.append("@googletest//:gtest")
    lucid_cc_test(
        name = name,
        srcs = srcs,
        deps = deps,
        size = size,
        tags = tags + ["googletest"],
        defines = _get_defines(defines),
        **kwargs
    )

def lucid_srcs(name, srcs = None, hdrs = None, deps = [], subfolder = "", visibility = ["//visibility:public"]):
    """Returns three different lists of source files based on the name.

    Args:
        name: The name of the target. If the name is "srcs", the default "srcs", "hdrs", and "all_srcs" will be used.
            Otherwise, "srcs_" + name, "hdrs_" + name, and "all_srcs_" + name will be used.
        srcs: A list of source files include in the filegroup. If None, common c++ source files extensions will be used.
        hdrs: A list of header files to include in the filegroup. If None, common c++ header files extensions will be used.
        deps: A list of dependencies. Used for the all_srcs filegroup.
        subfolder: The subfolder to use for the tarball.
        visibility: A list of visibility labels to apply to the filegroups.
    """
    if name == "srcs":
        srcs_name, hdrs_name, all_srcs_name = "srcs", "hdrs", "all_srcs"
    else:
        srcs_name, hdrs_name, all_srcs_name = "srcs_%s" % name, "hdrs_%s" % name, "all_srcs_%s" % name
    if srcs == None:
        srcs = native.glob(["*.cpp", "*.cc", "*.cxx", "*.c"], allow_empty = True)
    if hdrs == None:
        hdrs = native.glob(["*.h", "*.hpp"], allow_empty = True)
    native.filegroup(
        name = srcs_name,
        srcs = srcs + hdrs,
        tags = ["no-cpplint"],
        visibility = visibility,
    )
    native.filegroup(
        name = hdrs_name,
        srcs = hdrs,
        tags = ["no-cpplint"],
        visibility = visibility,
    )
    native.filegroup(
        name = all_srcs_name,
        srcs = srcs + hdrs + deps,
        visibility = visibility,
    )

def lucid_hdrs_tar(name, hdrs = None, deps = [], subfolder = "", visibility = ["//visibility:public"]):
    """Returns three different lists of source files based on the name.

    Args:
        name: The name of the target. If the name is "srcs", the default "srcs", "hdrs", and "all_srcs" will be used.
            Otherwise, "srcs_" + name, "hdrs_" + name, and "all_srcs_" + name will be used.
        hdrs: A list of header files to include in the filegroup. If None, common c++ header files extensions will be used.
        subfolder: The subfolder to use for the tarball.
        deps: A list of dependencies. Used for the all_srcs filegroup.
        visibility: A list of visibility labels to apply to the filegroups.
    """
    if hdrs == None:
        hdrs = native.glob(["*.h", "*.hpp"], allow_empty = True)
    pkg_tar(
        name = name,
        srcs = hdrs,
        extension = "tar.gz",
        package_dir = subfolder,
        deps = deps,
        visibility = visibility,
    )

def lucid_pybind_library(
        name,
        srcs = None,
        deps = [],
        copts = [],
        linkopts = [],
        linkstatic = None,
        defines = [],
        features = [],
        **kwargs):
    """Creates a rule to declare a pybind11 library.

    Args:
        name: The name of the library.
        srcs: A list of source files to compile.
        deps: A list of dependencies.
        copts: A list of compiler options.
        linkopts: A list of linker options.
        linkstatic: Whether to link statically.
        defines: A list of defines to add to the library.
        features: A list of features to add to the library.
        **kwargs: Additional arguments to pass to pybind_library.
    """
    pybind_library(
        name = name,
        srcs = srcs,
        deps = deps,
        copts = _get_copts(copts),
        linkopts = _get_linkopts(linkopts),
        linkstatic = _get_static(linkstatic),
        defines = _get_defines(defines),
        features = features,  # Do not use _get_features here, as it will add fully_static_link and that is not supported
        **kwargs
    )

def lucid_pybind_extension(
        name,
        srcs,
        deps = [],
        copts = [],
        linkopts = [],
        linkstatic = None,
        defines = [],
        features = [],
        **kwargs):
    """Creates a rule to declare a pybind11 extension.

    Args:
        name: The name of the extension.
        srcs: A list of source files to compile.
        deps: A list of dependencies.
        copts: A list of compiler options.
        linkopts: A list of linker options.
        linkstatic: Whether to link statically.
        defines: A list of defines to add to the extension.
        features: A list of features to add to the extension.
        **kwargs: Additional arguments to pass to pybind_extension.
    """
    pybind_extension(
        name = name,
        srcs = srcs,
        deps = deps,
        copts = _get_copts(copts),
        linkopts = _get_linkopts(linkopts),
        linkstatic = _get_static(linkstatic),
        defines = _get_defines(defines),
        features = features,  # Do not use _get_features here, as it will add fully_static_link and that is not supported
        **kwargs
    )
