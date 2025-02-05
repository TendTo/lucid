"""This module defines rules for integrating C++ and Python in Bazel.
It enables creating C++ binaries and libraries that depend on the python interpreter and python libraries.
Inspired by JBPennington @ https://github.com/JBPennington/pybind_cpp_w_python_example
"""

load("//tools:rules_cc.bzl", "lucid_cc_binary", "lucid_cc_library", "lucid_cc_test")

def _cc_py_runtime_impl(ctx):
    py3_runtime = ctx.toolchains[ctx.attr._python_toolchain].py3_runtime
    imports = []
    for dep in ctx.attr.deps:
        imports.append(dep[PyInfo].imports)
    python_path = ""
    for path in depset(transitive = imports).to_list():
        python_path += "../" + path + ":"

    py3_runfiles = ctx.runfiles(files = py3_runtime.files.to_list())
    runfiles = [py3_runfiles]
    for dep in ctx.attr.deps:
        dep_runfiles = ctx.runfiles(files = dep[PyInfo].transitive_sources.to_list())
        runfiles.append(dep_runfiles)
        runfiles.append(dep[DefaultInfo].default_runfiles)

    return [
        DefaultInfo(runfiles = ctx.runfiles().merge_all(runfiles)),
        platform_common.TemplateVariableInfo({
            "PYTHON3": "../" + str(py3_runtime.interpreter.path.removeprefix("external/")),
            "PYTHONPATH": python_path,
            "PYTHONHOME": "../" + str(py3_runtime.interpreter.dirname.rstrip("bin").removeprefix("external/")),
        }),
    ]

_cc_py_runtime = rule(
    implementation = _cc_py_runtime_impl,
    attrs = {
        "deps": attr.label_list(providers = [PyInfo]),
        "_python_toolchain": attr.string(default = "@rules_python//python:toolchain_type"),
    },
    toolchains = [
        "@rules_python//python:toolchain_type",
    ],
)

def lucid_cc_py_test(name, py_deps = [], data = [], env = {}, toolchains = [], **kwargs):
    py_runtime_target = name + "_py_runtime"
    _cc_py_runtime(
        name = py_runtime_target,
        deps = py_deps,
    )

    lucid_cc_test(
        name = name,
        data = data + [":" + py_runtime_target],
        env = env | {"__PYVENV_LAUNCHER__": "$(PYTHON3)", "PYTHONPATH": "$(PYTHONPATH)", "PYTHONHOME": "$(PYTHONHOME)"},
        toolchains = toolchains + [":" + py_runtime_target],
        **kwargs
    )

def lucid_cc_py_binary(name, py_deps = [], data = [], deps = [], env = {}, toolchains = [], **kwargs):
    py_runtime_target = name + "_py_runtime"
    _cc_py_runtime(
        name = py_runtime_target,
        deps = py_deps,
    )

    lucid_cc_binary(
        name = name,
        data = data + [":" + py_runtime_target],
        env = env | {"__PYVENV_LAUNCHER__": "$(PYTHON3)", "PYTHONPATH": "$(PYTHONPATH)", "PYTHONHOME": "$(PYTHONHOME)"},
        toolchains = toolchains + [":" + py_runtime_target],
        deps = deps,
        **kwargs
    )

def lucid_cc_py_library(name, py_deps = [], data = [], deps = [], local_defines = [], toolchains = [], **kwargs):
    py_runtime_target = name + "_py_runtime"

    lucid_cc_library(
        name = name,
        data = data + [":" + py_runtime_target],
        local_defines = local_defines + [
            'CPP_PYVENV_LAUNCHER="$(PYTHON3)"',
            'CPP_PYTHON_PATH="$(PYTHONPATH)"',
            'CPP_PYTHON_HOME="$(PYTHONHOME)"',
        ],
        toolchains = toolchains + [":" + py_runtime_target],
        deps = deps,
        **kwargs
    )
