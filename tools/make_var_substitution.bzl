"""Provides a set of variables to the template engine."""

load("//tools:variables.bzl", "LUCID_AUTHOR", "LUCID_AUTHOR_EMAIL", "LUCID_DESCRIPTION", "LUCID_HOMEPAGE", "LUCID_LICENSE", "LUCID_NAME", "LUCID_SOURCE", "LUCID_TRACKER", "LUCID_VERSION")

def _make_var_substitution_impl(ctx):
    vars = dict(ctx.attr.variables)

    # Python
    py_runtime = ctx.toolchains[ctx.attr._python_toolchain].py3_runtime
    major = py_runtime.interpreter_version_info.major
    minor = py_runtime.interpreter_version_info.minor
    implementation = py_runtime.implementation_name
    if implementation == "cpython":
        tag = "cp" + str(major) + str(minor)
        vars["PYTHON_ABI_TAG"] = tag
        vars["PYTHON_TAG"] = tag
    else:
        fail("This rule only supports CPython.")

    # lucid
    vars["LUCID_NAME"] = LUCID_NAME
    vars["LUCID_VERSION"] = LUCID_VERSION
    vars["LUCID_AUTHOR"] = LUCID_AUTHOR
    vars["LUCID_AUTHOR_EMAIL"] = LUCID_AUTHOR_EMAIL
    vars["LUCID_DESCRIPTION"] = LUCID_DESCRIPTION
    vars["LUCID_HOMEPAGE"] = LUCID_HOMEPAGE
    vars["LUCID_LICENSE"] = LUCID_LICENSE
    vars["LUCID_SOURCE"] = LUCID_SOURCE
    vars["LUCID_TRACKER"] = LUCID_TRACKER
    vars["GUROBI_HOME"] = ctx.configuration.default_shell_env.get("GUROBI_HOME", "/opt/gurobi")

    return [platform_common.TemplateVariableInfo(vars)]

make_var_substitution = rule(
    implementation = _make_var_substitution_impl,
    attrs = {
        "variables": attr.string_dict(),
        "_python_toolchain": attr.string(default = "@rules_python//python:toolchain_type"),
    },
    doc = """Provides a set of variables to the template engine.
Variables are passed as a dictionary of strings.
The keys are the variable names, and the values are the variable values.

It also comes with a set of default variables that are always available:
- LUCID_NAME: The name of the lucid library.
- LUCID_VERSION: The version of the lucid library.
- LUCID_AUTHOR: The author of the lucid library.
- LUCID_AUTHOR_EMAIL: The author email of the lucid library.
- LUCID_DESCRIPTION: The description of the lucid library.
- LUCID_HOMEPAGE: The homepage of the lucid library.
- LUCID_LICENSE: The license of the lucid library.
- PYTHON_ABI_TAG: The Python ABI tag (e.g., cp36, cp311).
- PYTHON_TAG: The Python tag (e.g., cp36, cp311).

Example:
```python
make_var_substitution(
    variables = {
        "MY_VARIABLE": "my_value",
    },
)
```

This will make the variable `MY_VARIABLE` available to the template engine.
""",
    toolchains = [
        "@rules_python//python:toolchain_type",
    ],
)
