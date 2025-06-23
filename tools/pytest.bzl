"""Macro with some preconfigurations for testing with pytest."""

load("@pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, deps = [], args = [], **kwargs):
    """Call pytest from a py_test rule, taking care of the common arguments and dependencies.

    Args:
        name: The name of the rule.
        srcs: The source files to test.
        deps: The dependencies of the rule.
        args: The arguments to pass to pytest.
        **kwargs: Additional arguments to pass to py_test.
    """
    py_test(
        name = name,
        srcs = ["//tools:pytest_main.py"] + srcs,
        main = "//tools:pytest_main.py",
        args = ["--capture=no"] + args + ["$(location :%s)" % x for x in srcs],
        python_version = "PY3",
        srcs_version = "PY3",
        deps = deps + [requirement("pytest")],
        **kwargs
    )

def pylucid_py_test(name, srcs = None, deps = [], args = [], data = [], size = "small", tags = [], **kwargs):
    """Call pytest from a py_test rule, taking care of the common arguments and dependencies.

    By default, sets size="small" because that indicates a unit test.
    If a list of srcs is not provided, it will be inferred from the name, by capitalizing each _-separated word and appending .py.
    For example, pylucid_py_test(name = "test_foo_bar") will look for TestFooBar.py.

    Args:
        name: The name of the rule.
        srcs: The source files to test.
        deps: The dependencies of the rule.
        args: The arguments to pass to pytest.
        data: The data dependencies of the rule.
        size: The size of the test.
        tags: The tags to apply to the test. Ease of use for filtering tests.
        **kwargs: Additional arguments to pass to py_test.
    """
    if srcs == None:
        srcs = ["".join([word.capitalize() for word in name.split("_")]) + ".py"]
    pytest_test(
        name = name,
        srcs = srcs,
        args = args,
        deps = deps + [
            "//bindings/pylucid:pylucid_lib",
            requirement("numpy"),
            requirement("sympy"),
            requirement("pyparsing"),
        ],
        data = data,
        size = size,
        tags = tags + ["pylucid"],
        **kwargs
    )

def pylucid_gurobi_py_test(name, srcs = None, deps = [], args = [], data = [], size = "small", tags = [], env = {}, **kwargs):
    """Call pytest from a py_test rule, taking care of the common arguments and dependencies required for Gurobi.

    By default, sets size="small" because that indicates a unit test.
    If a list of srcs is not provided, it will be inferred from the name, by capitalizing each _-separated word and appending .py.
    For example, pylucid_py_test(name = "test_foo_bar") will look for TestFooBar.py.
    Additionally, a the gurobi licence, stored in a "guribi.lic" file is expected to be found in the same folder.

    Args:
        name: The name of the rule.
        srcs: The source files to test.
        deps: The dependencies of the rule.
        args: The arguments to pass to pytest.
        data: The data dependencies of the rule.
        size: The size of the test.
        tags: The tags to apply to the test. Ease of use for filtering tests.
        env: Environment variables
        **kwargs: Additional arguments to pass to py_test.
    """
    pylucid_py_test(
        name = name,
        srcs = srcs,
        args = args,
        deps = deps,
        size = size,
        tags = tags,
        data = data + select({
            "//tools:gurobi_build": ["gurobi.lic"],
            "//conditions:default": [],
        }),
        env = env | select({
            "//tools:gurobi_build": {"GRB_LICENSE_FILE": "$(location gurobi.lic)"},
            "//conditions:default": {},
        }),
        **kwargs
    )
