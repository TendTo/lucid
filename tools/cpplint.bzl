"""Used to lint all C++ files in the targets of a BUILD file"""

load("@rules_python//python:defs.bzl", "py_test")

_SOURCE_EXTENSIONS = [
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".C",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
]

# Do not lint generated protocol buffer files.
_IGNORE_EXTENSIONS = [
    ".pb.h",
    ".pb.cc",
]

# The cpplint.py command-line argument so it doesn't skip our files!
_EXTENSIONS_ARGS = ["--extensions=" + ",".join(
    [ext[1:] for ext in _SOURCE_EXTENSIONS],
)]

def cpplint(name = "cpplint", exclude_srcs = [], data = [], extra_args = []):
    """Add a cpplint target for every c++ source file in each target in the BUILD file so far.

    For every rule in the BUILD file so far, adds a test rule that runs
    cpplint over the C++ sources listed in that rule.  Thus, BUILD file authors
    should call this function at the *end* of every C++-related BUILD file.

    By default, only the CPPLINT.cfg from the project root and the current
    directory are used.  Additional configs can be passed in as data labels.

    Args:
        name: name of the cpplint target. Defaults to "cpplint".
        exclude_srcs: list of source files to exclude from linting.
        data: additional data to include in the py_test() rule.
        extra_args: additional arguments to pass to cpplint.py.
    """
    native.filegroup(
        name = "cpplint_files",
        srcs = native.glob(
            [
                "**/*" + ext
                for ext in _SOURCE_EXTENSIONS
            ],
            exclude = [
                "**/*" + ext
                for ext in _IGNORE_EXTENSIONS
            ] + exclude_srcs,
            exclude_directories = 1,
            allow_empty = True,
        ),
    )

    # Google cpplint.
    py_test(
        name = name,
        srcs = ["@cpplint"],
        data = data + [":cpplint_files", "//:CPPLINT.cfg"],
        args = _EXTENSIONS_ARGS + ["$(locations :cpplint_files)"] + extra_args,
        main = "@cpplint//:cpplint.py",
        size = "small",
        tags = ["cpplint"],
    )
