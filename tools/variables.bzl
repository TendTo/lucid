LUCID_NAME = "lucid"
LUCID_VERSION = "0.0.1"
LUCID_AUTHOR = "Ernesto Casablanca"
LUCID_AUTHOR_EMAIL = "casablancaernesto@gmail.com"
LUCID_DESCRIPTION = "Lifting-based Uncertain Control Invariant Dynamics"
LUCID_HOMEPAGE = "https://github.com/TendTo/bazel-cpp-template"
LUCID_SOURCE = "https://github.com/TendTo/bazel-cpp-template"
LUCID_TRACKER = "https://github.com/TendTo/bazel-cpp-template/issues"
LUCID_LICENSE = "Apache-2.0"

# Can't parse the list
LUCID_CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

def _expose_variable_impl(_):
    pass

expose_variable = rule(
    implementation = _expose_variable_impl,
    attrs = {
        "value": attr.string(
            mandatory = True,
            doc = "The value of the variable to expose.",
        ),
    },
)
