LUCID_NAME = "lucid"
LUCID_VERSION = "0.0.1"
LUCID_AUTHOR = "AUTHOR"
LUCID_AUTHOR_EMAIL = "author@email.com"
LUCID_DESCRIPTION = "Lifting-based Uncertain Control Invariant Dynamics"
LUCID_HOMEPAGE = "https://gitlab.com/lucidtoolsource/lucid"
LUCID_SOURCE = "https://gitlab.com/lucidtoolsource/lucid"
LUCID_TRACKER = "https://gitlab.com/lucidtoolsource/lucid/issues"
LUCID_LICENSE = "Apache-2.0"

# Can't parse the list
LUCID_CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Typing :: Typed",
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
