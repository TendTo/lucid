# Contributing

Some notes on developing and contributing to the project.

## Folder structure

The folder structure is as follows:

```bash
.
├── docs                    # Documentation
├── bindings                # Bindings in other languages
│   └── pylucid             # Python bindings (pylucid)
├── lucid                   # Main application
│   ├── util                # Modules (e.g., utility module)
│   │   ├── BUILD.bazel     # Utility library BUILD file
│   │   └── utility.h
│   ├── BUILD.bazel         # Main application BUILD file
│   └── main.cpp
├── third_party             # Third party libraries
│   ├── BUILD.bazel         # Third party libraries BUILD file
│   └── spdlog.BUILD.bazel  # spdlog BUILD file
├── tools                   # Tools and scripts
│   ├── BUILD.bazel
│   ├── cpplint.bzl         # cpplint rules
│   ├── generate_version_header.bat # Script to generate version header (Windows)
│   ├── generate_version_header.sh  # Script to generate version header (Linux)
│   ├── git_archive.bzl
│   ├── rules_cc.bzl
│   ├── workspace_status.bat # Script to generate the stable-status.txt file (Windows)
│   └── workspace_status.sh  # Script to generate the stable-status.txt file (Linux)
├── .bazelignore    # Bazel ignore file
├── .bazelrc        # Bazel configuration file
├── .bazelversion   # Bazel version lock file
├── .clang-format   # Clang format configuration file
├── CPPLINT.cfg     # cpplint configuration file
├── BUILD.bazel     # Root BUILD file
└── MODULE.bazel    # Root MODULE file
```

## Utility commands

```bash
# Build the main application.
# The executable can be found in the bazel-bin/lucid directory
bazel build //lucid
```

```bash
# Run the main application and pass an argument (e.g. 2)
bazel run //lucid -- 2
```

```bash
# Build the main application in debug mode
bazel build //lucid --config=dbg
```

```bash
# Build the main application in release mode
bazel run //lucid --config=opt
```

```bash
# Run all the tests
bazel test //tests/...
```

```bash
# Only run a specific tagged test
bazel test //tests/... --test_tag_filters=lucid
```

```bash
# Lint all the code
bazel test //lucid/...
```

```bash
# Build the documentation
# The documentation can be found in the bazel-bin/docs directory
bazel build //docs
```

```bash
# Remove all the build files
bazel clean
```

```bash
# Run the depend-what-you-use (DWYU) analysis on the main application
bazel build //lucid --config=dwyu
```

```bash
# Get some information about the cpp toolchain as bazel sees it
bazel cquery "@bazel_tools//tools/cpp:compiler" --output starlark --starlark:expr 'providers(target)'
```

```bash
# Get the dependencies of the main application.
# See https://docs.bazel.build/versions/main/query-how-to.html
bazel query --noimplicit_deps --notool_deps 'deps(//lucid)'
```

```bash
# Build the wheel for the python bindings
# The wheel can be found in the bazel-bin/pylucid directory
bazel build --config=py --python_version=3.13 //bindings/pylucid:pylucid_wheel
```

```bash
# Run the python bindings tests
bazel test --config=py --python_version=3.13 //bindings/pylucid/tests/...
```

```bash
# Run the python bindings main application
bazel run //bindings/pylucid --config=py --python_version=3.13
```
