# Installation

## Docker

### Requirements

- [Docker](https://www.docker.com/)
- [Gurobi Web License Service (WLS) license](https://www.gurobi.com/features/web-license-service/)

### Using Lucid with Docker

A pre-build Docker image is available on the [GitHub repository](https://github.com/TendTo/lucid/pkgs/container/lucid).
First, pull the image from the repository's container registry:

```bash
# Pull the image
docker pull ghcr.io/tendto/lucid:main

# If you want to use the source version, you can build it yourself
# Build the image
docker build -t lucid .
```

Then, simply run the image.
You have the option to run the main script, to which you have to pass the configuration, or the GUI, which can be accessed via a web browser at `http://localhost:3661`.

```bash
# Run the image
# Mount the script you want to run (e.g. /path/to/script.py) somewhere in the container (e.g. /scripts)
# Keep in mind that you need to mount a Gurobi WS License (gurobi.lic) in the container for the Gurobi solver to work.
docker run --name lucid -it --rm \
  -v/path/to/script.py:/scripts \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  ghcr.io/tendto/lucid:main /scripts/script.py

# Run the GUI.
# Keep in mind that you need to mount a Gurobi WS License (gurobi.lic) in the container for the Gurobi solver to work.
docker run --name lucid -it --rm -p 3661:3661 \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  --entrypoint pylucid-gui ghcr.io/tendto/lucid:main
```

## Lucid (source)

### Requirements

These are the tools and libraries required to build Lucid from source, along with the version used during development for each of them.
Other versions may work as well, but they have not been tested.

- [Bazel](https://bazel.build/) 8.1.1
  - We suggest using [bazelisk](https://github.com/bazelbuild/bazelisk) to manage Bazel's version.
- C++ compiler with C++20 support
  - **On Linux**: [gcc](https://gcc.gnu.org/) 11.4.0
  - **On Windows**: [msvc](https://visualstudio.microsoft.com/) 19.32.31332
  - **On macOS**: [Clang/LLVM](https://clang.llvm.org/) 15.0.0
- [Gurobi](https://www.gurobi.com/) 12.0.1

### Gurobi requirements

While there are other solvers supported by Lucid, [Gurobi](https://www.gurobi.com/) is the recommended one.
Being a commercial solver, it must be installed separately and requires a valid license to run.

Before the installation, ensure that the `GUROBI_HOME` environment variable is set correctly.

[//]: # "@tabbed"
[//]: # "@tab"

## Linux

Ensure that the `GUROBI_HOME` environment variable is set correctly with `printenv | grep GUROBI_HOME`.
You can set it for the duration of the current shell with `export GUROBI_HOME=/path/to/gurobi` (e.g., `export GUROBI_HOME=/opt/gurobi1201/linux64`).

[//]: # "@end-tab"
[//]: # "@tab"

## Windows

Using [powershell](https://learn.microsoft.com/powershell/scripting/overview?view=powershell-7.5), ensure that the `GUROBI_HOME` environment variable is set correctly with `[Environment]::GetEnvironmentVariable("GUROBI_HOME")`.
You can set it for the duration of the current shell with `[Environment]::SetEnvironmentVariable("GUROBI_HOME","\path\to\gurobi")` (e.g., `[Environment]::SetEnvironmentVariable("GUROBI_HOME","C:\gurobi1201\win64")`.

[//]: # "@end-tab"
[//]: # "@tab"

## macOS

Ensure that the `GUROBI_HOME` environment variable is set correctly with `printenv | grep GUROBI_HOME`.
You can set it for the duration of the current shell with `export GUROBI_HOME=/path/to/gurobi` (e.g., `export GUROBI_HOME=/Library/gurobi1201/macos_universal2`).

[//]: # "@end-tab"
[//]: # "@end-tabbed"

Instead of setting the `GUROBI_HOME` environment variable, you can add the flag `--repo_env=GUROBI_HOME=/path/to/gurobi` when running Bazel or set the `default_gurobi_home` parameter in the `MODULE.bazel` file.
The `--action_env=GUROBI_HOME=/path/to/gurobi` flag will make it so that the gurobi installation is added to the `rpath` of the binary.

When running the software, ensure that the Gurobi license file can be found via the `GRB_LICENSE_FILE` environment variable.
For more information, refer to the [Gurobi documentation](https://support.gurobi.com/hc/en-us/articles/13443862111761-How-do-I-set-system-environment-variables-for-Gurobi).

### Building Lucid

Assuming all requirements have been met, the first step is to obtain the source code by cloning the repository.

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git

# Move to the root of the repository
cd lucid
```

Then, run the following command to build the software:

```bash
# Build lucid
bazel build //lucid
```

The binary will be located in the `bazel-bin/lucid` directory.
If you also want to run it immediately, taking advantage of the environment provided by Bazel, use the following command:

```bash
# Build and run lucid
bazel run //lucid -- [args]
```

### Build options

Lucid comes with a few predefined build configuration for the most common use cases.
Just add the `--config` flag followed by the desired configuration when running Bazel.

| Configuration | Optimized | Debug | Assertions | Input checks | Logging | Verbose Eigen | Parallelized (OMP) | Used for              |
| ------------- | --------- | ----- | ---------- | ------------ | ------- | ------------- | ------------------ | --------------------- |
| **default**   | ?         | ?     | Yes        | Yes          | Yes     | No            | No                 | Default build         |
| `dbg`         | No        | Yes   | Yes        | Yes          | Yes     | Yes           | No                 | Testing and debugging |
| `snt`         | No        | Yes   | Yes        | Yes          | Yes     | No            | No                 | Memory sanitization   |
| `opt`         | Yes       | No    | No         | Yes          | Yes     | No            | Yes                | Production            |
| `py`          | Yes       | No    | No         | Yes          | Yes     | No            | Yes                | Python bindings       |
| `bench`       | Yes       | No    | No         | No           | No      | No            | Yes                | Benchmarking          |

For example, to build Lucid with the `opt` configuration, you can run:

```bash
# Build with the opt configuration
bazel build --config=opt //lucid
```

If you want even more fine-grained control over the build, you can also use the following flags or even add more custom compiler flags.

| Flag                                   | Description                                                 |
| -------------------------------------- | ----------------------------------------------------------- |
| `enable_static_build`                  | Build Lucid as a static library. Defaults to `False`.       |
| `enable_dynamic_build`                 | Build Lucid as a dynamic library. Defaults to `False`.      |
| `enable_python_build`                  | Build Lucid with Python bindings. Defaults to `False`.      |
| `enable_omp_build`                     | Build Lucid with OpenMP support. Defaults to `False`.       |
| `enable_benchmark_build`               | Build Lucid with benchmarking support. Defaults to `False`. |
| `enable_matplotlib_build`              | Build Lucid with Matplotlib support. Defaults to `False`.   |
| `enable_gurobi_build`                  | Build Lucid with Gurobi support. Defaults to `True`.        |
| `enable_alglib_build`                  | Build Lucid with ALGLIB support. Defaults to `True`.        |
| `enable_highs_build`                   | Build Lucid with HiGHS support. Defaults to `True`.         |
| `enable_verbose_eigen_build`           | Enable verbose logging for Eigen. Defaults to `False`.      |
| `python_version`                       | Specify the Python version to use for the Python bindings.  |
| `compilation_mode=[fastbuild,dbg,opt]` | Use Bazel's compilation modes. Default to `fastbuild`.      |

Example of a build command with custom flags.
Some combination of these flags may not be compatible with each other.

```bash
# Build with custom flags
bazel build \
  --compilation_mode=opt \
  --enable_dynamic_build=True \
  --enable_matplotlib_build=False \
  --enable_gurobi_build=True \
  --enable_verbose_eigen_build=False \
  --action_env=GUROBI_HOME=/path/to/gurobi \
  --cxxopt=-gdwarf-4 \
  --cxxopt=-O3 \
  --cxxopt=-DNDEBUG \
  //lucid
```
