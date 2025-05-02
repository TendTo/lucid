# Installation

## From Docker

A pre-build Docker image is available on the [GitHub repository](https://github.com/TendTo/lucid/pkgs/container/lucid).
To use it, run the following command:

```bash
# Pull the image
docker pull ghcr.io/tendto/lucid:main
# Run the image
# Mount the script you want to run (e.g. /path/to/script.py) somewhere in the container (e.g. /scripts)
# Keep in mind that you need to mount a Gurobi Web License (gurobi.lic) in the container
docker run --name lucid -it --rm -v/path/to/my/scripts:/scripts --volume=/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro ghcr.io/tendto/lucid:main /scripts/script.py
```

## From source

### Requirements

These are the tools and libraries required to build Lucid from source, along with the version used during development for each of them.
Other versions may work as well, but they have not been tested.

- [Bazel](https://bazel.build/) 8.1.1
  - We suggest using [bazelisk](https://github.com/bazelbuild/bazelisk) to manage Bazel's version.
- C++ compiler with C++20 support
  - [gcc](https://gcc.gnu.org/) 11.4.0
  - [msvc](https://visualstudio.microsoft.com/) 19.32.31332
- [Gurobi](https://www.gurobi.com/) 12.0.1

> [!NOTE]  
> Both a Gurobi installation and a valid license are required to build and run Lucid.  
> To indicate the location of the Gurobi installation, ensure that the `GUROBI_HOME` environment variable is set correctly.
> Alternatively, provide the flag `--repo_env=GUROBI_HOME=/path/to/gurobi` when running Bazel or set the `default_gurobi_home` parameter in the `MODULE.bazel` file.

### Building Lucid

Assuming all requirements have been met, the first step is to obtain the source code by cloning the repository.

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git
# Change directory
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
Just add the `--config` flag to the build command to use one of them.

| Configuration | Optimisations | Debug symbols | Assertions | Checks | Static linking | Used for              |
| ------------- | ------------- | ------------- | ---------- | ------ | -------------- | --------------------- |
| **default**   | ?             | ?             | Yes        | Yes    | No             | A fast, default build |
| `dbg`         | No            | Yes           | Yes        | Yes    | No             | Testing and debugging |
| `opt`         | Yes           | No            | No         | Yes    | No             | Production            |
| `bench`       | Yes           | No            | No         | No     | Yes            | Benchmarking          |
| `py`          | Yes           | No            | No         | No     | Yes            | Python bindings       |
