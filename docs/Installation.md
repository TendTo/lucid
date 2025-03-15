# Installation

## From source

### Requirements

These are the tools and libraries required to build Lucid from source, along with the version used during development for each of them.
Other versions may work as well, but they have not been tested.

- [Bazel](https://bazel.build/) 8.1.1
  - We suggest using [bazelisk](https://github.com/bazelbuild/bazelisk) to manage Bazel's version.
- C++ compiler
  - [gcc](https://gcc.gnu.org/) 11.4.0
  - [msvc](https://visualstudio.microsoft.com/) 19.32.31332
- [Gurobi](https://www.gurobi.com/) 12.0.1

### Building Lucid

Assuming all requirements have been met, the first step is to obtain the source code by cloning the repository.

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git
# Change directory
cd lucid
```

Before starting the build process, you will need to specify the directory where Gurobi is installed.
In the `MODULE.build` file, change the `GUROBI_HOME` variable to the correct path (e.g., `/opt/gurobi912/linux64` or `C:/gurobi912/win64`).  
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

## From Docker

A pre-build Docker image is available on the [GitHub repository](https://github.com/TendTo/lucid/pkgs/container/lucid).
To use it, run the following command:

```bash
# Pull the image
docker pull ghcr.io/tendto/lucid:main
# Run the image
docker run -it ghcr.io/tendto/lucid:main
```
