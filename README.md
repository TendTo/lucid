# <img alt="Icon" src="docs/_static/logo.svg" align="left" width="35" height="35"> LUCID

_Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems_

## Quick installation

For more details, see the [installation](https://gitlab.com/lucidtoolsource/lucid/-/blob/main/docs/Installation.md) or the [Pylucid](https://gitlab.com/lucidtoolsource/lucid/-/blob/main/docs/Pylucid.md) chapters.

If you encounter any errors, please refer to the [Troubleshooting](https://gitlab.com/lucidtoolsource/lucid/-/blob/main/docs/FAQ.md#troubleshooting) section or open an issue.

[//]: # "@tabbed"
[//]: # "@tab"

### Docker (pre-built)

**Requirements**

- [Docker](https://www.docker.com/)
- [Gurobi Web License Service (WLS) license](https://www.gurobi.com/features/web-license-service/)

**Command**

```bash
# Build the image
docker pull registry.gitlab.com/lucidtoolsource/lucid:latest

# Run the image on script/path/to/script.py.
# Needs a Gurobi WS licence to use the Gurobi solver.
docker run --name lucid -it --rm \
  -v/path/to/script.py:/scripts \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  registry.gitlab.com/lucidtoolsource/lucid:latest /scripts/script.py

# Run the GUI.
# Needs a Gurobi WS licence to use the Gurobi solver.
docker run --name lucid -it --rm -p 3661:3661 \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  --entrypoint pylucid-gui \
  registry.gitlab.com/lucidtoolsource/lucid:latest
```

[//]: # "@end-tab"
[//]: # "@tab"

### Docker (source)

**Requirements**

- [Docker](https://www.docker.com/)
- [Gurobi Web License Service (WLS) license](https://www.gurobi.com/features/web-license-service/)

**Command**

```bash
# Build the image
docker build -t lucid .

# Run the image on script/path/to/script.py.
# Needs a Gurobi WS licence to use the Gurobi solver.
docker run --name lucid -it --rm \
  -v/path/to/script.py:/scripts \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  lucid /scripts/script.py

# Run the GUI.
# Needs a Gurobi WS licence to use the Gurobi solver.
docker run --name lucid -it --rm -p 3661:3661 \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  --entrypoint pylucid-gui lucid
```

[//]: # "@end-tab"
[//]: # "@tab"

### Python (pre-built)

**Requirements**

- [Python](https://www.python.org/) 3.8 or higher
- [Linux, Glibc >= 2.35](https://gist.github.com/richardlau/6a01d7829cc33ddab35269dacc127680), [Windows](https://www.microsoft.com/windows) or [ARM macOS](https://www.apple.com/macos/) operating system
- [Gurobi](https://www.gurobi.com/) 12.0.0 or higher

> [!WARNING]  
> The pre-built Python package expects to find the _Gurobi Optimizer >= 12.0_ installed on your system.
> You can freely download it from the [Gurobi website](https://www.gurobi.com/downloads/) (a login may be required).
> You **do not need** to have a valid license if you don't plan to use the Gurobi solver.

**Installation commands**

```bash
# Install pylucid
pip install "pylucid[gui,plot]" --index-url "https://gitlab.com/api/v4/projects/71977529/packages/pypi/simple"

# Ensure pylucid is installed correctly
python3 -c "import pylucid; print(pylucid.__version__)"
```

[//]: # "@end-tab"
[//]: # "@tab"

### Python (source)

**Requirements**

- [Bazel](https://bazel.build/) 8.1.1
- [Python](https://www.python.org/) 3.8 or higher
- C++ compiler with C++20 support
  - **On Linux**: [gcc](https://gcc.gnu.org/) 11.4.0
  - **On Windows**: [msvc](https://visualstudio.microsoft.com/) 19.32.31332
  - **On macOS**: [Clang/LLVM](https://clang.llvm.org/) 15.0.0
- [Gurobi](https://www.gurobi.com/) 12.0.0 or higher

**Installation commands**

```bash
# Clone the repository
git clone https://gitlab.com/lucidtoolsource/lucid.git

# Move to the root of the repository
cd lucid

# Install the python wrapper (pylucid)
pip install ".[gui,plot]"

# Ensure pylucid is installed
python3 -c "import pylucid; print(pylucid.__version__)"
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"
