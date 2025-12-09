# <img alt="Icon" src="docs/_static/logo.svg" align="left" width="35" height="35"> LUCID

_Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems_

[![lucid CI](https://github.com/TendTo/lucid/actions/workflows/lucid.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/lucid.yml)
[![pylucid CI](https://github.com/TendTo/lucid/actions/workflows/pylucid.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/pylucid.yml)
[![Docker CI](https://github.com/TendTo/lucid/actions/workflows/docker.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/docker.yml)
[![Docs CI](https://github.com/TendTo/lucid/actions/workflows/docs.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/docs.yml)

> [!TIP]  
> Try out the [online demo](https://tendto.github.io/lucid/demo//)!
> Note that the memory in the browser is limited, so only small problems can be solved.

> [!WARNING]  
> This project is under active development.
> Features and APIs may change without prior notice.
> Please refer to the [changelog](docs/Changelog.md) for the latest updates.

## Quick installation

For more details, see the [installation](docs/Installation.md) or the [Pylucid](docs/Pylucid.md) sections.
If you encounter any errors, please refer to the [Troubleshooting](docs/FAQ.md#troubleshooting) section or open an [issue](https://github.com/TendTo/lucid/issues).

[//]: # "@tabbed"
[//]: # "@tab"

### Docker (pre-built)

**Requirements**

- [Docker](https://www.docker.com/)
- (_Optional_) [Gurobi Web License Service (WLS) license](https://www.gurobi.com/features/web-license-service/)

**Command**

```bash
# Pull the image
docker pull ghcr.io/tendto/lucid:main

# Run the image on script/path/to/script.py.
# Needs a Gurobi WS licence to use the Gurobi solver.
docker run --name lucid -it --rm \
  -v/path/to/script.py:/scripts \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  ghcr.io/tendto/lucid:main /scripts/script.py

# Run the GUI.
# Needs a Gurobi WS licence to use the Gurobi solver.
docker run --name lucid -it --rm -p 3661:3661 \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  --entrypoint pylucid-gui ghcr.io/tendto/lucid:main
```

[//]: # "@end-tab"
[//]: # "@tab"

### Docker (source)

**Requirements**

- [Docker](https://www.docker.com/)
- (_Optional_) [Gurobi Web License Service (WLS) license](https://www.gurobi.com/features/web-license-service/)

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
- (_Optional_) [Gurobi licence](https://www.gurobi.com/)

**Installation commands**

```bash
# Create a virtual environment (optional)
python3 -m venv .venv

# Activate the virtual environment on Linux (optional)
source .venv/bin/activate

# Activate the virtual environment on Windows (optional)
.venv\Scripts\activate

# Install pylucid
pip install "pylucid[gui]" --index-url "https://gitlab.com/api/v4/projects/71977529/packages/pypi/simple"

# Ensure pylucid is installed correctly
python3 -c "import pylucid; print(pylucid.__version__)"
```

[//]: # "@end-tab"
[//]: # "@tab"

### Python (from source)

**Requirements**

- [Bazel](https://bazel.build/) 8.1.1
- [Python](https://www.python.org/) 3.8 or higher
- C++ compiler with C++20 support.  
  We tested in particular:
  - **On Linux**: [gcc](https://gcc.gnu.org/) 11.4.0
  - **On Windows**: [msvc](https://visualstudio.microsoft.com/) 19.32.31332
  - **On macOS**: [Clang/LLVM](https://clang.llvm.org/) 15.0.0
- (_Optional_) [Gurobi licence](https://www.gurobi.com/)

**Installation commands**

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git

# Move to the root of the repository
cd lucid

# Create a virtual environment (optional)
python3 -m venv .venv

# Activate the virtual environment on Linux (optional)
source .venv/bin/activate

# Activate the virtual environment on Windows (optional)
.venv\Scripts\activate

# Install the python wrapper (pylucid)
pip install ".[gui]"

# Ensure pylucid is installed
python3 -c "import pylucid; print(pylucid.__version__)"
```

[//]: # "@end-tab"
[//]: # "@tab"

### Source

**Requirements**

- [Bazel](https://bazel.build/) 8.1.1
- C++ compiler with C++20 support.  
  We tested in particular:
  - **On Linux**: [gcc](https://gcc.gnu.org/) 11.4.0
  - **On Windows**: [msvc](https://visualstudio.microsoft.com/) 19.32.31332
  - **On macOS**: [Clang/LLVM](https://clang.llvm.org/) 15.0.0
- (_Optional_) [Gurobi licence](https://www.gurobi.com/)

**Installation commands**

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git

# Move to the root of the repository
cd lucid

# Compile and run lucid
bazel run //lucid -- [args]
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"
