# LUCID: Lifting-based Uncertain Control Invariant Dynamics

[![lucid CI](https://github.com/TendTo/lucid/actions/workflows/lucid.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/lucid.yml)
[![pylucid CI](https://github.com/TendTo/lucid/actions/workflows/pylucid.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/pylucid.yml)
[![Docker CI](https://github.com/TendTo/lucid/actions/workflows/docker.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/docker.yml)
[![Docs CI](https://github.com/TendTo/lucid/actions/workflows/docs.yml/badge.svg)](https://github.com/TendTo/lucid/actions/workflows/docs.yml)

<div style="display: none">
<!-- Necessary to load the image in the Doxygen output folder. Do not remove! -->
![Lucid banner](docs/_static/lucid-banner.png)
</div>

Simple modern template for a C++ project using Bazel with modules.

> My integral is your table.  
> -- _[Oliver Schön](https://oliverschon.com/)_

## Quick installation

For more details, see the [installation instructions](docs/Installation.md).

### Docker

> [!Note]  
> You will need a [Gurobi Web License Service (WLS) license](https://www.gurobi.com/features/web-license-service/)

```bash
# Pull the image
docker pull ghcr.io/tendto/lucid:main
# Run the image on script/path/to/script.py.
# Needs the Gurobi WS licence /path/to/gurobi.lic
docker run --name lucid -it --rm \
  -v/path/to/script.py:/scripts \
  -v/path/to/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
  ghcr.io/tendto/lucid:main /scripts/script.py
```

### Source

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git
# Change directory
cd lucid
# Compile and run lucid
bazel run //lucid -- [args]
```

### Python (from source)

```bash
# Clone the repository
git clone https://github.com/TendTo/lucid.git
# Change directory
cd lucid
# Create a virtual environment (optional)
python3 -m venv .venv
# Activate the virtual environment on Linux (optional)
source .venv/bin/activate
# Activate the virtual environment on Windows (optional)
.\.venv\Scripts\activate
# Install the python wrapper (pylucid)
pip install .
# Ensure pylucid is installed
python3 -c "import pylucid; print(pylucid.__version__)"
```
