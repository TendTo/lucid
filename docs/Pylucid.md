# Pylucid

Lucid provides a thin wrapper around the C++ library, called **pylucid**, allowing you to use the main features of Lucid from Python.
There are multiple ways to use it.
You can either run it withing the Bazel environment, or you can install it in your Python environment.

> [!NOTE]  
> All bindings use the C++ Lucid library under the hood, which means that you need to ensure all the requirements listed in the [Installation](Installation.md) section are met.

> [!IMPORTANT]  
> Building the bindings on Windows with GUI support requires an additional step.
> See the [Building on Windows](#building-on-windows) section for more details.

## Installing Pylucid (pre-built)

We provide a pre-built Pylucid package that can be installed directly with pip without requiring any compilation.

### Requirements

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

If you notice any errors, please refer to the [Troubleshooting](FAQ.md#troubleshooting) section or open an issue.

## Installing Pylucid (source)

[//]: # "@tabbed"
[//]: # "@tab"

### Local installation (recommended)

You can install the bindings in your Python environment using the following command:

```bash
# Make sure you are in the lucid root directory
pip install .
```

It is possible to customise the installation, including optional dependencies, to enable additional features.

```bash
# Install the bindings with optional dependencies
# GUI => Graphical User Interface, pylucid-gui
# Verification => Verify the barrier via the dReal SMT solver
# Plot => Plot the results using plotly
pip install ".[gui,verification,plot]"
```

> [!TIP]  
> It is recommended to use a virtual environment or a conda environment to avoid conflicts with other packages.

This will install the bindings in your Python environment, allowing you to use them directly from Python.
After installing, you can run the following command to check if everything is working correctly:

```bash
pylucid --version
```

[//]: # "@end-tab"
[//]: # "@tab"

### Bazel installation

Instead of installing the bindings in your Python environment, you can also run your scripts within the environment managed by Bazel.
This can be useful if you want to keep the hermetic setup provided by the build system.  
Write your script somewhere in the `lucid` directory.
In the same directory, create or update the file called `BUILD.bazel` with the following content:

```bzl
# BUILD.bazel
py_binary(
    name = "my_script",
    srcs = ["main.py"],
    main = "main.py",
    python_version = "PY3",
    deps = [
        "@lucid//bindings/pylucid:pylucid_lib",
    ],
)
```

Then, run your script using the following command:

```bash
bazel run //path/to/your/script:your_script
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"

### Building on Windows

The Javascript Bazel rules, needed to build the GUI, have limited support for Windows, so you need to run the build script manually.
First, ensure you have the required dependencies installed:

- [Node.js](https://nodejs.org/)
- [pnpm](https://pnpm.io/)

Then, run the following command from the `lucid` root directory:

```bash
scripts\build_frontend.bat
```

You can then follow the standard [installation procedure](#installing-pylucid).

## Use

There are two main ways to use the bindings.

[//]: # "@tabbed"
[//]: # "@tab"

### Providing a configuration (recommended)

Lucid tries to be as flexible and user-friendly as possible, providing multiple ways to configure your scenarios.
You can provide a configuration as a **JSON file**, a **Yaml file** a **Python script** or just using the **command line arguments**.
For more details, see the [Configuration](Configuration.md) section.

For instance, a scenario could be run as follows:

```bash
pylucid --system_dynamics 'x1 / 2' \
        --X_bounds 'RectSet([-1], [1])' \
        --X_init 'RectSet([-0.5], [0.5])' \
        --X_unsafe 'MultiSet([RectSet([-1], [-0.9]), RectSet([0.9], [1])])' \
        --seed 42 \
        --gamma 1.0 \
        --time_horizon 5 \
        --num_samples 1000 \
        --lambda 1e-3 \
        --sigma_f 15.0 \
        --sigma_l 1.75555556 \
        --num_frequencies 4 \
        --plot \
        --verify \
        --oversample_factor 32.0 \
        -v 4 \
        --problem_log_file problem.lp
```

See `pylucid --help` for more details on the available options.

[//]: # "@end-tab"
[//]: # "@tab"

### Main script

Create a Python script, for example `my_main.py`, with the following content:

```python
# my_main.py
from pylucid import *
from pylucid.pipeline import pipeline


# Model
system_dynamics = lambda x: 0.5 * x
noisy_system_dynamics = lambda x: system_dynamics(x) + np.random.normal(scale=0.01)

# Sets
X_bounds = RectSet([(-1, 1)]) # State space
X_init = RectSet([(-0.5, 0.5)]) # Initial set
X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)])) # Unsafe set

# Sampling
x_samples = X_bounds.sample(1000)
xp_samples = noisy_system_dynamics(x_samples)

# Estimator
estimator = KernelRidgeRegressor(
    kernel=GaussianKernel(sigma_f=1.0, sigma_l=1.7775),
    regularization_constant=1e-3,
)

pipeline(
    args=Configuration(
        system_dynamics=system_dynamics,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        x_samples=x_samples,
        xp_samples=xp_samples,
        estimator=estimator,
        oversample_factor=16.0,
        plot=True,
        verify=True,
    ),
)
```

Then, run it with

```bash
python3 my_main.py
```

[//]: # "@end-tab"
[//]: # "@tab"

### Graphical User Interface (GUI)

Lucid provides a graphical user interface (GUI) to interactively configure and run scenarios.

> [!NOTE]  
> The GUI is not installed by default.
> To use the GUI, you need to have installed pylucid with `pip install .[gui]`.

To launch the GUI, use the command

```bash
pylucid-gui
```

The program will spin up a web server and open the GUI in your default web browser.
By default, the web page will be available at [http://127.0.0.1:3661](http://127.0.0.1:3661), but you can change the port by using the `--port` option.

The GUI makes it more user-friendly to configure scenarios, run them, and visualize the results.

To see all available options, run

```bash
pylucid-gui --help
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"
