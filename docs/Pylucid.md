# Pylucid

Lucid provides a thin wrapper around the C++ library, called **pylucid**, allowing you to use the main features of Lucid from Python.
There are multiple ways to use it.
You can either run it withing the Bazel environment, or you can install it in your Python environment.

> [!NOTE]  
> All bindings use the C++ Lucid library under the hood, which means that you need to ensure all the requirements listed in the [Installation](Installation.md) section are met.

> [!IMPORTANT]  
> Building the bindings on Windows with GUI support requires an additional step.
> See the [Building on Windows](#building-on-windows) section for more details.

## Installing Pylucid

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
pip install .[gui,verification,plot]
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
> To use the GUI, you need to have installed pylucid with `pip install .[gui]` or `pip install .[all]`.

To launch the GUI, use the command

```bash
pylucid-gui
```

The program will spin up a web server and open the GUI in your default web browser.
By default, the web page will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000), but you can change the port by using the `--port` option.

The GUI makes it more user-friendly to configure scenarios, run them, and visualize the results.

To see all available options, run

```bash
pylucid-gui --help
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"

## Troubleshooting

#### ImportError: libpython3.12.so.1.0: cannot open shared object file: No such file or directory

This error occurs when the expected Python shared library is not found in the expected location on the system
To fix this, you need to set the `LD_LIBRARY_PATH` environment variable to include the path to the Python library.
You can do this by running the following command:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/python/lib"
```

For example, if you are using Python 3.12 via `conda`

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(conda info --base)/envs/your_env/lib"
```

This change will only last for the current session.

#### LucidNotSupportedException: ...the following dependency was not included during compilation: 'X'

This error indicates that some optional dependencies were not included during the compilation of the bindings.
As a result, some features may not be available.
To resolve this, you must recompile the bindings from source, ensuring that the correct [flags are set](Installation.md#build-options) and the requirements are met.

For Gurobi support, see [Building with Gurobi](Installation.md#gurobi-requirements).

#### The process does not stop after pressing `Ctrl+C`

Pylucid is a thin wrapper around the C++ Lucid library, which means that when you press `Ctrl+C`, the Python interpreter sends a signal to the C++ executable, which may not handle it immediately.  
Long running operations may not notice the signal until they complete, which can cause the process to appear unresponsive.
In most cases the process will stop soon, (e.g., after an iteration of the optimiser has completed), but if a immediate termination is required, the safest way to stop the process is to close the terminal.

#### Address already in use. Port X is in use by another program

When running the GUI, you may encounter this error if the default port (5000) is already in use by another program.
To resolve this, you can either stop the program using that port or specify a different port when launching the GUI:

```bash
pylucid-gui --port 5001
```

#### GUI is stuck

This can happen for a variety of reasons, usually related to the web server being unable to conclude a long operation in a timely manner.
The simplest solution is to restart the process.
To stop the server, you can press `Ctrl+C` in the terminal where the GUI was launched or simply kill the terminal session.
