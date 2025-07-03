# Bindings

Lucid is a C++ library, but it also provides bindings for

- [Python](#python)

## Python

Lucid provides a thin wrapper around the C++ library, called **pylucid**, allowing you to use the main features of Lucid from Python.
There are multiple ways to use it.
You can either run it withing the Bazel environment, or you can install it in your Python environment.

> [!NOTE]  
> All bindings use the C++ Lucid library under the hood, which means that you need to ensure all the requirements listed in the [Installation](Installation.md) section are met.

> [!IMPORTANT]  
> Building the bindings on Windows with GUI support requires an additional step.
> See [Building on Windows](#building-on-windows-gui) for more details.

### Installing in your Python environment

You can install the bindings in your Python environment using the following command:

```bash
# Make sure you are in the lucid root directory
pip install .
```

> [!TIP]  
> It is recommended to use a virtual environment or a conda environment to avoid conflicts with other packages.

This will install the bindings in your Python environment, allowing you to use them directly from Python.
After installing, you can run the following command to check if everything is working correctly:

```bash
pylucid --help
```

#### Building on Windows

The Javascript Bazel rules, needed to build the GUI, have limited support for Windows, so you need to run the build script manually.
First, ensure you have the required dependencies installed:

- [Node.js](https://nodejs.org/)
- [pnpm](https://pnpm.io/)

Then, run the following command from the `lucid` root directory:

```bash
scripts\build_frontend.bat
```

You can then follow

### Use

There are two main ways to use the bindings.

[//]: # "@tabbed"
[//]: # "@tab"

#### Configuration script (recommended)

Create a Python script, for example `my_config.py`, with the following content:

```python
# my_config.py
import numpy as np
from pylucid import *


def scenario_config() -> "ScenarioConfig":
    # System dynamics
    f_det = lambda x: 1 / 2 * x
    f = lambda x: f_det(x) + np.random.normal(scale=0.8)
    # Safety specification
    gamma = 1
    T = 5  # Time horizon
    X_bounds = RectSet([(-1, 1)])  # State space
    X_init = RectSet([(-0.5, 0.5)])  # Initial set
    X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))  # Unsafe set
    # Parameters and inputs
    N = 1000
    x_samples = X_bounds.sample(N)
    xp_samples = f(x_samples)
    # Initial estimator hyperparameters. Can be tuned later
    regularization_constant = 1e-3
    sigma_f = 15.0
    sigma_l = np.array([1.75555556])
    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.
    # Estimator
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=sigma_f, sigma_l=sigma_l),
        regularization_constant=regularization_constant,
    )
    # Depending on the tuner selected in the dictionary above, the estimator will be fitted with different parameters.
    estimator.fit(x=x_samples, y=xp_samples, MedianHeuristicTuner())

    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=T,
        gamma=gamma,
        f_det=f_det,  # The deterministic part of the system dynamics
        num_freq_per_dim=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=estimator.get(Parameter.SIGMA_F),
        problem_log_file="problem.lp",  # The lp file containing the optimization problem
        iis_log_file="iis.ilp",  # The ilp file containing the IIS if the problem is infeasible
    )
```

Then, you can run the configuration script using the `pylucid` command:

```bash
python3 -m pylucid my_config.py
```

Use the `--help` option to see the available options:

```bash
python3 -m pylucid --help
```

The `scenario_config` function can also accept a `CLIArgs` argument, which allows you to pass the parsed command line arguments to the function.
This is useful if you want to customize the scenario configuration based on command line arguments.
E.g.,

```python
# my_config.py
import numpy as np
from pylucid import *


def scenario_config(args: CLIArgs = CLIArgs(seed=42)) -> "ScenarioConfig":
  # ...
  np.random.seed(args.seed)  # Use the seed from the command line arguments
  # ...
  return ScenarioConfig(...)
```

[//]: # "@end-tab"
[//]: # "@tab"

#### Main script

Create a Python script, for example `my_main.py`, with the following content:

```python
# my_main.py
import numpy as np
from pylucid import *
from pylucid.pipeline import pipeline

# System dynamics
f_det = lambda x: 1 / 2 * x
f = lambda x: f_det(x) + np.random.normal(scale=0.8)
# Safety specification
gamma = 1
T = 5  # Time horizon
X_bounds = RectSet([(-1, 1)])  # State space
X_init = RectSet([(-0.5, 0.5)])  # Initial set
X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))  # Unsafe set
# Parameters and inputs
N = 1000
x_samples = X_bounds.sample(N)
xp_samples = f(x_samples)
# Initial estimator hyperparameters. Can be tuned later
regularization_constant = 1e-3
sigma_f = 15.0
sigma_l = np.array([1.75555556])
num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.
# Estimator
estimator = KernelRidgeRegressor(
    kernel=GaussianKernel(sigma_f=sigma_f, sigma_l=sigma_l),
    regularization_constant=regularization_constant,
)
# Depending on the tuner selected in the dictionary above, the estimator will be fitted with different parameters.
estimator.fit(x=x_samples, y=xp_samples, MedianHeuristicTuner())

pipeline(
    x_samples=x_samples,
    xp_samples=xp_samples,
    X_bounds=X_bounds,
    X_init=X_init,
    X_unsafe=X_unsafe,
    T=T,
    gamma=gamma,
    f_det=f_det,  # The deterministic part of the system dynamics
    num_freq_per_dim=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
    estimator=estimator,  # The estimator used to model the system dynamics
    sigma_f=estimator.get(Parameter.SIGMA_F),
    problem_log_file="problem.lp",  # The lp file containing the optimization problem
    iis_log_file="iis.ilp",  # The ilp file containing the IIS if the problem is infeasible
)
```

Then, you can run the script as follows:

```bash
python3 my_main.py
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"

### Running with Bazel

Instead of installing the bindings in your Python environment, you can also run your scripts within the environment managed by Bazel.
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

### Troubleshooting

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
