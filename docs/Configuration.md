# Configuration

Lucid supports a wide range of configuration options and can run on systems with different parameters.
Hence, it is important to understand how to configure Lucid for your specific use case.

For the user's convenience, Lucid can be configured from the command line or through a configuration file.
Moreover, the configuration can be expressed as a [json](https://www.json.org/json-en.html) or [yaml](https://yaml.org/) file for convenience, or as an arbitrary Python script for a complex custom configuration.

## Command line options

The command line options are used to configure Lucid at runtime.
To know the available options, you can run the following command:

```bash
lucid --help
```

Lucid will run a default scenario with the parameters provided through the command line.

## Configuration file

[//]: # "@tabbed"
[//]: # "@tab"

### YAML configuration (recommended)

Create a file named `config.yaml` and define your configuration options.
You can use the following example as a template:

```yaml
# yaml-language-server: \$schema=https://tendto.github.io/lucid/configuration_schema.json
# Basic configuration
verbose: 3 # LOG_INFO
seed: 42

# System transition function
# (x1, x2, ..., xn) are the components of the n-dimensional input state space
# All components of the input state space must appear in the system dynamics
# Each element of the list corresponds to a component of the output state space
# E.g., the following system has 3D input state space (x1, x2, x3)
# and 2D output state space (y1, y2)
system_dynamics:
  - x1**2 + x2 / 2 + cos(x3) # y1
  - 2 * x1 + sin(-x2) # y2

# Sets definition
# RectSets can be defined 3 ways:
X_bounds:
  RectSet: # - As a pair of lower and upper bounds
    lower: [-3, -2, 0.1]
    upper: [2.5, 1, 0.2]
X_init:
  MultiSet: # MultiSet are just lists of RectSets
    - RectSet: # - As a list of lower and upper bounds pair for each dimension
        - [0.1, 0.2]
        - [0.1, 0.2]
        - [0.1, 0.2]
    - RectSet:
        - [0.1, 0.2]
        - [0.1, 0.2]
        - [0.1, 0.2]

X_unsafe: "RectSet([0.4, 0.1, 1], [0.8, 0.3, 0])" # - As a string

# Algorithm parameters
gamma: 10.0
c_coefficient: 1.0
lambda: 0.0001
num_samples: 1000
time_horizon: 10

# Feature map parameters
sigma_f: 15.1
sigma_l: [1.1, 2.0, 3.0] # Can be a single float or a list
num_frequencies: 4
oversample_factor: 2.1
num_oversample: -1

# Execution options
noise_scale: 0.01
plot: true
verify: true

# Output options
problem_log_file: "problem.lp"
iis_log_file: "iis.ilp"
```

Then, you can run Lucid with the following command:

```bash
lucid config.yaml
```

The full [JSON schema](https://json-schema.org/) of the configuration options can be found [here](/bindings/pylucid/configuration_schema.json).

[//]: # "@end-tab"
[//]: # "@tab"

### JSON configuration

Create a file named `config.json` and define your configuration options.
You can use the following example as a template:

```json
{
  "$schema": "https://tendto.github.io/lucid/configuration_schema.json",
  // Basic configuration
  "verbose": 3, // LOG_INFO
  "seed": 42,
  /*
   * System transition function
   * (x1, x2, ..., xn) are the components of the n-dimensional input state space
   * All components of the input state space must appear in the system dynamics
   * Each element of the list corresponds to a component of the output state space
   * E.g., the following system has 3D input state space (x1, x2, x3)
   * and 2D output state space (y1, y2)
   */
  "system_dynamics": [
    "x1**2 + x2 / 2 + cos(x3)", // y1
    "2 * x1 + sin(-x2)" // y2
  ],
  /*
   * Sets definition
   * RectSets can be defined 3 ways:
   */
  "X_bounds": {
    // - As a pair of lower and upper bounds
    "RectSet": {
      "lower": [-3, -2, 0.1],
      "upper": [2.5, 1, 0.2]
    }
  },
  "X_init": {
    // MultiSet are just lists of RectSets
    "MultiSet": [
      {
        // - As a list of lower and upper bounds pair for each dimension
        "RectSet": [
          [0.1, 0.2],
          [0.1, 0.2],
          [0.1, 0.2]
        ]
      },
      {
        "RectSet": [
          [0.1, 0.2],
          [0.1, 0.2],
          [0.1, 0.2]
        ]
      }
    ]
  },
  // - As a string
  "X_unsafe": "RectSet([0.4, 0.1, 1], [0.8, 0.3, 0])",
  "gamma": 10.0,
  "c_coefficient": 1.0,
  "lambda": 0.0001,
  "num_samples": 1000,
  "time_horizon": 10,
  "sigma_f": 15.1,
  "sigma_l": [1.1, 2.0, 3.0],
  "num_frequencies": 4,
  "oversample_factor": 2.1,
  "num_oversample": -1,
  "noise_scale": 0.01,
  "plot": true,
  "verify": true,
  "problem_log_file": "problem.lp",
  "iis_log_file": "iis.ilp"
}
```

Then, you can run Lucid with the following command:

```bash
lucid config.json
```

The full [JSON schema](https://json-schema.org/) of the configuration options can be found [here](/bindings/pylucid/configuration_schema.json).

[//]: # "@end-tab"
[//]: # "@tab"

### Python configuration

Create a file named `config.py` and define your configuration options.
There are no restrictions on how what you do in the script, as long as you define a `scenario_config(args)` function that returns a `ScenarioConfig` object.
For example:

```python
# config.py
from pylucid import *


# args will contain the command line arguments.
# You can choose to use or ignore them.
def scenario_config(args: CLIArgs):
    # Model
    f_det = lambda x: 0.5 * x
    f = lambda x: f_det(x) + np.random.normal(scale=args.noise_scale)

    # Sets
    X_bounds = RectSet([(-1, 1)])
    X_init = RectSet([(-0.5, 0.5)])
    X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))

    # Sampling
    x_samples = X_bounds.sample(args.num_samples)
    xp_samples = f(x_samples)

    # Estimator
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=args.sigma_f, sigma_l=args.sigma_l),
        regularization_constant=args.lambda_,
    )

    # Lucid configuration
    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=args.time_horizon,
        gamma=args.gamma,
        num_freq_per_dim=args.num_frequencies,
        f_det=f_det,
        estimator=estimator,
        sigma_f=args.sigma_f,
        oversample_factor=args.oversample_factor,
        problem_log_file=args.problem_log_file,
    )
```

Then, you can run Lucid with the following command:

```bash
lucid config.py
```

[//]: # "@end-tab"
[//]: # "@end-tabbed"

## Mixing configuration files and command line options

Configuration files and command line options can be mixed, but with a few caveats.

- Only a **single** configuration file can be used at a time.
- The **order** in which the configuration files and command line options are applied **is the same** as the order in which they are **specified**.
  - `pylucid config.yaml --verbose 2` _first_ applies the configuration from `config.yaml` and _then_ overrides `verbose` with the value `2`.
  - `pylucid --verbose 2 config.yaml` _first_ sets `verbose` to `2` and _then_ overrides it with the value from `config.yaml`, if present
- Python configuration files are always applied **last**, as they have complete control over what to do with the command line arguments.
  - `pylucid config.py --verbose 2 ` will forward the `verbose` argument to the user's `scenario_config` function, which can then choose to use it or ignore it.

> [!IMPORTANT]
> When multiple configuration sources compete, the last specified option will take precedence.
