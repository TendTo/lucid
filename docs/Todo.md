# To do

## Graphical overview

```mermaid
flowchart LR
    subgraph wp1 [Work package 1]
        direction TB
        a1["(A1) Reimplementation of existing code"]
        a2["(A2) Defining general components/structures"]
        a3["(A3) Runs on different/any user defined case study withing the current scope"]
        a4["(A4) Automatic hyperparameter optimisation"]
    end
    m1>"(M1) Useful tool"]
    subgraph Extensions
        direction RL
        e1(["(E1) Inductive k barriers"])
        m2>"(M2) Control synthesis"]
        e2(["(E2) Reachability specifications"])
        e3(["(E3) Verification with dreal (or Z3 / CVC5)"]):::available
        e4(["(E4) Change of the LP solver"]):::in-development
        e5(["(E5) Non linear optimizer for the synthesis"])
        e6(["(E6) Guiding the optimizing process via heuristics"])
        e7(["(E7) Pipeline of kernels"])
        e7(["(E8) GUI"])
    end
    a1 --> a2 --> a3 --> a4
    wp1 --> m1

classDef available stroke:#0b0,stroke-width:2px;
classDef in-development stroke:#bb0,stroke-width:2px,stroke-dasharray:2px;
classDef planned stroke:#0bb,stroke-width:2px,stroke-dasharray:5px;
classDef not-planned stroke:#b00,stroke-width:2px,stroke-dasharray:15px;
```

## Work package 1

### Activity 1

- [x] Complete initial implementation reproducing the results of the paper for the first test case
- [x] Rename variables
- [x] Fix current implementation correctness with integration testing
- [x] Implement second test case
- [x] Implement a proper fftn
- [x] Check whether it is possible to avoid the padding to get the same information
- [x] Implement the padding
- [x] Handle permutations
- [x] Eagerly update the permutation in the tensor

### Activity 2

- [ ] Add splitter (division of X and Y into training and validation sets)
- [x] Add scorer (how to evaluate the performance of the model)

### Activity 3

- [ ] Run Automated Anaesthesia
- [ ] Run Building Automation System

### Activity 4

- [x] Median Heuristic tuner
- [x] Grid search tuner
- [x] LBFGS tuner
- [x] Parallelise the hyperparameter optimisation (grid search)

## Extensions

### Extension 3

- [x] Add support for dReal

### Extension 4

- [x] Add support ALGLIB
- [x] Add support HiGHS
- [ ] Add support SoPlex

### Extension 8

- [x] Parse the arithmetic expressions
- [x] Create Flask server
- [x] Create HTML interface

## Miscellaneous

## Adding new features

- [x] Centralised seed setting
- [x] Create the flask server
- [x] Create the HTML interface
- [x] Allow estimator, kernel and feature_map selection
- [x] Improve the default scenario_configuration
- [x] Support multi-set in frontend
- [x] Improve error handling in the frontend
- [x] Support more kinds of functions in preview-graph
- [x] Support csv upload for samples
- [ ] Highlight set on over
- [ ] Update figure without rerendering it

### Performance

- [ ] Use templates instead of polymorphism where possible
- [ ] Parallelization via OpenMP and/or GPU
- [ ] Remove lattice points from $X_{\hat{N}}$ that intersect with the unsafe set $X_u$ (effort 1/performance +2)
- [ ] Eliminate cross-frequencies from the basis, thus reducing the problem size significantly (effort 1/performance +3.5)
- [ ] Use cheating coefficient c_coefficient<1 and use dReal to certify if the constraints hold on the entire continuous sets X_bounds, X_init, X_unsafe (effort 1/performance +4)
- [ ] Include variables $min_{\hat N}^{X_0}$ and $max_{\hat N}^{X_u}$ for each individual initial and unsafe region; applies only when multiple initial and/or unsafe regions are given (effort 2/performance +4)
- [ ] Allow to define dimension-wise independent lattice densities; only helps if no. of frequencies can also be specified to be dimension-wise differently (effort 2/performance +3)
- [ ] Use global optimizer instead of LP solver; operate directly on nonlinear (in x) trigonometric barrier conditions (effort 4/performance +5)
- [ ] Partition state space and define variables $min_{\Delta}$ and $max_{\hat N}^{X}$ for each local partition to make the conditions less conservative (effort 5/performance +5)

### Documentation

- [ ] Add docstrings to all functions

### Distribution

- [ ] Publish the package to PyPI with a wheel for Linux, Windows and MacOS

## Tests

- [x] Use the LP in a white-box environment to check whether it can produce the barriers
- [x] Feature map visualisation
- [x] Add fill_area of tolerance over the barrier
- [x] Plot f_det vs estimator
- [ ] Write down formulas on overleaf

- [x] Linear fourier probabilities
- [x] Alglib constraint on eta
- [ ] Add report button
- [ ] Make it run on a server
- [x] Bug with assertion in main
- [ ] Finalise styling
- [ ] Use a [custom Plotly bundle](https://github.com/plotly/plotly.js/blob/master/CUSTOM_BUNDLE.md) to [reduce the final bundle size](https://github.com/plotly/react-plotly.js#customizing-the-plotlyjs-bundle)
- [ ] Add timer

- [x] Add simple Polytopic Sets
- [x] Add other sets
- [x] tool paper aaai vs this different one
- [ ] Make gurobi builds work on CI/CD
- [ ] Align all benchmarks configurations
- [ ] Improve GUI results (time)
- [ ] Add a kill button

- [ ] Remove dependency on system installation of Gurobi, rather depend on a gurobipy version we know it's compatible

### How to do things right

- [x] Create periodic domain
- [x] Create a lattice on the periodic domain (user provided resolution)
- [x] Pass this to the barrier, along with all the other sets (Bounds, Initial, Unsafe)
- [x] Filter the points that fall in each set, possibly enlarged (strong suggestion, something like 10%)
  - [x] Add `change_size` method to sets
  - [ ] Should have a check to verify when some Initial and Usafe sets intersect
  - [ ] Notice that enlarging the sets may require handling the wrapping (pacman-stype)
- [x] These are the points that will go into the LP
  - The feature map must be evaluated on all points
  - Possibly the estimator too!
- [x] The points that are NOT into any of the sets are used to compute additional coefficients via PSO-cpp. See the Python script
- [ ] Use the H matrix!!!
