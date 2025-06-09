# To do

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
- [ ] Add scorer (how to evaluate the performance of the model)

### Activity 3

- [ ] Run Automated Anaesthesia
- [ ] Run Building Automation System

### Activity 4

- [x] Median Heuristic tuner
- [ ] Grid search tuner
- [ ] LBFGS tuner
- [ ] Parallelise the hyperparameter optimisation

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
        e3(["(E3) Verification with dreal (or Z3 / CVC5)"])
        e4(["(E4) Change of the LP solver"])
        e5(["(E5) Non linear optimizer for the synthesis"])
        e6(["(E6) Guiding the optimizing process via heuristics"])
        e7(["(E7) Pipeline of kernels"])
    end
    a1 --> a2 --> a3 --> a4
    wp1 --> m1
```

## Miscellaneous

### Performance

- [ ] Use templates instead of polymorphism where possible
- [ ] Parallelization via OpenMP and/or GPU

### Documentation

- [ ] Add docstrings to all functions

### Distribution

- [ ] Publish the package to PyPI with a wheel for Linux, Windows and MacOS

## Notes

- Expectation oracle: given a set of pairs (input, output), produce an oracle such
  that $\mathbb{E}_{x^+\approx\bar{t}(\cdot|x,u)}[\cdot(x^+)] = y$ for all pairs $(x,y)$


## Tests

- [ ] Use the LP in a white-box environment to check whether it can produce the barriers
- [x] Feature map visualisation
- [x] Add fill_area of tolerance over the barrier
- [x] Plot f_det vs estimator
- [ ] Write down formulas on overleaf
