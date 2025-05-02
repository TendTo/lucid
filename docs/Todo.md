# To do

## Work package 1

### Activity 1

- [x] Complete initial implementation reproducing the results of the paper for the first test case
- [x] Rename variables
- [x] Fix current implementation correctness with integration testing
- [x] Implement second test case
- [x] Implement a proper fftn
- [x] Check whether it is possible to avoid the padding to get the same information
    - If it is, implement the alternative
    - [x] Otherwise, implement the padding
- [x] Handle permutations
- [x] Keep track of permutations
    - Is it better to do a lazy permutation, carrying the updated strides and axes (and dims?)
    - [x] Or just eagerly update the data structure?

### Activity 2

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
        e1(["(E1) Inductive k barriers"])
        m2>"(M2) Control synthesis"]
        e2(["(E2) Reachability specifications"])
    end
    a1 --> a2 --> a3 --> a4
    wp1 --> m1
```


## Notes

- Expectation oracle: given a set of pairs (input, output), produce an oracle such that $\mathbb{E}_{x^+\approx\bar{t}(\cdot|x,u)}[\cdot(x^+)] = y$ for all pairs $(x,y)$