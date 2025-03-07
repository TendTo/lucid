# To do

## Work package 1

### Activity 1

- [x] Complete initial implementation reproducing the results of the paper for the first test case
- [ ] Rename variables
- [ ] Fix current implementation correctness with integration testing
- [ ] Implement second test case

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
