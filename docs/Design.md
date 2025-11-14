# Design

## UML

```mermaid
---
  config:
    class:
      hideEmptyMembersBox: true
---
classDiagram
    class Tuner {
        <<abstract>>
        +tune(Estimator, Matrix inputs, Matrix outputs)
    }
    class GridSearchTuner {
        -parameters: ParameterValues[]
        +GridSearchTuner(ParameterValues[])
    }
    class BFGSTuner {
        - lb: Vector
        - ub: Vector
        +BFGSTuner(lb, ub)
    }
    class MedianHeuristicTuner {
    }

    class Parametrizable {
        <<interface>>
        +get(HyperParameter parameter) [int | double | Vector]
        +set(HyperParameter parameter, [int | double | Vector] value)
    }
    class GradientOptimizable {
        <<interface>>
        +gradient() Vector
        +objective_value() doube
        +objective_function(Vector x, Vector gradient)
    }
    class Estimator {
        <<abstract>>
        Estimator(Tuner)
        +fit(Matrix inputs, Matrix outputs)
        +fit(Matrix inputs, Matrix outputs, Tuner)
        +predict(Matrix inputs) Matrix
        +score(Matrix inputs, Matrix outputs) double
        +clone() Estimator
    }
    class KernelRidgeRegressor {
        -kernel: Kernel
        -coefficients: Matrix
        -regularisation: double
        +KernelRidgeRegressor(Kernel, double regularisation)
    }
    class GaussianProcess

    class Kernel {
        <<abstract>>
        +apply(Vector x, Vector y) double
        +clone() Kernel
    }
    class GaussianKernel {
        -gamma: double
        -sigma_l: double
        -sigma_f: double
        +GaussianKernel(double sigma_l, double sigma_f)
    }
    class LinearKernel

    class GramMatrix {
        -gram_matrix: Matrix
        +GramMatrix(Kernel, Matrix initial_states)
        +inverse_mult(Matrix state) Matrix
    }

    Tuner <|-- GridSearchTuner
    Tuner <|-- BFGSTuner
    Tuner <|-- MedianHeuristicTuner
    Estimator o-- Tuner

    Parametrizable <|.. Kernel
    Parametrizable <|.. Estimator

    Estimator <|-- GaussianProcess
    Estimator <|-- KernelRidgeRegressor


    Kernel <|-- LinearKernel
    Kernel <|-- GaussianKernel

    GradientOptimizable <|.. KernelRidgeRegressor
    GradientOptimizable <|.. GaussianKernel
    KernelRidgeRegressor --> GramMatrix
    KernelRidgeRegressor o-- Kernel

```

## LP dependency graph

```mermaid
flowchart TD
  X("$$\mathcal{X}$$")
  Xu("$$\mathcal{X}_u$$")
  X0("$$\mathcal{X}_0$$")
  XsX("$$\mathcal{\tilde{X}} \setminus \mathcal{X}$$")
  XsX0("$$\mathcal{\tilde{X}} \setminus \mathcal{X}_0$$")
  XsXu("$$\mathcal{\tilde{X}} \setminus \mathcal{X}_u$$")
  D("$$\phi(\mathcal{\tilde{Xp}}) - \phi(\mathcal{\tilde{X}})$$")
  DXsX("$$\phi(\mathcal{\tilde{Xp}} \setminus \mathcal{X}) - \phi(\mathcal{\tilde{X}} \setminus \mathcal{X})$$")
  BminX0("$$\min_{x \in \mathcal{X}_0} \phi(x)^T b $$")
  BmaxXu("$$\max_{x \in \mathcal{X}_u} \phi(x)^T b $$")
  BmaxX("$$\max_{x \in \mathcal{X}} \phi(x)^T b $$")
  BdminX("$$\max_{x \in \mathcal{X}} \phi(x)^T (Hb - b) $$")
  BminXsX("$$\min_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}} \phi(x)^T b $$")
  BmaxXsX("$$\max_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}} \phi(x)^T b $$")
  BminXsX0("$$\min_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}_0} \phi(x)^T b $$")
  BmaxXsX0("$$\max_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}_0} \phi(x)^T b $$")
  BminXsXu("$$\min_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}_u} \phi(x)^T b $$")
  BmaxXsXu("$$\max_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}_u} \phi(x)^T b $$")
  BdminsX("$$\min_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}} \phi(x)^T (Hb - b) $$")
  BdmaxsX("$$\max_{x \in \mathcal{\tilde{X}} \setminus \mathcal{X}} \phi(x)^T (Hb - b) $$")
  Asx0("$$A^{S \setminus X_0}_{\tilde{N}}$$")
  Asxu("$$A^{S \setminus X_u}_{\tilde{N}}$$")
  Asx("$$A^{S \setminus X}_{\tilde{N}}$$")
  C("$$\left(1 - \frac{2 f_{max}}{\tilde{Q}}\right)^{-n/2}$$")

  hateta("$$\hat{\eta}$$")
  hatgamma("$$\hat{\gamma}$$")
  hatdelta("$$\hat{\Delta}$$")
  hatxi("$$\hat{\xi}$$")

  eta("$$\eta$$")
  gamma("$$\gamma$$")
  delta("$$c - \varepsilon \bar{B} \kappa$$")

  XsX --> BminXsX
  XsX --> BmaxXsX
  XsX --> Asx
  XsX0 --> BminXsX0
  XsX0 --> BmaxXsX0
  XsX0 --> Asx0
  XsXu --> BminXsXu
  XsXu --> BmaxXsXu
  XsXu --> Asxu

  X0 --> BminX0
  Xu --> BmaxXu
  X --> BmaxX
  D --> BdminX
  DXsX --> BdminsX
  DXsX --> BdmaxsX

  C --> hateta
  eta --> hateta
  Asx0 --> hateta
  BminX0 --> hateta
  BminXsX0 --> hateta
  BmaxXsX0 --> hateta

  C --> hatgamma
  gamma --> hatgamma
  Asxu --> hatgamma
  BmaxXu --> hatgamma
  BminXsXu --> hatgamma
  BmaxXsXu --> hatgamma


  C --> hatdelta
  delta --> hatdelta
  Asx --> hatdelta
  BdminX --> hatdelta
  BdminsX --> hatdelta
  BdmaxsX --> hatdelta

  C --> hatxi
  Asx --> hatxi
  BmaxX --> hatxi
  BminXsX --> hatxi
  BmaxXsX --> hatxi
```

Where:

- $\mathcal{X} \subseteq \mathbb{R}^{n}$: State space we are interested in modeling.
- $\mathcal{X}_0 \subseteq \mathcal{X}$: Initial subset of the state space.
- $\mathcal{X}_u \subseteq \mathcal{X}$: Unsafe subset of the state space.
- $\phi : [0, 1]^n \to \mathbb{R}^{2m+1}$ Truncated Fourier feature map.
    - $\phi(x) = \begin{bmatrix} w_0 &  \sqrt{2} w_1\cos(\omega_1^T P(x)) & \sqrt{2} w_1 \sin(\omega_1^T P(x)) & \dots & \sqrt{2} w_m\cos(\omega_m^T P(x)) & \sqrt{2} w_m \sin(\omega_m^T P(x)) \end{bmatrix}^T$ where $P$ is simply a map from $\mathcal{X}$ to $[0, 1]^n$.
- $\mathcal{\tilde{X}} \subseteq \mathbb{R}^n$: State space with the property of being the smallest subset of $\mathbb{R}^n$ on which $\phi$ is periodic. 
- $b \in \mathbb{R}^{2m+1}$ Weight vector. It contains the decision variables of the LP.
- $K \in \mathbb{R}^{(2m+1) \times(2m+1)}$ Gram matrix. It is computed as $K_{ij} = \sum_{k=1}^{n} \phi_i(x_k) \phi_j(x_k)$, where $x_k$ are the training samples.

## Sequence Diagram

```mermaid
sequenceDiagram
    create participant Bounds
    actor User

    Note over User: Initialising environment
    User ->> Bounds: sample()
    destroy Bounds
    Bounds ->> User: return training_inputs
    User ->> User: f(state) -> training_outputs

    Note over User: Initialising estimator
    create participant Tuner
    User ->> Tuner: new(params)
    Tuner ->> User: return Tuner
    create participant Estimator
    User ->> Estimator: new(kernel, tuner)

    Note over User: Training the estimator
    User ->>+ Estimator: fit(training_inputs, training_outputs)
    Estimator ->>+ Tuner: tune(Estimator, training_inputs, training_outputs)
    loop Tuning
        Tuner ->> Estimator: set(parameter, value)
        Tuner ->> Estimator: update(training_inputs, training_outputs)
        Tuner ->> Estimator: score(training_inputs, training_outputs)
    end
    Tuner ->>- Estimator: set(parameter, best_value)
    Estimator ->>- User: return estimator

    Note over User: Predict with the estimator
    User ->>+ Estimator: predict(new_inputs)
    Estimator ->>- User: return predictions
```

## Idea

```mermaid
---
config:
  layout: dagre
  theme: neo
---
flowchart TD
 subgraph Dyn["Open-Loop System"]
        A["Agent"]
        Env["Environment"]
  end
 subgraph Sys["Closed-Loop System"]
        Dyn
        C["Controller"]
  end
 subgraph lucid["LUCID (what we are building)"]
        WM["Kernel-Based <br> World Model"]
        Syn["Syn"]
  end
 subgraph Syn["Synthesis Engine"]
        Ver["Verification Engine"]
        RL["Controller Learner"]
  end
    Sys -- Behaviour <br> Data --> WM
    C <-- Interaction --> A
    Env <-- Interaction --> A
    WM -- Efficient and Compressed Dynamics Representation --> Ver
    RL -- Candidate Controller --> Ver
    Syn -- "Deploy Robust <br> Controller" --> C
    Ver -- Feedback --> RL
    Spec["Specification Block"] -- Automaton --> Ver
    A["Agent"]
    Env["Environment"]
     A:::Rose
     Env:::Sky
    classDef Rose stroke-width:1px, stroke-dasharray:none, stroke:#FF5978, fill:#FFDFE5, color:#8E2236
    classDef Sky stroke-width:1px, stroke-dasharray:none, stroke:#374D7C, fill:#E2EBFF, color:#374D7C

subgraph V["Synthesis Engine via Control Barrier (merged verification and controller learner)"]
    START1:::hidden -- Efficient and Compressed Dynamics Representation --> EO
    EO["Expectation Oracle (conditional on policy)"] <-- Candidate Solution (Barrier + Control Policy) --> Opt
    START3:::hidden -- Parametrised Control Policy Template --> Opt
    START4:::hidden -- Parametrised Barrier Template --> Opt
    Opt["Nonlinear Tuner"] -- Valid Barrier + Control Policy + SatProb OR Unsat -->END:::hidden
    START2:::hidden -- Automaton --> Opt
end
```
