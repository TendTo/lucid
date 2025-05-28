# Features

This section describes briefly the features supported by Lucid.

## Legend

The color legend below describes the status of each feature in the library.

```mermaid
flowchart TB
    a["Available"]:::available
    i["In active development"]:::in-development
    p["Planned for the future"]:::planned
    n["Not planned"]:::not-planned

classDef available stroke:#0b0,stroke-width:2px;
classDef in-development stroke:#bb0,stroke-width:2px,stroke-dasharray:2px;
classDef planned stroke:#0bb,stroke-width:2px,stroke-dasharray:5px;
classDef not-planned stroke:#b00,stroke-width:2px,stroke-dasharray:15px;
```

## Estimators

```mermaid
flowchart TB
    Estimator["Estimator"]
    u{{"Unsupervised"}}:::not-planned
    c{{"Classification"}}:::not-planned
    r{{"Regression"}}
    s{{"Supervised"}}
    p{{"Parametric"}}:::not-planned
    np{{"Non parametric"}}
    GP["Gaussian Process"]:::planned
    KR["Kernel Ridge"]:::available
    SVR["SVR"]:::planned

    Estimator --> s
    Estimator --> u
    s --> r
    s --> c
    r ---> np
    r --> p
    np --> KR
    np --> SVR
    np --> GP

classDef available stroke:#0b0,stroke-width:2px;
classDef in-development stroke:#bb0,stroke-width:2px,stroke-dasharray:2px;
classDef planned stroke:#0bb,stroke-width:2px,stroke-dasharray:5px;
classDef not-planned stroke:#b00,stroke-width:2px,stroke-dasharray:15px;
```

### Definitions

- **Unsupervised**: Estimators that do not require labeled data for training.
- **Supervised**: Estimators that require labeled data for training.
- **Classification**: Estimators that predict discrete labels or categories.
- **Regression**: Estimators that predict continuous values.

### Formulae

#### Kernel Ridge

A regression technique that combines ridge regression with kernel methods.

##### Loss Function

$$
\ell(w) = \frac{1}{2} \|Y - w^T\Phi(X)\|^2_2 + \frac{1}{2} \lambda n \|w\|^2_2
$$

where:

- $Y$ is the target vector,
- $X$ is the input data,
- $\Phi(X)$ is the feature map over the input data,
- $w$ is the weight vector we want to learn,
- $\lambda$ is the regularization parameter,
- $n$ is the number of samples.

To obtain the optimal weights, we solve the following equation:

$$
w = \Phi(X)(\Phi(X)^T\Phi(X) + \lambda n I)^{-1}Y
$$

##### Prediction

To predict a new output $\hat{y}$ for a new input $x$, we use the following formula:

$$
\hat{y} = w^T \Phi(x) = y(\Phi(X)^T\Phi(X) + \lambda n I)^{-1}\Phi(x)^T\Phi(X) = y(K + \lambda n I)^{-1}\kappa(x)
$$

where $K$ is the Gram matrix computed from the training data, and $\kappa(x)$ is the kernel function applied to the new input $x$ against all training inputs.

## Supported Kernels

```mermaid
flowchart TB
    gaussian["Gaussian"]:::available
    l["Linear"]:::planned
    p["Polynomial"]:::planned
    s["Sigmoid"]:::planned

classDef available stroke:#0b0,stroke-width:2px;
classDef in-development stroke:#bb0,stroke-width:2px,stroke-dasharray:2px;
classDef planned stroke:#0bb,stroke-width:2px,stroke-dasharray:5px;
classDef not-planned stroke:#b00,stroke-width:2px,stroke-dasharray:15px;
```
