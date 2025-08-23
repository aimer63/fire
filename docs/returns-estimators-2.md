# Estimation of $\mu$ and $\sigma$ for Normal Distribution of Log Returns Using Sample Mean and Standard Deviation of observed Returns Rates

Given only the sample mean $\bar{R}$ and sample standard deviation $s_R$ of the return rate
data points $R_1, R_2, \ldots, R_n$, where $X_t = 1 + R_t$ is lognormally distributed for
the theoretical continuous process $R_t$, we know that $Y_t = \log(X_t) = \log(1 + R_t)$
follows a normal distribution $N(\mu, \sigma^2)$. We aim to derive estimators for
$\mu$ and $\sigma$ using $\bar{R}$ and $s_R$.

## Setup

- **Given Statistics**:
  - Sample mean: $\bar{R} = \frac{1}{n} \sum_{i=1}^n R_i$, estimating $E[R_t]$.
  - Sample standard deviation: $s_R = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (R_i - \bar{R})^2}$,
  with sample variance $s_R^2$ estimating $\text{Var}(R_t)$.
- **Theoretical Process**: $R_t$ is the theoretical continuous return rate process,
where $X_t = 1 + R_t$ is lognormally distributed.
- **Transformation**: $Y_t = \log(1 + R_t) \sim N(\mu, \sigma^2)$.
- **Goal**: Derive exact estimators $\hat{\mu}$ and $\hat{\sigma}$ using $\bar{R}$ and $s_R$.

## Derivation of Estimators

### Lognormal Properties

Since $1 + R_t$ is lognormal, if $Y_t = \log(1 + R_t) \sim N(\mu, \sigma^2)$, then:

- Expected value:

```math
  E[1 + R_t] = e^{\mu + \sigma^2/2} \implies E[R_t] = e^{\mu + \sigma^2/2} - 1.
```

- Variance:

```math
  \text{Var}(R_t) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}.
```

The sample statistics provide:

- $\bar{R} \approx E[R_t] = e^{\mu + \sigma^2/2} - 1$.
- $s_R^2 \approx \text{Var}(R_t) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$.

### Estimator for $\sigma^2$

From the mean equation:

```math
\bar{R} \approx e^{\mu + \sigma^2/2} - 1 \implies e^{\mu + \sigma^2/2} = \bar{R} + 1.
```

Square both sides:

```math
e^{2\mu + \sigma^2} = (\bar{R} + 1)^2.
```

Substitute into the variance equation:

```math
s_R^2 \approx (e^{\sigma^2} - 1) (\bar{R} + 1)^2.
```

Solve for $e^{\sigma^2}$:

```math
e^{\sigma^2} - 1 = \frac{s_R^2}{(\bar{R} + 1)^2} \implies e^{\sigma^2} = 1 + \frac{s_R^2}{(\bar{R} + 1)^2}.
```

Thus:

```math
\hat{\sigma}^2 = \log\left( 1 + \frac{s_R^2}{(\bar{R} + 1)^2} \right).
```

### Estimator for $\mu$

From the mean equation:

```math
e^{\mu + \sigma^2/2} = \bar{R} + 1 \implies \mu + \frac{\sigma^2}{2} = \log(\bar{R} + 1).
```

Solve for $\mu$:

```math
\mu = \log(\bar{R} + 1) - \frac{\sigma^2}{2}.
```

Substitute $\hat{\sigma}^2$:

```math
\hat{\mu} = \log(\bar{R} + 1) - \frac{1}{2} \log\left( 1 + \frac{s_R^2}{(\bar{R} + 1)^2} \right).
```

### Estimator for $\sigma$

The standard deviation estimator is:

```math
\hat{\sigma} = \sqrt{\hat{\sigma}^2} = \sqrt{\log\left( 1 + \frac{s_R^2}{(\bar{R} + 1)^2} \right)}.
```

## Final Estimators

The exact estimators for $N(\mu, \sigma^2)$ of $Y_t = \log(1 + R_t)$ are:

- **Mean**:

```math
  \hat{\mu} = \log(\bar{R} + 1) - \frac{1}{2} \log\left( 1 + \frac{s_R^2}{(\bar{R} + 1)^2} \right).
```

- **Variance**:

```math
  \hat{\sigma}^2 = \log\left( 1 + \frac{s_R^2}{(\bar{R} + 1)^2} \right).
```

- **Standard Deviation**:

```math
  \hat{\sigma} = \sqrt{\log\left( 1 + \frac{s_R^2}{(\bar{R} + 1)^2} \right)}.
```

These estimators use only $\bar{R}$ and $s_R$, derived from the lognormal properties of $1 + R_t$.

---
