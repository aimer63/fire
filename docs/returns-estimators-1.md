
# Estimation of $\mu$ and $\sigma$ for Normal Distribution of Log Returns

Given a series of return rate data points $R_1, R_2, \ldots, R_n$, where $X_t = 1 + R_t$
is lognormally distributed for the theoretical continuous process $R_t$, we know that
$Y_t = \log(X_t) = \log(1 + R_t)$ follows a normal distribution $N(\mu, \sigma^2)$.
We aim to derive estimators for the mean $\mu$ and variance $\sigma^2$ (and standard deviation $\sigma$)
 based on the sample of observed return rates $R_1, R_2, \ldots, R_n$.

## Setup

- **Data Points**: We observe return rates $R_1, R_2, \ldots, R_n$, representing the
percentage changes in asset value (e.g., $R_i = 0.05$ for a 5% return).
- **Theoretical Process**: $R_t$ is the theoretical continuous return rate process,
where $X_t = 1 + R_t$ is lognormally distributed.
- **Transformation**: For each observed return rate, compute $Y_i = \log(1 + R_i)$,
where $Y_t \sim N(\mu, \sigma^2)$.
- **Goal**: Estimate $\mu$, $\sigma^2$ and $\sigma$ using the sample $Y_1, Y_2, \ldots, Y_n$.

## Derivation of Estimators

### Estimator for $\mu$

Since $Y_t \sim N(\mu, \sigma^2)$, the sample mean is the maximum likelihood estimator (MLE) for $\mu$:

```math
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n Y_i = \frac{1}{n} \sum_{i=1}^n \log(1 + R_i).
```

**Unbiasedness**: The estimator is unbiased because:

```math
E[\hat{\mu}] = E\left[\frac{1}{n} \sum_{i=1}^n Y_i\right] = \frac{1}{n} \sum_{i=1}^n E[Y_i] = \frac{1}{n} \cdot n \mu = \mu.
```

### Estimator for $\sigma^2$

For the variance $\sigma^2$, the unbiased sample variance is:

```math
\hat{\sigma}^2 = \frac{1}{n-1} \sum_{i=1}^n (Y_i - \hat{\mu})^2,
```

where $Y_i = \log(1 + R_i)$ and $\hat{\mu}$ is the sample mean. Substituting:

```math
\hat{\sigma}^2 = \frac{1}{n-1} \sum_{i=1}^n \left( \log(1 + R_i) - \hat{\mu} \right)^2.
```

**Unbiasedness**: The $n-1$ denominator ensures unbiasedness for normal distributions,
correcting for the degree of freedom lost in estimating $\mu$.

### Estimator for $\sigma$

To estimate the standard deviation $\sigma$, take the square root of the variance estimator:

```math
\hat{\sigma} = \sqrt{\hat{\sigma}^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^n \left( \log(1 + R_i) - \hat{\mu} \right)^2}.
```

**Note**: $\hat{\sigma}$ is not unbiased for $\sigma$ due to the nonlinearity of the square root,
but it is consistent and widely used.

## Final Estimators

The estimators for the normal distribution $N(\mu, \sigma^2)$ of $Y_t = \log(1 + R_t)$ are:

- **Mean**:

```math
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n \log(1 + R_i).
```

- **Variance**:

```math
\hat{\sigma}^2 = \frac{1}{n-1} \sum_{i=1}^n \left( \log(1 + R_i) - \hat{\mu} \right)^2.
```

- **Standard Deviation**:

```math
\hat{\sigma} = \sqrt{\frac{1}{n-1} \sum_{i=1}^n \left( \log(1 + R_i) - \hat{\mu} \right)^2}.
```

These estimators are computed using the observed return rate data points $R_1, R_2, \ldots, R_n$.

**Note**: See also [Returns Estimators 2](returns-estimators-2.md) for estimators
derivation from return rates sample mean and standard deviation.

---
