
<!-- markdownlint-disable -->
<!-- prettier-ignore -->
# Derivations for the Log-normal Distribution of Asset Returns

This document outlines two common justifications for modeling asset returns as log-normally distributed.

---

## 1. Discrete-Time Model (via Central Limit Theorem) 📈

This approach builds from the idea of compounding returns over discrete time intervals (e.g., daily).

* **Multiplicative Compounding**: An asset's final price, $P_T$, is the initial price, $P_0$, multiplied by a series of periodic gross returns $(1+R_t)$:

```math
    \frac{P_T}{P_0} = \prod_{t=1}^{T} (1+R_t)
```

* **Conversion to a Sum**: By taking the natural logarithm, the product becomes a sum. Let $r_t = \ln(1+R_t)$ be the continuously compounded return for period $t$.

```math
    \ln\left(\frac{P_T}{P_0}\right) = \sum_{t=1}^{T} \ln(1+R_t) = \sum_{t=1}^{T} r_t
```

* **Central Limit Theorem (CLT)**: If the individual returns $r_t$ are assumed to be independent and identically distributed (i.i.d.) random variables, the CLT states that for a large number of periods $T$, their sum will be approximately **normally distributed**.

```math
    \ln\left(\frac{P_T}{P_0}\right) \sim \mathcal{N}(\mu_{sum}, \sigma_{sum}^2)
```


* **Conclusion**: By definition, if the logarithm of a variable is normally distributed, the variable itself is **log-normally distributed**. Therefore, the total gross return, $P_T/P_0$, is approximately log-normal.

---

## 2. Continuous-Time Model (via Black-Scholes & GBM) ⚙️

This approach is fundamental to the Black-Scholes option pricing model and assumes prices evolve continuously.

* **Geometric Brownian Motion (GBM)**: The model assumes the asset price, $S_t$, follows the stochastic differential equation (SDE):

```math
    dS_t = \mu S_t dt + \sigma S_t dW_t
```


Here, $\mu$ is the drift, $\sigma$ is the volatility, and $dW_t$ is the random increment from a Wiener process where $dW_t \sim \mathcal{N}(0, dt)$.


* **Applying Itô's Lemma**: To solve the SDE for $S_t$, we analyze the dynamics of its logarithm, $f(S_t) = \ln(S_t)$, using Itô's Lemma. This gives a simpler SDE for the log-price:

```math
    d(\ln S_t) = \left( \mu - \frac{\sigma^2}{2} \right)dt + \sigma dW_t
```

* **Integrating Over Time**: Integrating this SDE from time $0$ to $T$ yields the log-price at time $T$:

```math
    \ln(S_T) - \ln(S_0) = \left( \mu - \frac{\sigma^2}{2} \right)T + \sigma W_T
```

Since $W_T \sim \mathcal{N}(0, T)$, the entire right-hand side is a constant plus a normally distributed variable.


* **Conclusion**: The log of the future price, $\ln(S_T)$, is **normally distributed**.

```math
    \ln(\frac{S_T}{S_0}) \sim \mathcal{N}\left((\mu - \frac{\sigma^2}{2})T, \sigma^2 T \right)
```

Therefore, the gross return $S_T/S_0$ are, by definition, **log-normally distributed**.
