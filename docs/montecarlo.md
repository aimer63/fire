# Monte Carlo Simulation: Mathematical Foundations and Theory

## Introduction

Monte Carlo simulation is a class of computational algorithms that rely on repeated random sampling
to obtain numerical results. It is especially useful for systems with a significant degree of
uncertainty or complexity, where analytical solutions are intractable or impossible.

---

## Probability Spaces and Random Variables

Let $(\Omega, \mathcal{F}, P)$ be a probability space, where:

- $\Omega$ is the sample space,
- $\mathcal{F}$ is a $\sigma$-algebra of events,
- $P$ is a probability measure.

A **random variable** $X: \Omega \to \mathbb{R}$ is a measurable function. The expected value (mean)
of $X$ is:

$\mathbb{E}[X] = \int_{\Omega} X(\omega) \ dP(\omega)$

---

## Law of Large Numbers

Monte Carlo methods are grounded in the **Law of Large Numbers (LLN)**, which states that the sample
mean of independent, identically distributed (i.i.d.) random variables converges to the expected
value as the number of samples increases.

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with mean $\mu$:

The sample mean is

```math
\overline{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
```

Then, as $n \to \infty$:

$\overline{X}_n \to \mu$ (almost surely)

where "almost surely" denotes convergence with probability 1.

---

## Monte Carlo Estimation

Suppose we wish to estimate $\theta = \mathbb{E}[f(X)]$, where $f$ is a function and $X$ is a random
variable with known distribution. The Monte Carlo estimator is:

```math
\hat{\theta}_n = \frac{1}{n} \sum_{i=1}^n f(X_i)
```

where $X_1, \ldots, X_n$ are i.i.d. samples from the distribution of $X$.

By the LLN, $\hat{\theta}_n \to \theta$ as $n \to \infty$.

---

## Central Limit Theorem and Error Estimation

The **Central Limit Theorem (CLT)** provides a way to quantify the error of the Monte Carlo
estimator:

$\sqrt{n}(\hat{\theta}_n - \theta) \to \mathcal{N}(0, \sigma^2)$

where $\sigma^2 = \mathrm{Var}(f(X))$ and the arrow denotes convergence in distribution.

Thus, the standard error of the estimator is:

$\mathrm{SE}(\hat{\theta}_n) = \frac{\hat{\sigma}}{\sqrt{n}}$

where $\hat{\sigma}^2$ is the sample variance.

---

## Pseudorandom Number Generation

Monte Carlo simulation relies on high-quality pseudorandom number generators (PRNGs) to produce
sequences $(U_1, U_2, \ldots, U_n)$ that approximate i.i.d. samples from the uniform distribution on
$[0,1]$. These are then transformed to samples from other distributions (e.g., normal, log-normal)
using methods such as the inverse transform or Box-Muller.

---

## Example: Estimating $\pi$ with Monte Carlo

A classic example is estimating $\pi$ by simulating random points in the unit square and counting
how many fall inside the unit circle:

$\pi \approx 4 \cdot \frac{\text{Number of points inside circle}}{\text{Total number of points}}$

Let $(X_i, Y_i)$ be i.i.d. samples from $\mathrm{Uniform}(0,1)$:

```math
 I_i = 1 \: if \: X_i^2 + Y_i^2 \: \leq 1, \: 0 \;otherwise.
```

Then,

$\pi \approx 4 \cdot \frac{1}{n} \sum_{i=1}^n I_i$

---

## Applications in Finance

In finance, Monte Carlo simulation is used for:

- **Option pricing:** Estimating the expected payoff of derivatives.
- **Risk analysis:** Value-at-Risk (VaR), stress testing.
- **Portfolio simulation:** Modeling wealth evolution under stochastic returns.

---

## Monte Carlo Simulation for FIRE (Financial Independence / Early Retirement) Planning

Monte Carlo simulation is particularly well-suited for modeling the uncertainty inherent in
long-term financial planning, such as FIRE. The core idea is to simulate many possible future
scenarios for an individual's wealth, accounting for the randomness in investment returns,
inflation, expenses, and life events.

---

### Principle of FIRE Simulation with Monte Carlo

1. **Model the Financial System**

   - **Initial State:** Define the starting wealth, asset allocation, and other relevant financial
     parameters.
   - **Time Steps:** The simulation typically proceeds in discrete time steps (e.g., months or
     years).
   - **Random Variables:** At each step, model uncertain quantities such as investment returns and
     inflation as random variables drawn from specified probability distributions.

2. **Simulate Wealth Evolution**

   For each simulation run (trial):

   - **Sample Returns:** Draw random returns for each asset class (e.g., stocks, bonds) for each
     time step.
   - **Apply Returns and Expenses:** Update the portfolio value by applying returns and subtracting
     withdrawals or expenses.
   - **Adjust for Inflation:** Optionally, adjust all values to real (inflation-adjusted) terms.
   - **Track Key Events:** Monitor for events such as running out of money, reaching a target
     wealth, or other milestones.

3. **Repeat and Aggregate**

   - **Repeat:** Perform a large number of independent simulation runs (e.g., 10,000).
   - **Aggregate Results:** Collect statistics such as the probability of portfolio survival,
     distribution of final wealth, and time to depletion.

---

### Choosing Distributions and Parameters for Returns and Inflation

The effectiveness of a Monte Carlo FIRE simulation depends critically on how the probability
distributions for investment returns and inflation are chosen. Typically, these distributions and
their parameters are estimated from historical financial data or based on economic models.

- **Returns:** Asset returns (e.g., stocks, bonds) are often modeled using normal or log-normal
  distributions, with mean and standard deviation estimated from long-term historical returns. In
  some cases, more sophisticated models (e.g., fat-tailed or regime-switching distributions) may be
  used to better capture market behavior.

It is important to periodically review and update these assumptions to reflect changing economic
conditions or new data, as the choice of distributions and their parameters has a significant impact
on simulation outcomes.

---

### Pseudocode Example

```python
for sim in range(num_simulations):
    wealth = initial_wealth
    for t in range(num_years):
        # Sample annual return and inflation
        r = random_sample(return_distribution)
        i = random_sample(inflation_distribution)
        # Update wealth
        wealth = wealth * (1 + r) - annual_expenses * (1 + i)**t
        if wealth <= 0:
            record_failure(sim, t)
            break
    else:
        record_success(sim, wealth)
```

---

### Key Outputs and Interpretation

- **Success Rate:** Fraction of simulations where wealth remains positive throughout the retirement
  horizon.
- **Distribution of Final Wealth:** Provides insight into best/worst/average case outcomes.
- **Time to Depletion:** For failed simulations, how long the portfolio lasted.
- **Sensitivity Analysis:** How results change with different assumptions (e.g., withdrawal rate,
  asset allocation).

---

### Advantages of Monte Carlo for FIRE

- **Captures Uncertainty:** Models the full range of possible outcomes, not just averages.
- **Flexible:** Can incorporate complex features like variable spending, market shocks, or changing
  asset allocations.
- **Probabilistic Results:** Provides probabilities and risk measures, aiding better
  decision-making.

---

### Limitations

- **Model Risk:** Results depend on the accuracy of input distributions and assumptions.
- **Computational Cost:** Large numbers of simulations may be needed for stable estimates.
- **Not Predictive:** Provides a range of possible futures, not a single forecast.

---

### Summary

Monte Carlo simulation is a robust and flexible tool for FIRE planning, allowing individuals to
assess the probability of financial success under uncertainty. By simulating thousands of possible
futures, it provides a probabilistic foundation for retirement decisions.

---

## References

- Robert, C. P., & Casella, G. (2005). _Monte Carlo Statistical Methods_. Springer.
- Glasserman, P. (2004). _Monte Carlo Methods in Financial Engineering_. Springer.
- Wikipedia: [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
