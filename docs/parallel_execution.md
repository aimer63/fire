# Parallel Execution in firecast

## Overview

firecast achieves parallel execution of simulations using Python's
`concurrent.futures.ProcessPoolExecutor`. This allows multiple independent simulation runs (e.g.,
for Monte Carlo analysis) to be distributed across all available CPU cores, significantly speeding
up computation.

## How Parallelism Works

- Each simulation run is independent and can be executed in a separate process.
- The main process submits simulation jobs to a process pool, and each worker process runs its own
  simulation instance.
- This design is especially effective for large numbers of simulations, as each process can utilize
  a separate CPU core.

## Random Number Generation

- For correlated asset/inflation draws, `np.random.Generator.multivariate_normal` is used with a
  per-process random generator (`np.random.default_rng(self.seed)`), ensuring reproducibility and
  independence between processes.

## Caveat: Multivariate Draws and BLAS/LAPACK Threading

When using correlated draws (`multivariate_normal`), NumPy relies on underlying [BLAS/LAPACK](https://www.netlib.org/lapack/)
libraries (such as OpenBLAS or MKL) for linear algebra operations. By default, these libraries may
use all available CPU cores for a single process, which can cause:

- Only one simulation process to run at a time , even when using multiprocessing.
- Severe performance degradation in parallel workloads.

### Solution

**You must limit BLAS/LAPACK to a single thread per process** to achieve true parallelism.  
Set the following environment variables before running your simulation script:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
