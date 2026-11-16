# Bayesian optimization

`src/modOptimization.py` searches for the lowest observed coupled factor within a crack-length interval. Its function `bayesian_optimization(obj_func, bounds, n_iterations, n_initial=5)` expects `obj_func` to return `[t_CC, t_SC, t_EC]` and returns the best length, its factor, and the evaluated lengths and factors.

## Procedure

1. Generate `n_initial` uniformly distributed random lengths in the interval and evaluate each one.
2. For each subsequent iteration, predict the stress and energy criteria separately with two Gaussian processes. Each uses an RBF kernel with length scale 1.0 and diagonal regularization of `1e-6`.
3. At each candidate, the code takes the maximum of the two means and estimates a combined spread from the maximum of their lower bounds `mean - standard deviation`. It uses these values in an expected-improvement acquisition with `xi = 0.01`.
4. Select the highest acquisition among 100 equally spaced candidates, evaluate the objective and append the observations.

## Surrogate and acquisition

For each criterion, the kernel is `k(x, x') = exp(-(x - x')² / (2 · 1.0²))`. With covariance matrix `K` plus `1e-6` on its diagonal, the code calculates mean `μ = k*ᵀ K⁻¹ y` and variance `v = max(0, k(x*, x*) - k*ᵀ K⁻¹ k*)` at each candidate.

Let `μ_SC`, `μ_EC` and `σ_SC`, `σ_EC` be the two predicted means and standard deviations. The code uses `μ = max(μ_SC, μ_EC)` and `σ = |μ - max(μ_SC - σ_SC, μ_EC - σ_EC)|`. With the best observed factor `y_best` and `ξ = 0.01`, it sets `z = (y_best - μ - ξ) / σ` when `σ > 0`, and `z = 0` otherwise. The acquisition is `EI = (y_best - μ - ξ) Φ(z) + σ φ(z)`, where `Φ` and `φ` are the standard normal cumulative distribution and density. This spread is the approximation used by the implementation; it is not necessarily the exact uncertainty of the maximum of the two processes.

Example 2 uses five initial points and ten further iterations. The search returns the best **observed** value; it does not guarantee a global minimum or check convergence. Initial points are unseeded, so runs can differ. The kernel and candidate grid also affect exploration. The code uses `math`, `random` and NumPy.

[Index](../index.md) · [Example 2](../04-guides-and-examples/03-example-2-length-optimization.md)
