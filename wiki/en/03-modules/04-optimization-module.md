# 04. Bayesian Optimization Module

Navigation: [Wiki Home](../index.md) > [03. Modules](01-module-ffmcracking.md) > Optimization Module

================================================================================
1. PURPOSE AND MATHEMATICAL MOTIVATION
================================================================================

The source file `src/modOptimization.py` implements a 1D Bayesian Optimization algorithm designed to find the critical crack length lc that minimizes structural failure load:

  lc = argmin_{l} t_CC(l)

Why Bayesian Optimization?
1. Expensive Evaluations: Each objective evaluation requires running an Abaqus CAE finite element simulation with XFEM.
2. Non-Differentiability: The coupled objective t_CC(l) = max(t_SC(l), t_EC(l)) has a discontinuous first derivative at the optimum intersection. Gradient-based optimizers (Newton, gradient descent) fail or show instability.
3. Bounded Domain: The search is confined to [L_min, L_max], efficiently exploring the candidate space with minimal Abaqus calls.

Dependency constraint:
The module contains zero third-party dependencies (no SciPy or Scikit-Learn); it relies solely on standard Python libraries (`math`, `random`) and `numpy` to ensure full portability in Abaqus.

================================================================================
2. MATHEMATICAL COMPONENTS
================================================================================

2.1. Radial Basis Function (RBF) Kernel
--------------------------------------------------------------------------------
Measures spatial correlation between candidate crack lengths x1 and x2:

  k(x1, x2) = exp( - (x1 - x2)^2 / (2 * l_scale^2) )

In the current code, length scale is fixed: `length_scale = 1.0`.

2.2. Surrogate Model: Gaussian Process Regression (GPR)
--------------------------------------------------------------------------------
Given evaluated observations X = [x_1, ..., x_n] with outputs y = [y_1, ..., y_n]:
1. Gram Covariance Matrix:
   K_ij = k(x_i, x_j) + sigma_noise^2 * delta_ij
   where sigma_noise^2 = 1.0e-6 is a numerical jitter term preventing singular conditioning.
2. Matrix Inversion:
   K_inv = np.linalg.inv(K).
3. Prediction at candidate x_*:
   - Mean prediction:
     mu(x_*) = k_*^T * K_inv * y
   - Variance prediction:
     sigma^2(x_*) = k(x_*, x_*) - k_*^T * K_inv * k_*
   where k_* = [k(x_*, x_1), ..., k(x_*, x_n)]^T.

2.3. Acquisition Function: Expected Improvement (EI)
--------------------------------------------------------------------------------
Selects the next candidate point maximizing expected improvement over current best minimum y_best = min(y):

  EI(x) = (y_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)

where:
  z = (y_best - mu(x) - xi) / sigma(x)
- Phi(z): Standard normal cumulative distribution function (CDF), implemented via `0.5 * (1.0 + math.erf(z / sqrt(2)))`.
- phi(z): Standard normal probability density function (PDF): `exp(-0.5 * z^2) / sqrt(2 * pi)`.
- xi = 0.01: Exploration parameter balancing exploitation against high-uncertainty exploration.

================================================================================
3. MAIN ALGORITHM: bayesian_optimization
================================================================================

Signature:
  `bayesian_optimization(obj_func, bounds, n_iterations, n_initial=5)`

Execution sequence:
1. Initial sampling:
   Generates `n_initial` points uniformly distributed in [bounds[0], bounds[1]] and evaluates the objective in Abaqus.
2. Optimization loop (Iterations 1 to n_iterations):
   a. Fits the Gaussian Process to all evaluated points.
   b. Discretizes domain into 100 candidate points.
   c. Evaluates Expected Improvement (EI) across candidates.
   d. Picks candidate x_next with highest EI.
   e. Runs Abaqus simulation at x_next and stores result.
3. Return:
   Returns optimal pair (x_opt, y_opt) with lowest evaluated critical factor.

Flowchart in text:

  [Generate 5 initial sample points]
                 |
                 v
     [Evaluate obj_func in Abaqus]
                 |
                 v
  +---> [Fit Gaussian Process (K, K_inv)]
  |              |
  |              v
  |     [Grid of 100 candidate points]
  |              |
  |              v
  |   [Compute Expected Improvement (EI)]
  |              |
  |              v
  |    [Select maximum of EI: x_next]
  |              |
  |              v
  |   [Run Abaqus at x_next: y_next]
  |              |
  +-- Iteration < n_iterations? (YES)
                 | (NO)
                 v
  [Return optimal solution: (lc_opt, tc_opt)]

================================================================================
4. DESIGN OBSERVATIONS AND CURRENT LIMITATIONS
================================================================================

Analysis of the optimization engine reveals:
- Reproducibility: Initial points use `random.uniform()` without a fixed seed (`random.seed()`), so separate runs may explore slightly different paths.
- Fixed hyperparameters: Length scale l_scale = 1.0 is static; normalizing search domains to [0, 1] improves robustness across scales.
- Stopping criteria: The loop runs for a fixed number of iterations (10) without checking convergence tolerance.

================================================================================
Navigation: [Previous: XFEM Modeling](03-xfem-method-and-energy.md) | [Wiki Home](../index.md) | [Next: Requirements and Setup](../04-guides-and-examples/01-requirements-and-setup.md)
================================================================================
