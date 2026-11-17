# 02. Developer Guide and Roadmap

Navigation: [Wiki Home](../index.md) > [05. FAQ and Development](01-troubleshooting-faq.md) > Developer Guide

================================================================================
1. DEVELOPMENT AND CODING STANDARDS
================================================================================

To ensure codebase stability, portability, and academic integrity, contributions to `src/` must adhere to four core rules:

1. Python 2.7 and Python 3.x Dual Compatibility:
   - Abaqus 2020-2023 embeds Python 2.7, whereas Abaqus 2024+ embeds Python 3.10+.
   - Do NOT use Python 3.6+ exclusive features (like f-strings) or Python 3.10+ syntax (like `match / case`) in `src/`.
   - Use `.format()` or `%` for string formatting, and enforce explicit floating-point division.

2. Zero External Dependency Policy:
   - `src/FFMcracking.py` and `src/modOptimization.py` must NOT depend on third-party libraries (no SciPy, Scikit-Learn, PyTorch, SymPy, or Pandas).
   - All numerical algorithms, optimization methods, and matrix operations must rely exclusively on `math`, `random`, `sys`, `os`, and `numpy`.

3. Strict Resource Management in Abaqus:
   - Always close output databases via `odb.close()` and model databases via `mdb.close()` to prevent file locks.
   - Purge intermediate scratch files (.lck, .prt, .fil) before launching jobs.

4. Generalization Across Geometries:
   - Do not implement hardcoded workarounds for specific meshes or `.inp` files.
   - Ensure coordinate transformations, boundary queries, and path operations remain geometry-agnostic.

================================================================================
2. DEVELOPMENT ROADMAP AND EXTENSION PATHS
================================================================================

Priority development directions based on code analysis and Garcia (2014):

2.1. Level 1: Core Robustness and Refactoring
--------------------------------------------------------------------------------
- Intelligent Cache Invalidation:
  Implement timestamp or hash verification for the input `.inp` file and load step name. Automatically invalidate `-without.odb` when modifications are detected.
- Path Normalization:
  Replace manual string concatenations with `os.path.join()` and `os.path.normpath()` for cross-platform portability.
- Rigorous Compression Handling:
  Enforce `min(snn) > 0` along prospective crack paths before computing pointwise stress ratios.

2.2. Level 2: Mixed-Mode Failure Formulations
--------------------------------------------------------------------------------
- Activate Shear Parameters:
  Incorporate active use of `tauc` and `Gc2` in `FFM_Material`.
- Mixed-Mode Stress Criteria:
  Implement combined stress criteria on candidate planes (Mohr-Coulomb equivalent stress or quadratic tension-shear):
  (sigma_nn / sigmac)^2 + (sigma_tn / tauc)^2 >= 1
- Mode-Dependent Toughness:
  Account for modal mixity at the crack tip using phenomenological laws (e.g., Hutchinson-Suo or power-law criteria).

2.3. Level 3: Multiparametric Optimization (Angle and Initiation Site)
--------------------------------------------------------------------------------
- Crack Initiation Angle Search:
  Include crack orientation angle theta as a primary design variable for unsymmetric notches or shear states.
- 2D/3D Bayesian Optimization:
  Extend the Gaussian Process engine in `modOptimization.py` to multivariate inputs (L, theta) to discover critical length and orientation concurrently.

2.4. Level 4: Composite Laminates and Bimaterial Interfaces
--------------------------------------------------------------------------------
- Transverse Cracking in Laminates:
  Simulate matrix transverse cracking in cross-ply laminates (detailed in Garcia, 2014, chapter 6).
- Adhesive and Interface Debonding:
  Apply the Coupled Criterion to predict debonding initiation in adhesive joints and fiber-matrix micro-scale interfaces.

================================================================================
Navigation: [Previous: Troubleshooting and FAQ](01-troubleshooting-faq.md) | [Wiki Home](../index.md)
================================================================================
