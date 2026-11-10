# 03. Example 2: Crack Length Optimization

Navigation: [Wiki Home](../index.md) > [04. Practical Guides](01-requirements-and-setup.md) > Example 2: Length Optimization

================================================================================
1. CASE STUDY OBJECTIVE
================================================================================

The script `samples/sample2_OpenHoleBasicOptimLenght/example2.py` illustrates the key application of Finite Fracture Mechanics: autonomously predicting the critical initiation crack jump length (lc) and the actual structural failure load (tc).

In engineering design, crack onset length is unknown a priori. The physical formulation seeks the global minimum of the coupled envelope:

  lc = argmin_{l in [L_min, L_max]} t_CC(l)

================================================================================
2. PARAMETER CONFIGURATION IN example2.py
================================================================================

Inspect `samples/sample2_OpenHoleBasicOptimLenght/example2.py`:

```python
###########################   CONFIGURATION   ################################
inpFilePath = 'C:/Users/username/FFMcracking/samples/sample2_OpenHoleBasicOptimLenght/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = 'C:/Users/username/FFMcracking/samples/sample2_OpenHoleBasicOptimLenght/compFiles/'

# Material failure properties
sigmac = 70.0    # Tensile strength (MPa)
Gc1 = 0.4        # Critical Mode I fracture toughness (N/mm)

# Search bounds for initiation length L (mm)
L_min = 1.0e-1   # 0.1 mm
L_max = 5.0      # 5.0 mm
bounds = (L_min, L_max)

# Number of Bayesian optimization iterations
n_iterations = 10
################################################==============================
```

Objective function definition:
```python
def objective(L):
    # Crack initiates at hole edge (x=1.0) and advances length L
    coordinates_1 = [(1.0, 0.0, 0.0), (1.0 + L, 0.0, 0.0)]
    Crack1 = FFM_Crack(coordinates_1)
    # Evaluates coupled factor in Abaqus
    return Compute_crit_factor_FFM(Crack1, Model1)[0]
```

================================================================================
3. RUNNING THE OPTIMIZATION
================================================================================

Run the script from your terminal using Abaqus Python:

```powershell
abaqus cae noGUI=example2.py
```

Execution timeline:
1. Initial sampling (5 evaluations):
   Evaluates 5 randomly chosen lengths across [0.1, 5.0].
   - The uncracked job (`-without.odb`) runs on the first evaluation.
   - Subsequent evaluations read `-without.odb` directly from cache.
2. Bayesian loop (10 iterations):
   - Fits the Gaussian Process model to accumulated observations.
   - Expected Improvement (EI) identifies the most promising candidate length.
   - Submits cracked XFEM job in Abaqus.
   - Updates Gaussian Process mean and variance estimates.
3. Convergence:
   After 15 total FE evaluations, returns the predicted global minimum.

================================================================================
4. RESULTS ANALYSIS
================================================================================

Terminal output summarizes convergence:

```text
--- FFM OPTIMIZATION RESULTS ---
Critical initiation length (lc): 1.14 mm
Critical failure load factor (tc): 1.628
```

Physical engineering insights:
1. Critical length lc = 1.14 mm:
   The structure does not fail by forming a submicron microcrack or an immediate full severance; it initiates with an instantaneous finite jump of 1.14 mm.
2. Critical load factor tc = 1.628:
   The reference load applied in the `.inp` file must be multiplied by 1.628 to trigger crack initiation.
3. At the optimum point:
   t_SC(1.14) approx t_EC(1.14) approx 1.628. Component capacity is governed by the simultaneous balance of tensile strength and fracture toughness.

Regression Testing Tip:
Before making any changes to the source code, run and record results from `example1.py` and `example2.py`. They provide an immediate baseline to ensure refactoring preserves physical results.

================================================================================
Navigation: [Previous: Example 1](02-example-1-fixed-crack.md) | [Wiki Home](../index.md) | [Next: Troubleshooting and FAQ](../05-faq-and-development/01-troubleshooting-faq.md)
================================================================================
