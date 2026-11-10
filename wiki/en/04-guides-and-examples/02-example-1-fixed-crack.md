# 02. Example 1: Fixed Crack

Navigation: [Wiki Home](../index.md) > [04. Practical Guides](01-requirements-and-setup.md) > Example 1: Fixed Crack

================================================================================
1. CASE STUDY OBJECTIVE
================================================================================

The script `samples/sample1_OpenHoleCrackFixed/example1.py` solves the direct Coupled Criterion problem: calculating critical load factor for a crack of predetermined position and length.

The finite element model is an open-hole plate subjected to uniaxial tension (`OpenHoleLongDEF-fmesh-00.inp`):
- Hole radius: R = 1.0 mm.
- Candidate crack: Straight segment starting at the hole edge (1.0, 0.0, 0.0) and extending horizontally along the ligament to (4.5, 0.0, 0.0).
- Evaluated length: L = 3.5 mm.

================================================================================
2. PARAMETER CONFIGURATION IN example1.py
================================================================================

Open `samples/sample1_OpenHoleCrackFixed/example1.py` and inspect configuration settings:

```python
###########################   CONFIGURATION   ################################
# 1. Path definitions
inpFilePath = 'C:/Users/username/FFMcracking/samples/sample1_OpenHoleCrackFixed/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = 'C:/Users/username/FFMcracking/samples/sample1_OpenHoleCrackFixed/compFiles/'

# 2. Material mechanical and fracture properties
sigmac = 70.0    # Tensile strength (MPa)
Gc1 = 0.4        # Critical Mode I fracture toughness (N/mm)

# 3. Fixed crack coordinates (length L = 3.5 mm)
coordinates_1 = [(1.0, 0.0, 0.0), (4.5, 0.0, 0.0)]
################################################==============================
```

Path notes:
- Always ensure `outRoute` ends with a trailing separator `/` or `\\`.
- If `compFiles/` does not exist, FFMcracking creates it automatically.

================================================================================
3. RUNNING THE SIMULATION
================================================================================

Open a command prompt in `samples/sample1_OpenHoleCrackFixed/` and execute:

On Windows (PowerShell / CMD):
```powershell
abaqus cae noGUI=example1.py
```

On Linux:
```bash
abaqus cae noGUI=example1.py
```

Internal sequence:
1. Imports `FFMcracking.py` and builds `FFM_Material`, `FFM_Crack`, and `FFM_InputModel`.
2. Calls `Compute_crit_factor_FFM()`.
3. Runs uncracked elastic analysis and writes `OpenHoleLongDEF-fmesh-00-without.odb`.
4. Extracts normal stresses on [1.0, 4.5] and computes `t_SC`.
5. Injects XFEM crack and runs `OpenHoleLongDEF-fmesh-00-with.odb`.
6. Evaluates Betti reciprocal work RE and dissipated energy DE, computing `t_EC`.
7. Couplings both factors and displays final output.

================================================================================
4. RESULTS INTERPRETATION
================================================================================

At completion, the terminal displays the calculated triplet:

```text
[k_CC, k_SC, k_EC] = [1.8742, 1.8742, 1.3210]
```
(Exact values depend on mesh resolution and input parameters).

Physical interpretation:
- `k_SC = 1.8742`: Multiplier required for normal stress at the outer tip (x = 4.5) to reach 70 MPa.
- `k_EC = 1.3210`: Multiplier required for the system to release enough potential energy to create 3.5 mm of crack surface (DE = 0.4 * 3.5 = 1.4 mJ).
- `k_CC = 1.8742`: Governed by `max(k_SC, k_EC)`. For this specific 3.5 mm length, the stress condition is limiting; at any lower load, the crack cannot reach 3.5 mm even though sufficient energy is available.

================================================================================
Navigation: [Previous: Requirements and Setup](01-requirements-and-setup.md) | [Wiki Home](../index.md) | [Next: Example 2 Length Optimization](03-example-2-length-optimization.md)
================================================================================
