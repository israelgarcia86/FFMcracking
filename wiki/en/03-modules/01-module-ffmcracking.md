# 01. Central Module: FFMcracking.py

Navigation: [Wiki Home](../index.md) > [03. Modules](01-module-ffmcracking.md) > Central Module

================================================================================
1. MODULE PURPOSE
================================================================================

The source file `src/FFMcracking.py` is the operational core of the framework. It houses the data structures that parametrize the physical problem and the orchestrator function `Compute_crit_factor_FFM()`, which interacts with the Abaqus CAE environment and evaluates the Coupled Criterion.

================================================================================
2. DATA CLASSES
================================================================================

Three decoupled data classes define model inputs:

2.1. FFM_Material Class
--------------------------------------------------------------------------------
Encapsulates material fracture properties and sets XFEM damage suppression constants:

```python
class FFM_Material:
    def __init__(self, sigmac, tauc, Gc1, Gc2):
        self.sigmac = sigmac        # Tensile strength (Mode I)
        self.tauc = tauc            # Shear strength (Mode II)
        self.Gc1 = Gc1              # Critical fracture toughness (Mode I)
        self.Gc2 = Gc2              # Critical fracture toughness (Mode II)
        self.maxsigma = 5000e10 * sigmac      # Artificial XFEM initiation stress
        self.maxd = 1.0e10 * Gc1 / sigmac     # Artificial XFEM displacement limit
```

The XFEM geometric separator concept:
Abaqus requires damage initiation (MaxpsDamageInitiation) and evolution laws to activate an XFEM crack. However, in Finite Fracture Mechanics, Abaqus must NOT govern crack evolution via cohesive zone laws; all failure physics is handled by the Coupled Criterion in post-processing.

FFMcracking solves this by assigning an artificially elevated damage initiation threshold `maxsigma` (50,000 billion times sigmac). Consequently, stress never reaches this value, damage evolution is never triggered, and XFEM functions strictly as an elastic, traction-free geometric cut without conformal remeshing.

Mixed Mode note:
In the current release, `sigmac` and `Gc1` are active in calculations (pure Mode I). The attributes `tauc` and `Gc2` are defined in the interface to support future mixed-mode formulations.

2.2. FFM_Crack Class
--------------------------------------------------------------------------------
Defines the spatial geometry of the candidate crack:

```python
class FFM_Crack:
    def __init__(self, tips_coordinates):
        self.tips_coordinates = tips_coordinates
```

- `tips_coordinates`: A list of two tuples containing 3D coordinates of crack endpoints: `[(x1, y1, z1), (x2, y2, z2)]`.
- Even though current models are 2D, coordinates must specify z = 0.0 because Abaqus `Path` objects require 3D inputs.

2.3. FFM_InputModel Class
--------------------------------------------------------------------------------
Bundles model configuration parameters and provides helper accessors:

```python
class FFM_InputModel:
    def __init__(self, inpFilePath, inpFileName, nameStep, outRoute, material):
        self.inpFilePath = inpFilePath
        self.inpFileName = inpFileName
        self.nameStep = nameStep
        self.outRoute = outRoute
        self.material = material
```

Accessor methods:
- `extract_nameStep()`: Name of the load step containing loads to scale.
- `extract_outRoute()`: Output directory path.
- `extract_material()`: Returns the `FFM_Material` instance.
- `extract_inpFileTotal()`: Full path to `.inp` file.

================================================================================
3. MAIN ORCHESTRATOR: Compute_crit_factor_FFM
================================================================================

Signature:
  `Compute_crit_factor_FFM(CrackFFM, OriginalModel)`

Parameters:
- `CrackFFM` (`FFM_Crack` instance): Candidate crack geometry.
- `OriginalModel` (`FFM_InputModel` instance): Base elastic model setup.

Return:
- List of three floats: `[t_CC, t_SC, t_EC]`
  - `t_CC`: Coupled critical load factor `max(t_SC, t_EC)`.
  - `t_SC`: Stress Criterion factor.
  - `t_EC`: Energy Criterion factor.

Operational sequence:
1. Environment initialization:
   Imports Abaqus APIs (`abaqus`, `odbAccess`, `caeModules`), closes open model databases (`mdb.close()`), and switches working directory to `outRoute`.
2. Stress Criterion evaluation (Uncracked model):
   - Checks if `<jobName>-without.odb` exists. If not, imports `.inp`, defines output request `EC0`, and submits job.
   - Opens ODB, extracts stresses along crack path, and computes `t_SC = sigmac / min(sigma_nn)`.
   - Extracts boundary nodal displacements and reactions (State 1).
3. Energy Criterion evaluation (XFEM cracked model):
   - Re-imports model into fresh `mdb`.
   - Injects artificial damage thresholds (`maxsigma`, `maxd`).
   - Calls `generate_crack()` to create crack geometry and XFEM enrichment.
   - Submits `<jobName>-with` job.
   - Extracts boundary nodal displacements and reactions (State 2).
   - Computes reciprocal work `RE` via Betti's Theorem.
   - Measures effective length and computes `DE = Gc1 * length`.
   - Computes `t_EC = sqrt(DE / |RE|)`.
4. Coupling and return:
   - Computes `t_CC = max(t_SC, t_EC)`.
   - Closes all ODB and MDB databases.
   - Returns `[t_CC, t_SC, t_EC]`.

================================================================================
Navigation: [Previous: Simulation Lifecycle](../02-architecture/02-simulation-lifecycle.md) | [Wiki Home](../index.md) | [Next: Stress Extraction](02-stress-extraction.md)
================================================================================
