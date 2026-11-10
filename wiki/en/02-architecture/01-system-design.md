# 01. General System Design

Navigation: [Wiki Home](../index.md) > [02. Architecture](01-system-design.md) > System Design

================================================================================
1. LAYERED ARCHITECTURE
================================================================================

FFMcracking is structured into four decoupled layers with distinct responsibilities. This ensures problem setup, numerical optimization, fracture mechanics routines, and Abaqus execution operate independently.

Layer architecture in text:

  +---------------------------------------------------------------+
  | 1. APPLICATION LAYER (samples/example1.py, example2.py)       |
  |    - Specifies geometry (.inp), material (sigmac, Gc1), crack |
  +---------------------------------------------------------------+
                                 |
                                 v
  +---------------------------------------------------------------+
  | 2. OPTIMIZATION LAYER (src/modOptimization.py)                |
  |    - Gaussian Process Regression (GPR) + RBF Kernel           |
  |    - Expected Improvement (EI) minimizing t_CC(l)             |
  +---------------------------------------------------------------+
                                 |
                                 v
  +---------------------------------------------------------------+
  | 3. FFM ANALYSIS LAYER (src/FFMcracking.py)                    |
  |    - Data classes: FFM_Material, FFM_Crack, FFM_InputModel    |
  |    - Orchestrator: Compute_crit_factor_FFM()                  |
  |    - SC evaluation: extractStresses, evaluate_stress_criterion|
  |    - EC evaluation: generate_crack (XFEM), evaluate_RE, DE    |
  +---------------------------------------------------------------+
                                 |
                                 v
  +---------------------------------------------------------------+
  | 4. FINITE ELEMENT KERNEL (Abaqus CAE / Standard)              |
  |    - Model Database (mdb) and Result Database (odbAccess)     |
  |    - Linear equation system solver                            |
  +---------------------------------------------------------------+

Mermaid Diagram:
```mermaid
graph TD
    subgraph Layer1["1. Application Layer (samples/)"]
        Ex1["example1.py (Fixed Crack)"]
        Ex2["example2.py (Optimization lc)"]
    end

    subgraph Layer2["2. Optimization Layer (modOptimization.py)"]
        BO["bayesian_optimization()"]
        EI["expected_improvement()"]
        GP["gaussian_process()"]
        RBF["rbf_kernel()"]
    end

    subgraph Layer3["3. FFM Analysis Layer (FFMcracking.py)"]
        CC["Compute_crit_factor_FFM()"]
        ExtractS["extractStressesBetween2points()"]
        EvalSC["evaluate_stress_criterion()"]
        CrackGen["generate_crack() [XFEM]"]
        EvalRE["evaluate_RE() [Betti]"]
        EvalDE["evaluate_DE()"]
    end

    subgraph Layer4["4. Abaqus CAE / Standard Kernel"]
        MdbEngine["mdb (Model Database API)"]
        OdbEngine["openOdb() (Result Database API)"]
        StandardSolver["Abaqus/Standard Solver"]
    end

    Ex1 -->|Direct call| CC
    Ex2 -->|Defines objective t_CC(l)| BO
    BO -->|Evaluates candidate l| CC
    BO --> EI --> GP --> RBF

    CC --> ExtractS & EvalSC & CrackGen & EvalRE & EvalDE
    CC --> MdbEngine & StandardSolver & OdbEngine
```

================================================================================
2. COMPONENT DETAILS
================================================================================

2.1. Application Layer (samples/ Directory)
--------------------------------------------------------------------------------
Provides user-facing execution scripts:
- `example1.py`: Solves the direct problem for a crack with fixed geometry and length.
- `example2.py`: Solves the inverse problem by optimizing crack length within [L_min, L_max] to discover the critical onset length lc.

2.2. Optimization Layer (src/modOptimization.py)
--------------------------------------------------------------------------------
A lightweight, self-contained Bayesian optimization engine:
- Zero heavy dependencies (no SciPy, Scikit-Learn, or PyTorch); relies strictly on Python standard libraries and NumPy.
- Models the non-differentiable objective t_CC(l) using Gaussian Process Regression.
- Uses Expected Improvement (EI) to balance exploration and exploitation.

2.3. FFM Analysis Layer (src/FFMcracking.py)
--------------------------------------------------------------------------------
Core fracture mechanics engine:
- Data Containers:
  - `FFM_Material`: Stores strength (sigmac, tauc) and toughness (Gc1, Gc2) parameters, along with XFEM deactivation penalties (maxsigma, maxd).
  - `FFM_Crack`: Defines 3D spatial crack tip coordinates.
  - `FFM_InputModel`: Bundles file paths, load step name, and material reference.
- Core Orchestrator:
  - `Compute_crit_factor_FFM()` submits Abaqus jobs, parses ODBs, and computes [t_CC, t_SC, t_EC].

2.4. Abaqus CAE / Standard Kernel
--------------------------------------------------------------------------------
Finite element simulation engine accessed via Python scripting:
- Creates models from `.inp` files (`mdb.ModelFromInputFile`).
- Injects field output requests (`FieldOutputRequest`).
- Defines XFEM enrichment features without conformal remeshing.
- Solves models non-interactively (`job.submit(waitForCompletion=True)`).

================================================================================
Navigation: [Previous: Finite Fracture Mechanics](../01-fundamentals/02-finite-fracture-mechanics.md) | [Wiki Home](../index.md) | [Next: Simulation Lifecycle](02-simulation-lifecycle.md)
================================================================================
