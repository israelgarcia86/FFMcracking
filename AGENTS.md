# AGENTS.md — Guidelines for AI Coding Agents in FFMcracking

This file provides system instructions, domain context, mechanical foundations, development constraints, and operational guidelines for AI coding assistants (Antigravity, Copilot, Codex, Claude Code, etc.) working on the FFMcracking repository.

================================================================================
1. PROJECT OVERVIEW AND DOMAIN CONTEXT
================================================================================

FFMcracking is a scientific computational framework that implements the Coupled Criterion (CC) of Finite Fracture Mechanics (FFM), originally formulated by Dominique Leguillon (2002) and developed by Israel G. Garcia in his doctoral dissertation (University of Seville, 2014; supervised by V. Mantic).

The framework interfaces with the Abaqus CAE finite element suite to automate crack initiation prediction in elastic solids with geometric stress concentrators (open holes, notches, reentrant corners, and adhesive joints).

1.1. Core Physics and Mechanics
--------------------------------------------------------------------------------
- Finite Fracture Mechanics:
  Classical Linear Elastic Fracture Mechanics (LEFM, Griffith/Irwin) fails at crack initiation because for initial crack length a -> 0, the energy release rate vanishes (G -> 0), predicting infinite initiation load. Pointwise stress criteria fail because stress at sharp reentrant corners is mathematically singular (sigma -> inf), predicting premature failure at near-zero load.
  FFM resolves this by postulating that crack initiation occurs as an abrupt, finite jump of length lc.

- The Coupled Criterion (CC):
  A crack of length l initiates under load multiplier t when two conditions are met SIMULTANEOUSLY:
  1. Stress Criterion (SC - Pointwise, Leguillon 2002):
     t * sigma_nn(x) >= sigma_c   for all x in [0, l]
     => t_SC(l) = sigma_c / min_{x in [0, l]} (sigma_nn(x))
     Because stress decays with distance from the concentrator, t_SC(l) is strictly monotonically increasing with l.
  2. Energy Criterion (EC - Incremental Balance):
     -Delta Pi(t) >= Delta Gamma = Gc * l
     Under proportional loading in linear elasticity: -Delta Pi(t) = t^2 * (-Delta Pi(1)).
     Defining RE = -Delta Pi(1) and DE = Gc * l:
     => t_EC(l) = sqrt(DE / |RE|) = sqrt((Gc * l) / |RE|)
     For short cracks, RE scales as l^2 or steeper, so t_EC(l) -> inf as l -> 0.
  3. Coupling and Optimal Initiation:
     t_CC(l) = max(t_SC(l), t_EC(l))
     Physical onset occurs at the global minimum:
     lc = argmin_{l > 0} t_CC(l)
     tc = t_CC(lc)

- Reciprocal Work Theorem of Betti:
  Instead of integrating strain energy across the entire mesh volume, RE is evaluated from boundary reaction forces and displacements on nodes with boundary conditions (BCs):
  RE = 0.5 * sum_{i in BCs} [ Ux2(i)*Fx1(i) + Uy2(i)*Fy1(i) - Ux1(i)*Fx2(i) - Uy1(i)*Fy2(i) ]
  where state 1 is uncracked and state 2 is cracked.

- Characteristic Length and Brittleness Number (Garcia, 2014, §3.5):
  l_ch = (E * Gc) / (sigma_c^2)
  gamma = (1 / sigma_c) * sqrt((Gc * E) / L) = sqrt(l_ch / L)
  When gamma >> 1 (brittle/large scale), energy dominates (LEFM). When gamma << 1, stress dominates.

- The XFEM Non-Cohesive "Geometric Cut" Concept:
  Abaqus requires damage properties to insert an XFEM crack. To prevent Abaqus from applying cohesive damage evolution, FFMcracking sets an unreachable damage initiation threshold:
  maxsigma = 5000e10 * sigmac
  maxd = 1.0e10 * Gc1 / sigmac
  This suppresses damage evolution entirely: the crack remains traction-free, the solid stays linear elastic, and XFEM functions strictly as a kinematic discontinuity without conformal remeshing. All failure physics is evaluated in post-processing.

================================================================================
2. MANDATORY BEHAVIORAL RULES (USER RULES)
================================================================================

Any agent operating in this repository must strictly adhere to the following rules without exception:

1. Initial Repository Inspection:
   At the start of any conversation or task, always inspect markdown files (*.md), the wiki/ directory, and README files to understand current codebase state and domain context.

2. Never Run Tests Autonomously:
   NEVER execute test suites, Abaqus verification jobs, or validation scripts unless the user explicitly instructs you to do so.
   Limit your actions to fixing code, refactoring, documenting, adding features, or explaining solutions. Testing is strictly triggered on user demand.

3. Change Impact, Scope, and Generalization:
   Always evaluate how changes affect the target script and other modules across the codebase.
   NEVER modify logic that impacts multiple scripts without first consulting and confirming with the user.
   Do not implement narrow, single-case workarounds: understand the root cause and implement clean, generalizable solutions that apply transversally.

4. Documentation and Comment Integrity:
   Preserve existing docstrings, mathematical explanations, and code comments unless explicitly instructed otherwise.

================================================================================
3. REPOSITORY ARCHITECTURE AND DIRECTORY LAYOUT
================================================================================

```text
FFMcracking/
|-- src/
|   |-- __init__.py            # Shared package entry point
|   |-- FFMcracking.py         # Main Abaqus interface & FFM Coupled Criterion solver
|   |-- modOptimization.py     # Bayesian optimization engine (GPR, RBF kernel, EI)
|   |-- modVisualization.py   # Abaqus result snapshots, contour maps and XY plot
|   +-- README.nd              # Module notes (legacy name)
|-- samples/
|   |-- sample1_OpenHoleCrackFixed/
|   |   |-- example1.py        # Fixed crack geometry calculation
|   |   |-- OpenHoleLongDEF-fmesh-00.inp
|   |   +-- README.md
|   +-- sample2_OpenHoleBasicOptimLenght/
|       |-- example2.py        # Bayesian optimization of crack length
|       |-- OpenHoleLongDEF-fmesh-00.inp
|       +-- README.md
|-- wiki/                      # Complete structured technical wiki
|   |-- README.md              # Language hub
|   |-- es/                    # Spanish wiki documentation
|   |   |-- _sidebar.md        # Fast navigation sidebar
|   |   |-- index.md           # Home and topic map
|   |   |-- 01-fundamentos/    # Theory and Finite Fracture Mechanics
|   |   |-- 02-arquitectura/   # System design and Abaqus job lifecycle
|   |   |-- 03-modulos/        # FFMcracking, stress extraction, XFEM, Bayes
|   |   |-- 04-guias-y-ejemplos/ # Step-by-step tutorials (Samples 1 & 2)
|   |   +-- 05-faq-y-desarrollo/ # FAQ, Abaqus gotchas (.lck), and roadmap
|   +-- en/                    # English wiki documentation (mirrored structure)
|       |-- _sidebar.md        # Fast navigation sidebar
|       |-- index.md           # Home and topic map
|       |-- 01-fundamentals/   # Theory and Finite Fracture Mechanics
|       |-- 02-architecture/   # System design and Abaqus job lifecycle
|       |-- 03-modules/        # FFMcracking, stress extraction, XFEM, Bayes
|       |-- 04-guides-and-examples/ # Step-by-step tutorials (Samples 1 & 2)
|       +-- 05-faq-and-development/ # FAQ, Abaqus gotchas (.lck), and roadmap
|-- doc/                       # Project documentation
|-- test/                      # Test scripts (DO NOT run without explicit user request)
|-- AGENTS.md                  # This instructions file
+-- README.md                  # Top-level repository overview
```

================================================================================
4. KEY ABSTRACTIONS AND ARCHITECTURAL GOD NODES
================================================================================

- `Compute_crit_factor_FFM(CrackFFM, OriginalModel)` (src/FFMcracking.py):
  Highest-degree architectural hub. Manages the dual-simulation lifecycle:
  1. Solves or reuses uncracked continuous job (`-without.odb`).
  2. Extracts normal stresses along candidate crack path (`extractStressesBetween2points()`).
  3. Evaluates stress criterion (`evaluate_stress_criterion()`).
  4. Injects XFEM crack (`generate_crack()`) and runs cracked job (`-with.odb`).
  5. Computes reciprocal work RE via Betti's theorem (`evaluate_RE()`).
  6. Computes dissipated energy DE (`evaluate_DE()`).
  7. Evaluates coupled criterion and closes databases.

- `FFM_Material`: Stores `sigmac`, `tauc`, `Gc1`, `Gc2`, and derived XFEM penalties (`maxsigma`, `maxd`).
- `FFM_Crack`: Stores crack tip coordinates `[(x1, y1, 0), (x2, y2, 0)]`.
- `FFM_InputModel`: Bundles file paths, load step name, and material instance.
- `bayesian_optimization(obj_func, bounds, n_iterations, n_initial)` (src/modOptimization.py):
  Black-box optimization engine for non-smooth $t_{CC}(l)$ using Gaussian Process Regression, RBF kernel, and Expected Improvement (EI).

================================================================================
5. RUNTIME ENVIRONMENT AND COMPATIBILITY CONSTRAINTS
================================================================================

1. Dual Python 2.7 and Python 3.x Compatibility:
   Abaqus 2020-2023 embeds Python 2.7; Abaqus 2024+ embeds Python 3.10+.
   RULE: Never use Python 3-only syntax (such as f-strings, match/case, or typing module imports) in `src/`. Always use `.format()` and explicit floating-point division.

2. Zero External Dependencies Policy:
   `src/FFMcracking.py` and `src/modOptimization.py` must NEVER depend on external packages like SciPy, Scikit-Learn, PyTorch, SymPy, or Pandas. Only standard library modules (`math`, `random`, `sys`, `os`) and `numpy` are allowed.

3. Strict Resource Management in Abaqus:
   Solver routines must close ODB pointers (`odb.close()`) and model databases (`mdb.close()`). Unclosed writable databases can leave `.lck` files that block later jobs. The post-processor may keep its own read-only copy open for viewing in Abaqus/CAE.

================================================================================
6. CRITICAL TRAPS AND DEFECTS IDENTIFIED IN CODEBASE
================================================================================

The following known defects and numerical sensitivities have been identified (refer to thesis and technical analysis):

1. Compression Handling in `evaluate_stress_criterion`:
   Current code checks `if max(snn) > 0:` but divides by `min(snn)`. If the path extends into a compressive zone, `min(snn)` is negative, producing negative load multipliers. The physical pointwise criterion requires `min(snn) > 0` throughout the initiation zone.

2. Multi-Instance Assemblies in `generate_crack`:
   The enriched domain `crackDomain` is currently populated with elements from the last iterated instance (`elements = e1`) instead of the accumulated element list (`elements1`), causing incomplete XFEM domains in multi-instance models.

3. Numerical Precision in Reciprocal Work RE:
   For short crack lengths (l << lc), displacements and forces in the cracked state differ minimally from the uncracked state. RE is computed by subtracting nearly equal quantities. Double precision solver execution (`precision=DOUBLE_PLUS_PACK`) is essential to prevent cancellation errors.

4. Uncracked Model Caching Invalidation:
   The `-without.odb` caching does not verify `.inp` timestamps or hashes. After a change to model geometry or the loading step, use a clean `compFiles/` output directory to avoid reusing stale results.

5. Path Portability:
   File paths currently rely on string concatenations expecting trailing `/`. When updating code, migrate to `os.path.join()` and `os.path.normpath()` for cross-platform portability.

================================================================================
7. GRAPHIFY KNOWLEDGE GRAPH COMMANDS
================================================================================

A code knowledge graph is maintained in `graphify-out/`. Use the `graphify` CLI to inspect relationships and navigate code:

- Update code graph after edits:
  powershell: graphify update .
- Inspect God Nodes (architectural hubs):
  powershell: graphify god-nodes
- Explain a function or class:
  powershell: graphify explain "Compute_crit_factor_FFM()"
  powershell: graphify explain "bayesian_optimization()"
- Find dependency paths:
  powershell: graphify path "Compute_crit_factor_FFM()" "evaluate_RE()"

================================================================================
8. DOCUMENTATION REFERENCES
================================================================================

For complete mathematical formulations, code listings, and execution tutorials:
- Spanish Wiki: wiki/es/index.md
- English Wiki: wiki/en/index.md
- Ph.D. Thesis Reference: I. Garcia, "Crack initiation in composites at micro and meso scales. Development and applications of finite fracture mechanics", University of Seville, 2014 (dir. V. Mantic).

================================================================================
9. SAMPLE MODELS AND EXPECTED OUTPUT
================================================================================

Both examples were already present in the original repository. Example 1 was added in `a085c9a` and Example 2 in `79cafcb`. The two current copies of `OpenHoleLongDEF-fmesh-00.inp` are identical. They describe the same 2D linear elastic plate with a hole. The Python examples set `sigmac = 70.0` and `Gc1 = 0.4` in the model's consistent units, with `tauc` and `Gc2` set to zero.

- Both inputs use a static step named `Loading` with `nlgeom=NO`. The upper edge (`Set-3`) has a prescribed vertical displacement `U2 = 0.01`. The bottom symmetry (`Set-2`, `YSYMM`) sets `U2 = 0`, and the vertical symmetry edge (`Set-1`, `XSYMM`) sets `U1 = 0`. Abaqus computes the reaction forces. Treat `0.01` as the reference displacement in model units.
- An increment is a solver advance within an analysis step. A frame is a state saved in the ODB. The visualization selects the last saved frame of `Loading`, at the full reference displacement for a completed sample run. The inspected Abaqus 2024 cracked jobs reached that state in one increment. An increment within a job is separate from a crack-length evaluation, which calls the solver again.
- Example 1 (`samples/sample1_OpenHoleCrackFixed/example1.py`) evaluates the horizontal crack from `(1.0, 0.0, 0.0)` to `(4.5, 0.0, 0.0)`, of length `3.5`. The solver returns the coupled, stress and energy factors in that order. The script prints the coupled factor.
- Example 2 (`samples/sample2_OpenHoleBasicOptimLenght/example2.py`) fixes the start at `(1.0, 0.0, 0.0)` and direction along X. It searches lengths in `[0.1, 5.0]`. Five random starting lengths are followed by ten evaluations selected using Expected Improvement on a 100-point candidate grid. The objective returns all three factors. The optimizer returns the best observed length and coupled factor. The script prints both values and the paired lists `xtrain` (lengths) and `ytrain` (coupled factors). Lengths can repeat and the search result is the minimum among the evaluated candidates.
- The sample plots show fields at the reference load. The coupled factor scales the prescribed displacement to the predicted onset under proportional loading. For example, `3.9020 * 0.01` gives about `0.0390` model units for the fixed crack.

================================================================================
10. RUNNING THE SAMPLES AND PRESERVING RESULTS
================================================================================

- Abaqus/CAE and Abaqus/Standard both need license access. When using an educational license server reached over VPN, establish the VPN connection first.
- Keep each sample folder directly under `samples/` with its script and input deck, and keep the shared `src/` package at the repository root. Both entry points derive the location of `src` from the working directory and check for `src/__init__.py`.
- In Abaqus/CAE, save any open work and select **File → Set Work Directory**. Set the absolute path of the sample folder, then use **File → Run Script** to select `example1.py` or `example2.py`. File selection alone leaves the working directory unchanged. The solver closes the active model database. The terminal equivalent is `abaqus cae noGUI=example1.py` or `abaqus cae noGUI=example2.py` from the corresponding sample directory.
- The jobs write `<input-stem>-EC-without.odb` and `<input-stem>-EC-with.odb` under the sample's `compFiles/`. The solver reuses the uncracked ODB by filename and replaces prior cracked-job files before each cracked solve. Use a clean copy of the sample under `samples/` when changing inputs or preserving earlier results. Numeric results appear in the Abaqus message area or terminal and can also be recorded in the session replay file `abaqus.rpy`. The `.sta` job file reports completion status and increments.
- Python 3 rejects function-local wildcard imports with `SyntaxError: import * only allowed at module level`. The Abaqus 2024 fix in `c22107c` moved the Abaqus wildcard imports to module scope in `src/FFMcracking.py`. The old copies beside the example scripts could still be selected by a bare `import FFMcracking`. Commit `65479d3` removed those copies and made the examples import the shared modules from `src`.

================================================================================
11. AUTOMATIC AND MANUAL VISUALIZATION
================================================================================

Commit `c99f650` on `visualizacion` adds the shared Abaqus post-processor `src/modVisualization.py`. Both samples call `CrackResults` with their `FFM_InputModel`. The post-processor reads existing solver output and leaves the calculations in the solver and optimizer.

- `save_if_best(factor)` copies the cracked job's ODB to `compFiles/visualization/crack.odb` for the first recorded result and each strict improvement. Ties retain the earlier copy. Example 1 records its sole evaluation. Example 2 records during the objective so the best ODB is preserved before a subsequent evaluation replaces the solver file.
- `show()` opens the snapshot read-only in `Viewport: 1`, selects the last saved frame of the configured step, and exports `crack-s22.png` on the undeformed model and `crack-opening-s22.png` on the deformed model. The uniform display scale of 150 highlights the XFEM opening and only affects the displayed geometry. With the GUI attached, the snapshot is left in the viewport. A headless run closes the ODB after export.
- `show(x_train, y_train)` also exports `length-optimization.png` from the paired evaluations provided by the optimizer. Blue circles are evaluated lengths and the red diamond is the lowest finite factor plotted. Duplicate points can overlap. Non-finite factors are omitted from the XY plot. The plot remains available in the CAE session as **FFM crack-length optimization**.
- Other FFMcracking examples can reuse the module when they produce a cracked ODB with the same naming convention and a stress field `S` containing `S22` in the configured step. The existing view expects `Viewport: 1`. S22 is the normal stress in direction 2, global Y for these inputs. It is perpendicular to the horizontal sample cracks. For an inclined crack, use a suitable coordinate transformation or component to interpret the normal stress. The map from the cracked ODB differs from the path stresses the solver extracts on the uncracked model for the stress criterion.
- To inspect results manually, open the cracked ODB in **Visualization**, choose **Result → Step/Frame → Loading** and the last saved frame, then select **Result → Field Output → S → S22**. Use **Plot → Contours** on the undeformed or deformed shape. To reproduce the enlarged opening, set the uniform scale to 150 in **Common Plot Options**. To create the optimization plot manually, enter the paired `xtrain` and `ytrain` values with **Tools → XY Data → Create**, or import two text columns. If the original example overwrote the best-case ODB, use a saved copy or rerun that candidate to inspect its field results.
- The top-level README and both language versions of the wiki's **Example 1** and **Example 2** guides explain the exported figures and where to find them.

================================================================================
12. VERIFIED ABAQUS REFERENCE RUNS
================================================================================

On 2026-11-21, both examples completed in Abaqus 2024 from a separate working copy of revision `65479d3`. Example 1 returned `3.9020145683789984`. Example 2 returned `3.74112289746173` at `L = 2.0303030303030307` after 15 evaluations of 11 distinct lengths. The final evaluation in that run matched the best observed one. These are reference values for that setup. The optimization starts from random points, so later runs may differ.

The images from those runs were exported in Abaqus with a separate local helper, `vistas.py`. That helper is outside the repository. The runs verified the shared `src` imports and the numerical outputs before the integrated post-processor was added. As of this documentation update, running the integrated visualization at `c99f650` end to end in Abaqus remains pending.
