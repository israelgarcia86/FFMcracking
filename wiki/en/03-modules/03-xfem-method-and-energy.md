# 03. XFEM Modeling and EC

Navigation: [Wiki Home](../index.md) > [03. Modules](01-module-ffmcracking.md) > XFEM Modeling and EC

================================================================================
1. ROLE OF XFEM IN THE FRAMEWORK
================================================================================

To calculate potential energy release (-Delta Pi), the elastic response of the body after crack initiation must be determined.

Conventional finite element methods require conformal remeshing for each candidate crack length. FFMcracking eliminates remeshing entirely by leveraging the Extended Finite Element Method (XFEM).

Through Partition of Unity, XFEM enriches nodal degrees of freedom in elements cut by the crack:
- Heaviside jump functions: represent discontinuous displacement jumps across crack faces.
- Asymptotic tip functions: capture the elasto-static singularity at the crack tip.

Conceptual role: "Geometric Separator", not cohesive damage model.
The code injects intentionally unreachable damage initiation thresholds (`maxsigma = 5000e10 * sigmac`). Consequently:
- The solid remains strictly linear elastic.
- Crack faces are traction-free.
- XFEM operates purely as a kinematic mechanism introducing geometric discontinuity at zero remeshing cost.

================================================================================
2. CRACK INSERTION: generate_crack
================================================================================

Signature:
  `generate_crack(crackObject, modelObject, session, regionToolset)`

Operations in Abaqus CAE:
1. `ConstrainedSketch`: Draws 2D straight segment between `point1` and `point2`.
2. `BaseWire`: Creates a 1D deformable wire part representing the crack.
3. Assembly Instance: Instantiates `Crack-Instance` in `rootAssembly`.
4. Enriched Domain (`crackDomain`): Selects mesh elements to receive XFEM enrichment.
5. Engineering Feature: Calls `a.engineeringFeatures.XFEMCrack(name='Crack-1', crackDomain=crackDomain, crackLocation=crackLocation)`.

Multi-instance note:
In the original code, `crackDomain` selects elements from the last iterated instance (`elements = e1`). For multi-instance assemblies, the domain should accumulate elements across all instances (`elements1`).

================================================================================
3. BETTI'S RECIPROCAL WORK THEOREM AND RE
================================================================================

The term RE represents released elastic energy under unit applied load:
  RE = -Delta Pi(1)

3.1. Mathematical Foundation
--------------------------------------------------------------------------------
In linear elastic bodies, incremental potential energy variation between uncracked state 1 and cracked state 2 can be formulated via Betti's Reciprocal Work boundary integral (Garcia, 2014, chap. 3):

  RE = 0.5 * integral_{boundary} [ u_i^(2) * T_i^(1) - u_i^(1) * T_i^(2) ] dS

where u denotes displacement and T denotes surface traction.

Because tractions vanish on traction-free boundaries (T = 0), and internal equilibrium holds everywhere, the boundary integral vanishes everywhere except at nodes with applied boundary conditions (prescribed displacements or reaction forces).

3.2. Nodal Implementation: evaluate_RE
--------------------------------------------------------------------------------
The function `extractDespForcefromNodes()` reads nodal displacements (U) and reaction forces (RF / NFORC) at boundary condition nodes from both ODB files.

Then `evaluate_RE()` evaluates discretely:

  RE = 0.5 * sum_{i in BCs} [ Ux2(i)*Fx1(i) + Uy2(i)*Fy1(i) - Ux1(i)*Fx2(i) - Uy1(i)*Fy2(i) ]

Computational benefits:
- Eliminates integrating strain energy density across thousands of mesh elements.
- Execution is instantaneous (operates on boundary nodes only).
- Yields high accuracy governed only by global mesh resolution.

Numerical sensitivity:
For very short cracks (l << lc), displacements and forces between states 1 and 2 are almost identical. RE is obtained from subtracting nearly equal numbers. Double precision solver execution (`precision=DOUBLE_PLUS_PACK`) is strongly recommended.

================================================================================
4. DISSIPATED ENERGY AND t_EC FACTOR
================================================================================

4.1. Dissipated Energy: evaluate_DE
--------------------------------------------------------------------------------
The function `evaluate_DE(crackObject, Gc1, ...)` calculates the energy cost of opening the new fracture surface:

  DE = Gc1 * effective_length

An inspection path sampled along results fields verifies geometric extent within the physical boundary.

4.2. Energy Criterion Critical Factor
--------------------------------------------------------------------------------
With DE and RE determined:

  t_EC = sqrt(DE / |RE|) = sqrt((Gc1 * effective_length) / |RE|)

================================================================================
Navigation: [Previous: Stress Extraction](02-stress-extraction.md) | [Wiki Home](../index.md) | [Next: Optimization Module](04-optimization-module.md)
================================================================================
