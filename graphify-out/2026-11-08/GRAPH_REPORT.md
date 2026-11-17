# Graph Report - FFMcracking  (2026-11-08)

## Corpus Check
- 35 files · ~15,182 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 204 nodes · 210 edges · 31 communities (28 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `54c5bec4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- FFMcracking.py
- FFMcracking: Coupled Criterion for Crack Initiation in Abaqus
- modOptimization.py
- FFM_InputModel
- PULL_REQUEST_TEMPLATE.md
- Bug Report
- Sample 1: Calculating required load factor for a crack onset according to the coupled criterion
- Sample 2: Calculating optimal crack length and critical factor for a crack onset according to the coupled criterion
- Feature Request
- Question
- wiki/README.md
- ⚙️ Funciones Clave
- ⚙️ Key Functions
- en/index.md
- Usage Guide and Examples
- AGENTS.md — Guidelines for AI Coding Agents in FFMcracking
- Guía Práctica de Uso y Ejemplos
- Finite Fracture Mechanics (FFM) Coupled Criterion
- 📐 Mathematical Framework
- Criterio Acoplado de Mecánica de Fractura Finita (FFM)
- 📐 Fundamento Matemático

## God Nodes (most connected - your core abstractions)
1. `Compute_crit_factor_FFM()` - 12 edges
2. `⚙️ Funciones Clave` - 9 edges
3. `AGENTS.md — Guidelines for AI Coding Agents in FFMcracking` - 8 edges
4. `⚙️ Key Functions` - 8 edges
5. `FFM_InputModel` - 7 edges
6. `FFMcracking: Coupled Criterion for Crack Initiation in Abaqus` - 6 edges
7. `expected_improvement()` - 5 edges
8. `Usage Guide and Examples` - 5 edges
9. `📐 Mathematical Framework` - 5 edges
10. `Guía Práctica de Uso y Ejemplos` - 5 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (31 total, 3 thin omitted)

### Community 0 - "FFMcracking.py"
Cohesion: 0.15
Nodes (17): Compute_crit_factor_FFM(), evaluate_DE(), evaluate_RE(), evaluate_stress_criterion(), extractDespForcefromNodes(), extractNodesDesp(), extractNodesForce(), extractStresses_crackpath() (+9 more)

### Community 1 - "FFMcracking: Coupled Criterion for Crack Initiation in Abaqus"
Cohesion: 0.18
Nodes (10): Contact, Current Features, Features, FFMcracking: Coupled Criterion for Crack Initiation in Abaqus, How It Works, How to Cite FFMcracking, How to contribute, Prerequisites (+2 more)

### Community 2 - "modOptimization.py"
Cohesion: 0.36
Nodes (6): bayesian_optimization(), expected_improvement(), gaussian_process(), norm_cdf(), norm_pdf(), rbf_kernel()

### Community 4 - "PULL_REQUEST_TEMPLATE.md"
Cohesion: 0.33
Nodes (5): Detailed Description of the Pull Request / Additional comments, PR Checklist, References, Summary of the Pull Request, Validation Steps Performed

### Community 5 - "Bug Report"
Cohesion: 0.40
Nodes (4): Additional Context, Bug Report, Reproduce, Symptom

### Community 6 - "Sample 1: Calculating required load factor for a crack onset according to the coupled criterion"
Cohesion: 0.40
Nodes (4): Contributing, License, Sample 1: Calculating required load factor for a crack onset according to the coupled criterion, Usage

### Community 7 - "Sample 2: Calculating optimal crack length and critical factor for a crack onset according to the coupled criterion"
Cohesion: 0.40
Nodes (4): Contributing, License, Sample 2: Calculating optimal crack length and critical factor for a crack onset according to the coupled criterion, Usage

### Community 10 - "wiki/README.md"
Cohesion: 0.14
Nodes (11): Arquitectura y Flujo de Trabajo, 🔄 Flujo de Ejecución Detallado (`Compute_crit_factor_FFM`), 🧩 Grafo de Nodos Clave (Análisis con Graphify), 🏛️ Visión General de la Arquitectura, 🗺️ Estructura de la Documentación, 🎯 Objetivo del Proyecto, 📂 Organización del Repositorio, Wiki de FFMcracking (Español) (+3 more)

### Community 11 - "⚙️ Funciones Clave"
Cohesion: 0.14
Nodes (14): 1. `FFM_Material`, 2. `FFM_Crack`, 3. `FFM_InputModel`, 🏛️ Clases Principales, `Compute_crit_factor_FFM(CrackFFM, OriginalModel)`, `evaluate_DE(crackObject, OriginalModel, odbECwith, session)`, `evaluate_RE(Ux1_BC, Ux2_BC, Uy1_BC, Uy2_BC, Fx1_BC, Fx2_BC, Fy1_BC, Fy2_BC)`, `evaluate_stress_criterion(stt, snn, stn, ModelObject)` (+6 more)

### Community 22 - "⚙️ Key Functions"
Cohesion: 0.15
Nodes (13): 1. `FFM_Material`, 2. `FFM_Crack`, 3. `FFM_InputModel`, `Compute_crit_factor_FFM(CrackFFM, OriginalModel)`, 🏛️ Core Classes, Core Module: FFMcracking.py, `evaluate_DE(crackObject, OriginalModel, odbECwith, session)`, `evaluate_RE(Ux1_BC, Ux2_BC, Uy1_BC, Uy2_BC, Fx1_BC, Fx2_BC, Fy1_BC, Fy2_BC)` (+5 more)

### Community 23 - "en/index.md"
Cohesion: 0.18
Nodes (8): Architecture & Execution Workflow, 🔄 Detailed Execution Workflow (`Compute_crit_factor_FFM`), 🧩 Graphify Architectural Insights, 🏛️ System Architecture, 🗺️ Documentation Index, FFMcracking Wiki (English), 🎯 Project Objective, 📂 Repository Layout

### Community 24 - "Usage Guide and Examples"
Cohesion: 0.18
Nodes (10): Configuration Snippet in `example1.py`, Configuration Snippet in `example2.py`, Execution Steps, Execution Steps, Objective Function, 🛠️ Operational Tips & Troubleshooting, 📋 Prerequisites, 🧪 Sample 1: Fixed Crack Geometry (`sample1_OpenHoleCrackFixed`) (+2 more)

### Community 25 - "AGENTS.md — Guidelines for AI Coding Agents in FFMcracking"
Cohesion: 0.20
Nodes (9): 1. Project Overview & Domain Context, 2. Mandatory Behavioral Rules (User Rules), 3. Repository Architecture & File Layout, 4. Key Abstractions & Architectural God Nodes, 5. Runtime Environment & Compatibility Constraints, 6. Graphify Knowledge Graph Commands, 7. Documentation References, AGENTS.md — Guidelines for AI Coding Agents in FFMcracking (+1 more)

### Community 26 - "Guía Práctica de Uso y Ejemplos"
Cohesion: 0.20
Nodes (10): 🛠️ Consejos y Solución de Problemas, Definición de la Función Objetivo, 🧪 Ejemplo 1: Grieta de Geometría Fija (`sample1_OpenHoleCrackFixed`), 🔬 Ejemplo 2: Optimización de Longitud de Grieta (`sample2_OpenHoleBasicOptimLenght`), Estructura de Parámetros en `example1.py`, Estructura de Parámetros en `example2.py`, Guía Práctica de Uso y Ejemplos, Pasos de Ejecución (+2 more)

### Community 27 - "Finite Fracture Mechanics (FFM) Coupled Criterion"
Cohesion: 0.22
Nodes (8): 1. Stress Criterion (SC), 2. Energy Criterion (EC), 3. Coupled Factor and Critical Crack Length, Betti's Reciprocal Work Theorem in `evaluate_RE`:, Finite Fracture Mechanics (FFM) Coupled Criterion, ⚠️ Limitations of Classical Theories, 📐 Mathematical Formulation, 💡 The Principle of the Coupled Criterion

### Community 28 - "📐 Mathematical Framework"
Cohesion: 0.25
Nodes (7): 1. RBF Kernel (Radial Basis Function), 2. Gaussian Process Regression (`gaussian_process`), 3. Acquisition Function: Expected Improvement (EI), 4. Bayesian Optimization Loop (`bayesian_optimization`), 📐 Mathematical Framework, Optimization Module: modOptimization.py, 🎯 Why Bayesian Optimization?

### Community 29 - "Criterio Acoplado de Mecánica de Fractura Finita (FFM)"
Cohesion: 0.25
Nodes (8): 1. Criterio de Tensiones (Stress Criterion, SC), 2. Criterio de Energía (Energy Criterion, EC), 3. Factor Crítico Acoplado y Longitud Crítica, Criterio Acoplado de Mecánica de Fractura Finita (FFM), 💡 El Principio del Criterio Acoplado, 📐 Formulación Matemática, ⚠️ Las Limitaciones de las Teorías Clásicas, Teorema del Trabajo Recíproco en `evaluate_RE`:

### Community 30 - "📐 Fundamento Matemático"
Cohesion: 0.29
Nodes (7): 1. Kernel RBF (Función de Base Radial), 2. Proceso Gaussiano (`gaussian_process`), 3. Función de Adquisición: Expected Improvement (EI), 4. Algoritmo de Optimización (`bayesian_optimization`), 📐 Fundamento Matemático, Módulo de Optimización: modOptimization.py, 🎯 ¿Por qué Optimización Bayesiana?

## Knowledge Gaps
- **100 isolated node(s):** `Symptom`, `Reproduce`, `Additional Context`, `Additional Context`, `Additional Context` (+95 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Módulo Principal: FFMcracking.py` connect `⚙️ Funciones Clave` to `wiki/README.md`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `Core Module: FFMcracking.py` connect `⚙️ Key Functions` to `en/index.md`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **What connects `Symptom`, `Reproduce`, `Additional Context` to the rest of the system?**
  _100 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `FFMcracking.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14761904761904762 - nodes in this community are weakly interconnected._
- **Should `wiki/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.1437908496732026 - nodes in this community are weakly interconnected._
- **Should `⚙️ Funciones Clave` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._