# Graph Report - FFMcracking  (2026-11-08)

## Corpus Check
- 53 files · ~18,848 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 295 nodes · 353 edges · 47 communities (44 shown, 3 thin omitted)
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
- en/index.md
- es/index.md
- 📚 Table of Contents
- 📚 Estructura Temática
- 📦 Component Overview
- AGENTS.md — Guidelines for AI Coding Agents in FFMcracking
- 01. Core Module: FFMcracking.py
- 02. Stress Extraction & Stress Criterion
- 03. XFEM Modeling & Energy Criterion
- 04. Bayesian Optimization Module
- 📦 Descripción de Componentes
- 01. Módulo Central: FFMcracking.py
- 02. Extracción de Tensiones y Criterio SC
- 03. Modelado XFEM y Criterio de Energía
- 04. Módulo de Optimización Bayesiana
- 📐 Mathematical Formulation of the Coupled Criterion (CC)
- 01. Requirements & Setup
- 03. Sample 2: Length Optimization
- ❓ Frequently Asked Questions
- 📐 Formulación Matemática del Criterio Acoplado (CC)
- 01. Requisitos e Instalación
- 03. Ejemplo 2: Optimización de Longitud de Grieta
- ❓ Preguntas Frecuentes
- 01. Overview and Motivation
- 02. Sample 1: Fixed Crack
- 01. Visión General y Motivación
- 02. Ejemplo 1: Grieta Fija

## God Nodes (most connected - your core abstractions)
1. `Compute_crit_factor_FFM()` - 12 edges
2. `AGENTS.md — Guidelines for AI Coding Agents in FFMcracking` - 8 edges
3. `FFM_InputModel` - 7 edges
4. `FFMcracking: Coupled Criterion for Crack Initiation in Abaqus` - 6 edges
5. `📚 Table of Contents` - 6 edges
6. `📚 Estructura Temática` - 6 edges
7. `expected_improvement()` - 5 edges
8. `01. Overview and Motivation` - 5 edges
9. `📦 Component Overview` - 5 edges
10. `02. Sample 1: Fixed Crack` - 5 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (47 total, 3 thin omitted)

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

### Community 10 - "en/index.md"
Cohesion: 0.18
Nodes (9): 02. Abaqus Simulation Lifecycle, 🔄 The Dual Simulation Lifecycle, ⚡ Uncracked Simulation Caching, 02. Developer Guide & Roadmap, 🛠️ Contribution & Coding Standards, 🗺️ Future Roadmap, Documentación de FFMcracking / FFMcracking Documentation Hub, 📌 Estructura General de la Wiki (+1 more)

### Community 11 - "es/index.md"
Cohesion: 0.21
Nodes (6): 02. Ciclo de Simulación en Abaqus, 🔄 El Ciclo de Vida Dual de Simulación, ⚡ Estrategia de Caché del Modelo Sin Grieta, 02. Guía para Desarrolladores y Hoja de Ruta, 🛠️ Buenas Prácticas de Desarrollo en FFMcracking, 🗺️ Hoja de Ruta (*Roadmap*)

### Community 22 - "📚 Table of Contents"
Cohesion: 0.25
Nodes (8): [1. Theoretical Fundamentals](01-fundamentals/01-introduction.md), [2. System Architecture](02-architecture/01-system-design.md), [3. System Modules & Technical Reference](03-modules/01-module-ffmcracking.md), [4. Practical Guides & Tutorials](04-guides-and-examples/01-requirements-and-setup.md), [5. Support, FAQ & Development](05-faq-and-development/01-troubleshooting-faq.md), FFMcracking Wiki — Official Documentation, 📚 Table of Contents, 🧭 Topic Navigation Map

### Community 23 - "📚 Estructura Temática"
Cohesion: 0.25
Nodes (8): [1. Fundamentos Teóricos](01-fundamentos/01-introduccion.md), [2. Arquitectura del Sistema](02-arquitectura/01-diseno-general.md), [3. Referencia Técnica de Módulos](03-modulos/01-modulo-ffmcracking.md), [4. Guías Prácticas y Tutoriales](04-guias-y-ejemplos/01-requisitos-e-instalacion.md), [5. Soporte, FAQ y Desarrollo](05-faq-y-desarrollo/01-resolucion-problemas-faq.md), 📚 Estructura Temática, 🧭 Mapa de Navegación de la Wiki, Wiki de FFMcracking — Documentación Oficial

### Community 24 - "📦 Component Overview"
Cohesion: 0.29
Nodes (7): 01. System Design, 1. Application Layer (`samples/`), 2. Optimization Layer (`src/modOptimization.py`), 3. FFM Core Layer (`src/FFMcracking.py`), 4. Abaqus CAE / ODB Layer, 📦 Component Overview, 🏗️ Layered Architecture

### Community 25 - "AGENTS.md — Guidelines for AI Coding Agents in FFMcracking"
Cohesion: 0.20
Nodes (9): 1. Project Overview & Domain Context, 2. Mandatory Behavioral Rules (User Rules), 3. Repository Architecture & File Layout, 4. Key Abstractions & Architectural God Nodes, 5. Runtime Environment & Compatibility Constraints, 6. Graphify Knowledge Graph Commands, 7. Documentation References, AGENTS.md — Guidelines for AI Coding Agents in FFMcracking (+1 more)

### Community 26 - "01. Core Module: FFMcracking.py"
Cohesion: 0.29
Nodes (7): 01. Core Module: FFMcracking.py, 1. `FFM_Material`, 2. `FFM_Crack`, 3. `FFM_InputModel`, 🏛️ Core Classes, 🛠️ Orchestrator Function: `Compute_crit_factor_FFM`, 📄 Source File: [`src/FFMcracking.py`](file:///C:/Users/manue/OneDrive/Escritorio/MIA/TFM/FFMcracking/src/FFMcracking.py)

### Community 27 - "02. Stress Extraction & Stress Criterion"
Cohesion: 0.29
Nodes (7): 02. Stress Extraction & Stress Criterion, 1. Spatial Discretization, 2. Local Orthonormal Basis, 3. Stress Component Extraction, 📐 Coordinate Transformation Pipeline, 🧮 Multiplier Evaluation: `evaluate_stress_criterion`, 🔍 Path Stress Extraction

### Community 28 - "03. XFEM Modeling & Energy Criterion"
Cohesion: 0.29
Nodes (7): 03. XFEM Modeling & Energy Criterion, 1. Released Energy ($RE$): `evaluate_RE`, 2. Dissipated Energy ($DE$): `evaluate_DE`, 3. Critical Energy Multiplier ($k_{EC}$):, 🛠️ Crack Generation: `generate_crack`, 🔬 Energy Balance Workflow, ⚡ The Extended Finite Element Method (XFEM)

### Community 29 - "04. Bayesian Optimization Module"
Cohesion: 0.29
Nodes (7): 04. Bayesian Optimization Module, 1. RBF Kernel (Radial Basis Function), 2. Surrogate Model: Gaussian Process (`gaussian_process`), 3. Dual Acquisition Function: Expected Improvement (EI), ⚙️ Mathematical Foundations, 🔁 Search Algorithm: `bayesian_optimization`, 📄 Source File: [`src/modOptimization.py`](file:///C:/Users/manue/OneDrive/Escritorio/MIA/TFM/FFMcracking/src/modOptimization.py)

### Community 30 - "📦 Descripción de Componentes"
Cohesion: 0.29
Nodes (7): 01. Diseño General del Sistema, 1. Capa de Aplicación (`samples/`), 2. Capa de Optimización (`src/modOptimization.py`), 3. Capa de Análisis FFM (`src/FFMcracking.py`), 4. Capa Abaqus CAE / ODB, 🏗️ Arquitectura por Capas, 📦 Descripción de Componentes

### Community 31 - "01. Módulo Central: FFMcracking.py"
Cohesion: 0.29
Nodes (7): 01. Módulo Central: FFMcracking.py, 1. `FFM_Material`, 2. `FFM_Crack`, 3. `FFM_InputModel`, 📄 Archivo Fuente: [`src/FFMcracking.py`](file:///C:/Users/manue/OneDrive/Escritorio/MIA/TFM/FFMcracking/src/FFMcracking.py), 🏛️ Clases del Sistema, 🛠️ Función Orquestadora: `Compute_crit_factor_FFM`

### Community 32 - "02. Extracción de Tensiones y Criterio SC"
Cohesion: 0.29
Nodes (7): 02. Extracción de Tensiones y Criterio SC, 1. Discretización Espacial, 2. Base Ortonormal Local, 3. Consulta de Componentes de Tensión, 🧮 Evaluación del Multiplicador: `evaluate_stress_criterion`, 🔍 Extracción de Tensiones en el Camino de Grieta, 📐 Procedimiento de Transformación de Coordenadas

### Community 33 - "03. Modelado XFEM y Criterio de Energía"
Cohesion: 0.29
Nodes (7): 03. Modelado XFEM y Criterio de Energía, 1. Energía Liberada ($RE$): `evaluate_RE`, 2. Energía Disipada ($DE$): `evaluate_DE`, 3. Factor Energético Crítico ($k_{EC}$):, 🔬 Balance Energético, ⚡ El Método de Elementos Finitos Extendidos (XFEM), 🛠️ Inserción de la Grieta: `generate_crack`

### Community 34 - "04. Módulo de Optimización Bayesiana"
Cohesion: 0.29
Nodes (7): 04. Módulo de Optimización Bayesiana, 1. Kernel RBF (Función de Base Radial), 2. Modelo Subrogado: Proceso Gaussiano (`gaussian_process`), 3. Función de Adquisición Dual: Expected Improvement (EI), 🔁 Algoritmo de Búsqueda: `bayesian_optimization`, 📄 Archivo Fuente: [`src/modOptimization.py`](file:///C:/Users/manue/OneDrive/Escritorio/MIA/TFM/FFMcracking/src/modOptimization.py), ⚙️ Componentes Matemáticos

### Community 35 - "📐 Mathematical Formulation of the Coupled Criterion (CC)"
Cohesion: 0.33
Nodes (6): 02. Finite Fracture Mechanics and the Coupled Criterion, 1. Stress Criterion (SC), 2. Energy Criterion (EC), 3. Coupling and Critical Crack Length $l_c$, Betti's Reciprocal Work Theorem for $RE$:, 📐 Mathematical Formulation of the Coupled Criterion (CC)

### Community 36 - "01. Requirements & Setup"
Cohesion: 0.33
Nodes (6): 01. Requirements & Setup, Option A (Recommended: Environment Variable), Option B (Local Copy), 📥 Repository Structure & Zero External Dependencies, ⚙️ Setting Up PYTHONPATH, 💻 System Prerequisites

### Community 37 - "03. Sample 2: Length Optimization"
Cohesion: 0.33
Nodes (6): 03. Sample 2: Length Optimization, 🎯 Benchmark Objective, 📝 Configuration in `example2.py`, 📈 Convergence Output, Objective Function, 🚀 Running via Terminal

### Community 38 - "❓ Frequently Asked Questions"
Cohesion: 0.33
Nodes (6): 01. Troubleshooting & FAQ, 1. Why do I encounter lock file (`.lck`) errors?, 2. Why does $k_{SC}$ return `inf` (infinity)?, 3. How to avoid out-of-memory errors during optimization?, 4. What if the mesh at the concentrator is too coarse?, ❓ Frequently Asked Questions

### Community 39 - "📐 Formulación Matemática del Criterio Acoplado (CC)"
Cohesion: 0.33
Nodes (6): 02. Mecánica de Fractura Finita y Criterio Acoplado, 1. Criterio de Tensión (Stress Criterion, SC), 2. Criterio de Energía (Energy Criterion, EC), 3. Acoplamiento y Determinación de la Longitud Crítica $l_c$, 📐 Formulación Matemática del Criterio Acoplado (CC), Teorema del Trabajo Recíproco de Betti para el cálculo de $RE$:

### Community 40 - "01. Requisitos e Instalación"
Cohesion: 0.33
Nodes (6): 01. Requisitos e Instalación, ⚙️ Configuración del Entorno de Ejecución, 📥 Estructura del Repositorio y Dependencias, Opción A (Recomendada: PYTHONPATH), Opción B (Copia local), 💻 Requisitos del Sistema

### Community 41 - "03. Ejemplo 2: Optimización de Longitud de Grieta"
Cohesion: 0.33
Nodes (6): 03. Ejemplo 2: Optimización de Longitud de Grieta, 📝 Configuración de Parámetros en `example2.py`, Definición de la Función Objetivo, 🚀 Ejecución desde la Terminal, 🎯 Objetivo de la Prueba, 📈 Salida y Análisis de Convergencia

### Community 42 - "❓ Preguntas Frecuentes"
Cohesion: 0.33
Nodes (6): 01. FAQ y Resolución de Problemas, 1. ¿Por qué obtengo un error de archivo bloqueado (`.lck`)?, 2. ¿Por qué $k_{SC}$ devuelve `inf` (infinito)?, 3. ¿Por qué se producen problemas de memoria en la optimización?, 4. ¿Qué ocurre si la malla en la entalla es demasiado gruesa?, ❓ Preguntas Frecuentes

### Community 43 - "01. Overview and Motivation"
Cohesion: 0.40
Nodes (5): 01. Overview and Motivation, 🚀 Conceptual Workflow of FFMcracking, 🛑 The Dilemma of Fracture at Stress Concentrators, 💡 The Finite Fracture Mechanics (FFM) Answer, 🎯 What is FFMcracking?

### Community 44 - "02. Sample 1: Fixed Crack"
Cohesion: 0.40
Nodes (5): 02. Sample 1: Fixed Crack, 🎯 Benchmark Objective, 📝 Configuration in `example1.py`, 📊 Result Interpretation, 🚀 Running via Terminal

### Community 45 - "01. Visión General y Motivación"
Cohesion: 0.40
Nodes (5): 01. Visión General y Motivación, 🛑 El Dilema de la Fractura en Concentradores de Tensión, 🚀 Flujo Conceptual de FFMcracking, 💡 La Respuesta de la Mecánica de Fractura Finita (FFM), 🎯 ¿Qué es FFMcracking?

### Community 46 - "02. Ejemplo 1: Grieta Fija"
Cohesion: 0.40
Nodes (5): 02. Ejemplo 1: Grieta Fija, 📝 Configuración de Parámetros en `example1.py`, 🚀 Ejecución desde la Terminal, 📊 Interpretación de Resultados, 🎯 Objetivo de la Prueba

## Knowledge Gaps
- **149 isolated node(s):** `Symptom`, `Reproduce`, `Additional Context`, `Additional Context`, `Additional Context` (+144 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FFMcracking Wiki — Official Documentation` connect `📚 Table of Contents` to `en/index.md`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `Wiki de FFMcracking — Documentación Oficial` connect `📚 Estructura Temática` to `es/index.md`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `01. Core Module: FFMcracking.py` connect `01. Core Module: FFMcracking.py` to `en/index.md`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **What connects `Symptom`, `Reproduce`, `Additional Context` to the rest of the system?**
  _149 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `FFMcracking.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14761904761904762 - nodes in this community are weakly interconnected._