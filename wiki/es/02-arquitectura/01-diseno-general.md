# 01. Diseno General del Sistema

Navegacion: [Portada Wiki](../index.md) > [02. Arquitectura](01-diseno-general.md) > Diseno General

================================================================================
1. ARQUITECTURA POR CAPAS
================================================================================

FFMcracking esta disenado mediante una estructura modular de cuatro capas con responsabilidades desacopladas. Esta organizacion garantiza que la definicion del caso por parte del usuario, los algoritmos matematicos de optimizacion, el analisis de fractura y la ejecucion en Abaqus operen de forma transparente e independiente.

Diagrama de capas en texto:

  +---------------------------------------------------------------+
  | 1. CAPA DE APLICACION (samples/example1.py, example2.py)       |
  |    - Define geometria (.inp), material (sigmac, Gc1) y grieta |
  +---------------------------------------------------------------+
                                 |
                                 v
  +---------------------------------------------------------------+
  | 2. CAPA DE OPTIMIZACION (src/modOptimization.py)              |
  |    - Proceso Gaussiano (GPR) + Kernel RBF                     |
  |    - Expected Improvement (EI) para minimizar t_CC(l)         |
  +---------------------------------------------------------------+
                                 |
                                 v
  +---------------------------------------------------------------+
  | 3. CAPA DE ANALISIS FFM (src/FFMcracking.py)                  |
  |    - Clases: FFM_Material, FFM_Crack, FFM_InputModel          |
  |    - Orquestador: Compute_crit_factor_FFM()                   |
  |    - Criterio SC: extractStresses, evaluate_stress_criterion  |
  |    - Criterio EC: generate_crack (XFEM), evaluate_RE, DE      |
  +---------------------------------------------------------------+
                                 |
                                 v
  +---------------------------------------------------------------+
  | 4. KERNEL DE ELEMENTOS FINITOS (Abaqus CAE / Standard)        |
  |    - Model Database (mdb) y Result Database (odbAccess)       |
  |    - Resolucion de sistemas de ecuaciones lineales            |
  +---------------------------------------------------------------+

Diagrama Mermaid:
```mermaid
graph TD
    subgraph Layer1["1. Capa de Aplicacion (samples/)"]
        Ex1["example1.py (Grieta Fija)"]
        Ex2["example2.py (Optimizacion lc)"]
    end

    subgraph Layer2["2. Capa de Optimizacion (modOptimization.py)"]
        BO["bayesian_optimization()"]
        EI["expected_improvement()"]
        GP["gaussian_process()"]
        RBF["rbf_kernel()"]
    end

    subgraph Layer3["3. Capa de Analisis FFM (FFMcracking.py)"]
        CC["Compute_crit_factor_FFM()"]
        ExtractS["extractStressesBetween2points()"]
        EvalSC["evaluate_stress_criterion()"]
        CrackGen["generate_crack() [XFEM]"]
        EvalRE["evaluate_RE() [Betti]"]
        EvalDE["evaluate_DE()"]
    end

    subgraph Layer4["4. Kernel Abaqus CAE / Standard"]
        MdbEngine["mdb (Model Database API)"]
        OdbEngine["openOdb() (Result Database API)"]
        StandardSolver["Abaqus/Standard Solver"]
    end

    Ex1 -->|Llamada directa| CC
    Ex2 -->|Define objetivo t_CC(l)| BO
    BO -->|Evalua candidatos l| CC
    BO --> EI --> GP --> RBF

    CC --> ExtractS & EvalSC & CrackGen & EvalRE & EvalDE
    CC --> MdbEngine & StandardSolver & OdbEngine
```

================================================================================
2. DESCRIPCION DETALLADA DE COMPONENTES
================================================================================

2.1. Capa de Aplicacion (Directorio samples/)
--------------------------------------------------------------------------------
Contiene los puntos de entrada ejecutables para el usuario final:
- `example1.py`: Resuelve un problema directo donde la grieta candidata posee una longitud y orientacion conocidas de antemano.
- `example2.py`: Resuelve el problema inverso/optimo: define un intervalo de busqueda [L_min, L_max] y busca la longitud critica lc que minimiza la carga admisible t_CC.

2.2. Capa de Optimizacion (src/modOptimization.py)
--------------------------------------------------------------------------------
Implementa un motor de Optimizacion Bayesiana ligero y completamente autocontenido:
- No requiere dependencias de terceros (como SciPy o Scikit-Learn); opera unicamente con los modulos basicos de Python y NumPy disponibles en cualquier instalacion de Abaqus.
- Modela la funcion objetivo no diferenciable y ruidosa t_CC(l) mediante un Proceso Gaussiano.
- Emplea la funcion de adquisicion Expected Improvement (EI) para balancear la exploracion de zonas con alta incertidumbre con la explotacion de los minimos descubiertos.

2.3. Capa de Analisis FFM (src/FFMcracking.py)
--------------------------------------------------------------------------------
Constituye el nucleo del framework y desempena dos funciones criticas:
1. Contenedores de datos estructurados:
   - `FFM_Material`: Almacena propiedades de resistencia (sigmac, tauc) y tenacidad (Gc1, Gc2), junto con las constantes de desactivacion del dano XFEM (maxsigma, maxd).
   - `FFM_Crack`: Representa la trayectoria geometrica del segmento de grieta (coordenadas tridimensionales de extremos).
   - `FFM_InputModel`: Centraliza las rutas de ficheros (.inp, carpetas de salida), el paso de carga y el enlace al material.
2. Logica de evaluacion de fractura:
   - Orquesta la generacion de los modelos Abaqus, la presentacion de jobs y la lectura de resultados ODB.
   - Aplica las rutinas de proyeccion tensional y el calculo de reciprocidad de Betti para entregar la terna [t_CC, t_SC, t_EC].

2.4. Kernel Abaqus CAE / Standard
--------------------------------------------------------------------------------
Es el motor computacional de elementos finitos subyacente. FFMcracking interactua con el a traves de su API de scripting en Python:
- Crea modelos a partir de archivos de texto .inp (`mdb.ModelFromInputFile`).
- Introduce solicitudes de salida de campo (`FieldOutputRequest`) necesarias para capturar tensiones y reacciones en nodos.
- Inserta interacciones cinematicas XFEM sin necesidad de remallado conformal.
- Ejecuta los solvers de Abaqus/Standard de forma no interactiva (`job.submit(waitForCompletion=True)`).

================================================================================
Navegacion: [Anterior: Mecanica de Fractura Finita](../01-fundamentos/02-mecanica-fractura-finita.md) | [Portada Wiki](../index.md) | [Siguiente: Ciclo de Simulacion](02-ciclo-de-simulacion.md)
================================================================================
