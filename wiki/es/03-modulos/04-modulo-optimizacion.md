# 04. Modulo de Optimizacion Bayesiana

Navegacion: [Portada Wiki](../index.md) > [03. Modulos](01-modulo-ffmcracking.md) > Modulo de Optimizacion

================================================================================
1. PROPOSITO Y ENTORNO MATEMATICO
================================================================================

El archivo fuente `src/modOptimization.py` implementa un algoritmo de Optimizacion Bayesiana unidimensional disenado para encontrar la longitud de grieta critica lc que minimiza la carga de rotura:

  lc = argmin_{l} t_CC(l)

Por que Optimizacion Bayesiana?
1. Evaluacion costosa: Cada evaluacion de la funcion objetivo requiere ejecutar un analisis de elementos finitos en Abaqus CAE (simulacion con grieta XFEM).
2. No diferenciabilidad: La funcion objetivo t_CC(l) = max(t_SC(l), t_EC(l)) presenta una discontinuidad en su derivada en el punto de cruce optimo. Los metodos clasicos basados en gradiente (Newton, descenso de gradiente) fallarian o resultarian inestables.
3. Rango acotado: La busqueda se acota en un intervalo admisible [L_min, L_max], explorando eficientemente el espacio de soluciones con el minimo numero de llamadas a Abaqus.

Regla de dependencias:
El modulo esta desarrollado sin librerias externas complejas (como SciPy o Scikit-Learn); unicamente utiliza los modulos estandar `math`, `random` y `numpy`. Esto garantiza portabilidad total dentro del entorno de ejecucion de Abaqus.

================================================================================
2. COMPONENTES MATEMATICOS
================================================================================

2.1. Kernel RBF (Radial Basis Function)
--------------------------------------------------------------------------------
Mide la correlacion espacial o similitud entre dos longitudes de grieta x1 y x2:

  k(x1, x2) = exp( - (x1 - x2)^2 / (2 * l_scale^2) )

En el codigo original, la escala de longitud caracteristica es fija: `length_scale = 1.0`.

2.2. Modelo Subrogado: Regresion por Proceso Gaussiano (GPR)
--------------------------------------------------------------------------------
Dado un conjunto de observaciones evaluadas X = [x_1, ..., x_n] con factores calculados y = [y_1, ..., y_n]:
1. Matriz de covarianza de Gram:
   K_ij = k(x_i, x_j) + sigma_noise^2 * delta_ij
   donde sigma_noise^2 = 1.0e-6 es un termino de regularizacion numerica (jitter) que evita el condicionamiento singular.
2. Inversion:
   Se obtiene K_inv = np.linalg.inv(K).
3. Prediccion en un punto candidato x_*:
   - Media predicha:
     mu(x_*) = k_*^T * K_inv * y
   - Varianza predicha:
     sigma^2(x_*) = k(x_*, x_*) - k_*^T * K_inv * k_*
   donde k_* = [k(x_*, x_1), ..., k(x_*, x_n)]^T.

2.3. Funcion de Adquisicion: Expected Improvement (EI)
--------------------------------------------------------------------------------
La funcion de adquisicion determina cual es el siguiente punto x que ofrece el mayor potencial de mejora sobre el mejor minimo actual y_best = min(y):

  EI(x) = (y_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)

donde:
  z = (y_best - mu(x) - xi) / sigma(x)
- Phi(z): Funcion de distribucion acumulada (CDF) de la normal estandar, implementada mediante `0.5 * (1.0 + math.erf(z / sqrt(2)))`.
- phi(z): Funcion de densidad de probabilidad (PDF) normal estandar: `exp(-0.5 * z^2) / sqrt(2 * pi)`.
- xi = 0.01: Parametro de exploracion que balancea el refinamiento de la solucion conocida con la exploracion de regiones con alta incertidumbre sigma(x).

================================================================================
3. ALGORITMO PRINCIPAL: bayesian_optimization
================================================================================

Firma:
  `bayesian_optimization(obj_func, bounds, n_iterations, n_initial=5)`

Esquema de ejecucion:
1. Muestreo inicial (Exploracion aleatoria):
   Genera `n_initial` puntos distribuidos uniformemente en el intervalo [bounds[0], bounds[1]] y evalua la funcion objetivo llamando a Abaqus.
2. Bucle de optimizacion (Iteraciones 1 a n_iterations):
   a. Ajusta el Proceso Gaussiano con todos los puntos evaluados hasta el momento.
   b. Discretiza el dominio de busqueda en una malla fina de 100 candidatos equidistantes.
   c. Evalua la funcion Expected Improvement (EI) en los 100 candidatos.
   d. Selecciona el candidato x_next con el maximo valor de EI.
   e. Lanza la simulacion de Abaqus en x_next y anade el resultado a la base de datos.
3. Retorno:
   Devuelve la pareja (x_opt, y_opt) correspondiente al punto evaluado con el menor factor critico observado.

Diagrama de flujo del algoritmo en texto:

  [Generar 5 puntos iniciales en el rango]
                  |
                  v
       [Evaluar obj_func en Abaqus]
                  |
                  v
  +---> [Ajustar Proceso Gaussiano (K, K_inv)]
  |               |
  |               v
  |     [Malla de 100 puntos candidatos]
  |               |
  |               v
  |   [Calcular Expected Improvement (EI)]
  |               |
  |               v
  |     [Seleccionar maximo de EI: x_next]
  |               |
  |               v
  |   [Ejecutar Abaqus en x_next: y_next]
  |               |
  +-- ¿Iteracion < n_iterations? (SI)
                  | (NO)
                  v
  [Retornar mejor solucion: (lc_opt, tc_opt)]

================================================================================
4. OBSERVACIONES DE DISENO Y LIMITACIONES ACTUALES
================================================================================

El analisis del modulo revela los siguientes aspectos tecnicos:
- Reproducibilidad: Las evaluaciones iniciales emplean `random.uniform()` sin semilla fijada (`random.seed()`), por lo que dos ejecuciones consecutivas pueden arrojar trayectorias de exploracion ligeramente distintas.
- Parametros fijos: La escala de longitud l_scale = 1.0 y el numero de iteraciones son estaticos; en problemas con rangos de busqueda muy pequenos o muy amplios, normalizar el dominio a [0, 1] mejora la convergencia.
- Criterio de parada: El bucle itera un numero prefijado de veces (10 iteraciones) sin comprobar convergencia por tolerancia de mejora en t_CC.

================================================================================
Navegacion: [Anterior: Modelado XFEM](03-metodo-xfem-y-energia.md) | [Portada Wiki](../index.md) | [Siguiente: Requisitos e Instalacion](../04-guias-y-ejemplos/01-requisitos-e-instalacion.md)
================================================================================
