# Optimización bayesiana

`src/modOptimization.py` busca el menor factor acoplado observado en un intervalo de longitudes. Su función `bayesian_optimization(obj_func, bounds, n_iterations, n_initial=5)` espera que `obj_func` devuelva `[t_CC, t_SC, t_EC]` y retorna la mejor longitud, el mejor factor y las listas de longitudes y factores evaluados.

## Procedimiento

1. Se generan `n_initial` longitudes aleatorias uniformes en el intervalo y se evalúa cada una.
2. Para cada iteración posterior se predicen por separado los criterios tensional y energético mediante dos procesos gaussianos. Cada uno usa un núcleo RBF con escala de longitud 1,0 y una regularización diagonal de `1e-6`.
3. En cada candidato, el código toma el máximo de las dos medias y estima una dispersión combinada a partir del máximo de sus límites inferiores `media - desviación`. Con ellos calcula una adquisición de mejora esperada, con `xi = 0.01`.
4. Elige el mayor valor de adquisición entre 100 candidatos equiespaciados, evalúa la función objetivo y añade las observaciones.

## Modelo y función de adquisición

Para cada criterio, el núcleo es `k(x, x') = exp(-(x - x')² / (2 · 1,0²))`. Con la matriz `K` de covarianzas más `1e-6` en la diagonal, el código calcula la media `μ = k*ᵀ K⁻¹ y` y la varianza `v = max(0, k(x*, x*) - k*ᵀ K⁻¹ k*)` en cada candidato.

Sean `μ_SC`, `μ_EC` y `σ_SC`, `σ_EC` las predicciones y desviaciones de ambos modelos. El código usa `μ = max(μ_SC, μ_EC)` y `σ = |μ - max(μ_SC - σ_SC, μ_EC - σ_EC)|`. Con el mejor factor observado `y_best` y `ξ = 0,01`, calcula `z = (y_best - μ - ξ) / σ` cuando `σ > 0`, y `z = 0` en otro caso. La adquisición es `EI = (y_best - μ - ξ) Φ(z) + σ φ(z)`, donde `Φ` y `φ` son la distribución acumulada y la densidad normal estándar. Esta combinación de desviaciones es la aproximación implementada; no representa necesariamente la incertidumbre exacta del máximo de los dos procesos.

El ejemplo 2 usa cinco puntos iniciales y diez iteraciones adicionales. La búsqueda devuelve el mínimo **observado**, no garantiza un mínimo global ni comprueba la convergencia. Los puntos iniciales no tienen semilla fijada, por lo que las ejecuciones pueden diferir. El núcleo y la malla de candidatos también condicionan la exploración. El código usa `math`, `random` y NumPy.

[Índice](../index.md) · [Ejemplo 2](../04-guias-y-ejemplos/03-ejemplo-2-optimizacion-longitud.md)
