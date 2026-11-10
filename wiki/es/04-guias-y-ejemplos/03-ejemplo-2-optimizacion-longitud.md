# 03. Ejemplo 2: Optimizacion de Longitud de Grieta

Navegacion: [Portada Wiki](../index.md) > [04. Guias Practicas](01-requisitos-e-instalacion.md) > Ejemplo 2: Optimizacion de Longitud

================================================================================
1. OBJETIVO DEL CASO DE ESTUDIO
================================================================================

El script `samples/sample2_OpenHoleBasicOptimLenght/example2.py` ilustra la aplicacion mas potente de la Mecanica de la Fractura Finita: descubrir de forma totalmente autonoma la longitud critica con la que se iniciara la grieta (lc) y la carga real de rotura de la estructura (tc).

En un diseno real, la longitud inicial de grieta es una incognita a priori. Por tanto, el problema fisico exige encontrar el minimo de la envolvente acoplada:

  lc = argmin_{l en [L_min, L_max]} t_CC(l)

================================================================================
2. CONFIGURACION DE PARAMETROS EN example2.py
================================================================================

Revise las variables principales en `samples/sample2_OpenHoleBasicOptimLenght/example2.py`:

```python
###########################   CONFIGURATION   ################################
inpFilePath = 'C:/Users/usuario/FFMcracking/samples/sample2_OpenHoleBasicOptimLenght/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = 'C:/Users/usuario/FFMcracking/samples/sample2_OpenHoleBasicOptimLenght/compFiles/'

# Propiedades de fallo del material
sigmac = 70.0    # Resistencia a traccion (MPa)
Gc1 = 0.4        # Tenacidad de fractura Modo I (N/mm)

# Rango admisible para la busqueda de la longitud L (mm)
L_min = 1.0e-1   # 0.1 mm
L_max = 5.0      # 5.0 mm
bounds = (L_min, L_max)

# Numero de iteraciones del optimizador Bayesiano
n_iterations = 10
################################################==============================
```

Definicion de la funcion objetivo:
```python
def objective(L):
    # La grieta nace en el borde del taladro (x=1.0) y avanza una distancia L
    coordinates_1 = [(1.0, 0.0, 0.0), (1.0 + L, 0.0, 0.0)]
    Crack1 = FFM_Crack(coordinates_1)
    # Evalua el factor acoplado en Abaqus
    return Compute_crit_factor_FFM(Crack1, Model1)[0]
```

================================================================================
3. EJECUCION DE LA OPTIMIZACION
================================================================================

Ejecute el script en su terminal mediante el interprete de Abaqus:

```powershell
abaqus cae noGUI=example2.py
```

Cronologia del calculo:
1. Muestreo inicial (5 evaluaciones):
   El algoritmo evalua 5 longitudes distribuidas aleatoriamente en [0.1, 5.0].
   - En la primera evaluacion se ejecuta el calculo sin grieta (`-without.odb`).
   - En las siguientes evaluaciones, el fichero sin grieta se lee directamente de cache.
2. Bucle Bayesiano (10 iteraciones):
   - Con los datos acumulados, se ajusta el Proceso Gaussiano.
   - La funcion Expected Improvement (EI) identifica la longitud candidata mas prometedora.
   - Se ejecuta el analisis con grieta XFEM en Abaqus para dicha longitud.
   - Se actualizan la media y la incertidumbre del modelo subrogado.
3. Convergencia:
   Tras 15 evaluaciones totales (5 iniciales + 10 iteraciones), el optimizador entrega el minimo global observado.

================================================================================
4. ANALISIS DE RESULTADOS
================================================================================

La ejecucion imprime el historial de puntos y concluye con el optimo predicho:

```text
--- RESULTADOS DE LA OPTIMIZACION FFM ---
Longitud critica de inicio (lc): 1.14 mm
Factor de carga critico (tc):    1.628
```

Interpretacion fisica y significado ingenieril:
1. Longitud critica lc = 1.14 mm:
   El modelo predice que cuando la estructura falle, no se formara una microgrieta infinitesimal de micras ni una fisura que atraviese la pieza; la fractura se iniciara bruscamente mediante un salto discontinuo de exactamente 1.14 mm.
2. Factor de carga critico tc = 1.628:
   La carga nominal aplicada en el archivo `.inp` debera multiplicarse por 1.628 para provocar la rotura inicial de la probeta.
3. En el punto optimo se verifica el balance fisico:
   t_SC(1.14) aprox t_EC(1.14) aprox 1.628. La capacidad portante del componente queda rigurosamente determinada por la conjuncion de la resistencia estatica y la tenacidad de fractura.

Recomendacion practica de verificacion (Red de seguridad):
Antes de realizar cualquier modificacion o refactorizacion en el codigo fuente, ejecute y guarde los resultados numericos de `example1.py` y `example2.py`. Serviran como prueba de regresion para constatar que las optimizaciones de codigo respetan escrupulosamente los valores fisicos de referencia.

================================================================================
Navegacion: [Anterior: Ejemplo 1](02-ejemplo-1-grieta-fija.md) | [Portada Wiki](../index.md) | [Siguiente: FAQ y Resolucion de Problemas](../05-faq-y-desarrollo/01-resolucion-problemas-faq.md)
================================================================================
