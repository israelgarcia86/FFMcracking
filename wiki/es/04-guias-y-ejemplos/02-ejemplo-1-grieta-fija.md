# 02. Ejemplo 1: Grieta Fija

Navegacion: [Portada Wiki](../index.md) > [04. Guias Practicas](01-requisitos-e-instalacion.md) > Ejemplo 1: Grieta Fija

================================================================================
1. OBJETIVO DEL CASO DE ESTUDIO
================================================================================

El script `samples/sample1_OpenHoleCrackFixed/example1.py` resuelve el problema directo del Criterio Acoplado: calcula el factor de carga critico admisible para una grieta de posicion y longitud previamente fijadas.

El modelo de elementos finitos corresponde a una placa plana con un taladro circular central sometida a traccion uniaxil (`OpenHoleLongDEF-fmesh-00.inp`).
- Radio del agujero: R = 1.0 mm.
- Grieta candidata: Segmento recto que nace en el borde del taladro en (1.0, 0.0, 0.0) y avanza horizontalmente a lo largo del ligamento hasta (4.5, 0.0, 0.0).
- Longitud evaluada: L = 3.5 mm.

================================================================================
2. CONFIGURACION DE PARAMETROS EN example1.py
================================================================================

Abra el fichero `samples/sample1_OpenHoleCrackFixed/example1.py` con su editor de texto y revise el bloque de configuracion:

```python
###########################   CONFIGURATION   ################################
# 1. Definicion de rutas del modelo
inpFilePath = 'C:/Users/usuario/FFMcracking/samples/sample1_OpenHoleCrackFixed/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = 'C:/Users/usuario/FFMcracking/samples/sample1_OpenHoleCrackFixed/compFiles/'

# 2. Propiedades mecanicas y de fractura del material
sigmac = 70.0    # Resistencia a traccion admisible (MPa)
Gc1 = 0.4        # Tenacidad critica a fractura en Modo I (N/mm)

# 3. Geometria de la grieta fija (longitud L = 3.5 mm)
coordinates_1 = [(1.0, 0.0, 0.0), (4.5, 0.0, 0.0)]
################################################==============================
```

Recomendaciones sobre las rutas:
- Asegurese de que la ruta en `outRoute` termine siempre con una barra separadora `/` o `\\`.
- Si el directorio `compFiles/` no existe, FFMcracking lo creara de forma automatica.

================================================================================
3. EJECUCION DE LA SIMULACION
================================================================================

Abra una consola en el directorio de `sample1_OpenHoleCrackFixed` y ejecute:

En Windows (PowerShell / CMD):
```powershell
abaqus cae noGUI=example1.py
```

En Linux:
```bash
abaqus cae noGUI=example1.py
```

Flujo interno durante la ejecucion:
1. El script importa `FFMcracking.py` y construye los objetos `FFM_Material`, `FFM_Crack` y `FFM_InputModel`.
2. Llama a `Compute_crit_factor_FFM()`.
3. Abaqus ejecuta el calculo elastico continuo y genera `OpenHoleLongDEF-fmesh-00-without.odb`.
4. Se extrae el perfil de tracciones normales en el segmento [1.0, 4.5] y se calcula `t_SC`.
5. Se inyecta la grieta con XFEM y se ejecuta el trabajo `OpenHoleLongDEF-fmesh-00-with.odb`.
6. Se evalua el trabajo reciproco de Betti RE y la energia disipada DE, obteniendo `t_EC`.
7. Se acoplan ambos factores imprimiendo el resultado final en la consola.

================================================================================
4. INTERPRETACION DE RESULTADOS
================================================================================

Al finalizar el script, la consola muestra la terna calculada:

```text
[k_CC, k_SC, k_EC] = [1.8742, 1.8742, 1.3210]
```
(Los valores numericos exactos dependeran de la malla y condiciones del .inp).

Significado fisico de cada valor:
- `k_SC = 1.8742`: Multiplicador necesario para que la tension normal en el vertice mas alejado (x = 4.5) alcance los 70 MPa requeridos para romper los enlaces del material.
- `k_EC = 1.3210`: Multiplicador necesario para que el sistema libere suficiente energia potencial como para pagar el coste superficial de abrir 3.5 mm de grieta (DE = 0.4 * 3.5 = 1.4 mJ).
- `k_CC = 1.8742`: Como rige `max(k_SC, k_EC)`, el factor limitante para esta longitud concreta de 3.5 mm es el criterio tensional. A una carga inferior a 1.8742, la grieta no podria alcanzar los 3.5 mm aunque hubiera energia disponible.

================================================================================
Navegacion: [Anterior: Requisitos e Instalacion](01-requisitos-e-instalacion.md) | [Portada Wiki](../index.md) | [Siguiente: Ejemplo 2 Optimizacion](03-ejemplo-2-optimizacion-longitud.md)
================================================================================
