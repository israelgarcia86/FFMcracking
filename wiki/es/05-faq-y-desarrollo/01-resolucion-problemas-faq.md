# 01. FAQ y Resolucion de Problemas

Navegacion: [Portada Wiki](../index.md) > [05. FAQ y Desarrollo](01-resolucion-problemas-faq.md) > FAQ y Resolucion de Problemas

================================================================================
1. PREGUNTAS FRECUENTES Y DIAGNOSTICO DE INCIDENCIAS
================================================================================

1.1. Error por archivo bloqueado (.lck)
--------------------------------------------------------------------------------
Sintoma:
Abaqus se detiene indicando que la base de datos o el job no pueden abrirse porque el archivo `.lck` ya existe en el directorio de salida.

Causa:
Si una simulacion previa se interrumpe de forma abrupta (por ejemplo, cancelando con Ctrl+C o por un error de sintaxis previo al cierre), Abaqus deja un fichero testigo con extension `.lck` que bloquea el acceso concurrente al modelo.

Solucion:
Elimine todos los archivos `.lck` del directorio `compFiles/` antes de relanzar el script.

En Windows (PowerShell):
```powershell
Get-ChildItem -Path compFiles -Filter *.lck | Remove-Item -Force
```

En Linux (Bash):
```bash
rm -f compFiles/*.lck
```

1.2. El multiplicador t_SC devuelve 'inf' o valores negativos
--------------------------------------------------------------------------------
Sintoma:
El criterio tensional entrega `inf` o un numero negativo no admisible.

Causas fisicas y soluciones:
- Caso 'inf': Ocurre cuando `max(snn) <= 0`. Significa que toda la trayectoria seleccionada para la grieta se encuentra bajo compresion pura. En Modo I (apertura por traccion pura), no existe tension capaz de abrir la fisura. Verifique el sentido de las cargas aplicadas en el archivo `.inp` y la orientacion del camino.
- Caso negativo: Ocurre cuando el camino atraviesa una zona traccionada cerca del concentrador pero penetra en una zona sometida a compresion. Como la funcion `evaluate_stress_criterion` toma el valor de `min(snn)`, si este es negativo, el cociente `sigmac / min(snn)` resulta negativo. Para solucionar este caso, acote la longitud maxima evaluada para que la grieta no sobrepase la frontera de traccion.

1.3. Resultados obsoletos tras modificar el archivo .inp
--------------------------------------------------------------------------------
Sintoma:
Se modifica la geometria, las cargas o el material en el archivo `.inp`, pero los resultados de `t_SC` no varian entre ejecuciones.

Causa:
Debido a la estrategia de cache del modelo continuo, si el fichero `<jobName>-without.odb` ya existe en la carpeta `outRoute`, FFMcracking no vuelve a lanzar la simulacion elástica inicial y lee el ODB antiguo.

Solucion:
Cuando modifique el fichero `.inp` o el paso de carga, borre manualmente los archivos generados en `compFiles/` para forzar un calculo limpio desde cero.

1.4. Sensibilidad numerica en el calculo de RE
--------------------------------------------------------------------------------
Sintoma:
Para longitudes de grieta extremadamente cortas (por ejemplo, l < 0.05 mm), los valores de RE o t_EC presentan oscilaciones o signos anomalos.

Causa:
El termino de reciprocidad de Betti (RE) calcula la diferencia entre los productos de fuerza y desplazamiento en el estado con grieta y sin grieta. Cuando la fisura es diminuta, ambos estados son casi identicos y la resta cancela cifras significativas.

Solucion:
- Evite fijar cotas inferiores de busqueda (L_min) inferiores a un elemento de malla en el frente de grieta.
- Configure los trabajos de Abaqus para resolver con doble precision (`precision=DOUBLE_PLUS_PACK`).

1.5. Colisiones de sesiones o recursos sucios en CAE
--------------------------------------------------------------------------------
Sintoma:
Errores al invocar `session.Path` o `session.viewports` indicando que el elemento ya existe o no se encuentra disponible.

Causa:
Interrupciones intermedias pueden dejar objetos huerfanos en la memoria de la sesion grafica.

Solucion:
Asegurese de que los scripts se ejecuten siempre en modo no interactivo (`abaqus cae noGUI=...`), lo que garantiza un entorno de memoria limpio e independiente en cada ejecucion.

================================================================================
Navegacion: [Anterior: Ejemplo 2](../04-guias-y-ejemplos/03-ejemplo-2-optimizacion-longitud.md) | [Portada Wiki](../index.md) | [Siguiente: Guia de Desarrollo](02-guia-desarrollo-y-extensiones.md)
================================================================================
