# Ejemplo 1: grieta fija

El ejemplo `samples/sample1_OpenHoleCrackFixed/example1.py` calcula el factor de carga para una grieta recta definida por los puntos `(1.0, 0.0, 0.0)` y `(4.5, 0.0, 0.0)`. Para esta geometría, la longitud candidata es 3,5 en las unidades del modelo. Las propiedades configuradas son `sigmac = 70.0` y `Gc1 = 0.4`.

## Preparación y ejecución

Copie `src/FFMcracking.py` a la carpeta del ejemplo o configure la ruta de búsqueda de módulos. Seleccione dicha carpeta como directorio de trabajo en Abaqus/CAE mediante **File → Set Work Directory** y ejecute `example1.py` con **File → Run Script**. También puede situarse en esa carpeta y ejecutar `abaqus cae noGUI=example1.py`.

El script toma la ruta de entrada del directorio de trabajo y usa `compFiles/` para los resultados. Compruebe que allí está `OpenHoleLongDEF-fmesh-00.inp` y que el paso `Loading` existe en el modelo.

## Qué calcula

El programa obtiene las tensiones del modelo sin grieta, calcula el criterio tensional, ejecuta el modelo con grieta XFEM y evalúa el criterio energético. La función devuelve `[t_CC, t_SC, t_EC]`, pero el script imprime únicamente `t_CC = max(t_SC, t_EC)`.

En una ejecución con Abaqus 2024, el factor impreso fue aproximadamente **3,9020**. Es un resultado de esa configuración, no un valor universal ni una validación experimental. Revise las unidades y los datos del modelo antes de interpretar el resultado.

El cálculo sin grieta puede reutilizarse desde `compFiles/`. Si cambia el `.inp` o el paso de carga, utilice una carpeta de salida limpia o retire el ODB anterior después de comprobar que ninguna ejecución lo está utilizando.

[Índice](../index.md) · [Ejemplo 2](03-ejemplo-2-optimizacion-longitud.md)
