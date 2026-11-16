# Ejemplo 2: optimización de la longitud de grieta

El script `samples/sample2_OpenHoleBasicOptimLenght/example2.py` busca el menor factor acoplado entre longitudes de grieta candidatas. La grieta comienza en `(1.0, 0.0, 0.0)` y su segundo extremo se define como `(1.0 + L, 0.0, 0.0)`. El intervalo configurado es `L_min = 0.1` a `L_max = 5.0`, en las unidades del modelo.

## Preparación y ejecución

Copie `FFMcracking.py` y `modOptimization.py` desde `src/` a la carpeta del ejemplo, o configure la ruta de módulos. Seleccione la carpeta de la muestra como directorio de trabajo y ejecute `example2.py` desde Abaqus/CAE. Desde una terminal situada en la misma carpeta puede usar `abaqus cae noGUI=example2.py`. Los resultados se escriben en `compFiles/`.

## Optimización e interpretación

La función objetivo devuelve los factores acoplado, tensional y energético. El algoritmo evalúa cinco longitudes iniciales aleatorias y otras diez candidatas. Reutiliza el ODB sin grieta si ya existe; cada nueva longitud requiere un análisis con grieta. La salida incluye el mejor valor observado y los puntos evaluados.

En una ejecución con Abaqus 2024 se obtuvo un factor mínimo observado de **3,7411** para `L ≈ 2,0303`. Los puntos iniciales son aleatorios; el resultado puede variar y no garantiza el mínimo global. Estas cifras describen esa ejecución, no una validación experimental. Consulte [el módulo de optimización](../03-modulos/04-modulo-optimizacion.md) para conocer el procedimiento y sus límites.

Si cambia el modelo o el paso de carga, use una salida limpia o retire el ODB sin grieta anterior tras verificar que no está en uso.

[Índice](../index.md) · [Ejemplo 1](02-ejemplo-1-grieta-fija.md)
