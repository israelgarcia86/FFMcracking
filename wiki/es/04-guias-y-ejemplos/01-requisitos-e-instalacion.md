# Requisitos y preparación

Los ejemplos requieren Abaqus/CAE, Abaqus/Standard, acceso a una licencia y NumPy en el entorno de Python de Abaqus. La ejecución que se describe aquí se comprobó con Abaqus 2024; no se han verificado todas las versiones compatibles.

## Módulos

Utilice el repositorio completo y conserve cada carpeta de ejemplo dentro de `samples/`. Ambos ejemplos importan `src.FFMcracking` directamente. El ejemplo 2 también importa `src.modOptimization`.

## Ejecución

Seleccione como directorio de trabajo la carpeta del ejemplo antes de ejecutarlo. En Abaqus/CAE use **File → Set Work Directory** y después **File → Run Script**. En una terminal, sitúese en esa carpeta y ejecute:

```text
abaqus cae noGUI=example1.py
```

Para el segundo ejemplo, utilice `example2.py`.

Los scripts obtienen la carpeta de entrada mediante `os.getcwd()` y escriben los resultados en `compFiles/`. Si se usa otro directorio, pueden buscar el archivo `.inp` en una ubicación incorrecta. Los cálculos generan archivos en esa carpeta.

[Índice](../index.md)
