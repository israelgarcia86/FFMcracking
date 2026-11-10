# 01. Requisitos e Instalacion

Navegacion: [Portada Wiki](../index.md) > [04. Guias Practicas](01-requisitos-e-instalacion.md) > Requisitos e Instalacion

================================================================================
1. REQUISITOS DEL SISTEMA
================================================================================

Para ejecutar las simulaciones de FFMcracking, el entorno de trabajo debe cumplir los siguientes requisitos tecnicos:

1. Suite de Elementos Finitos Abaqus:
   - Requiere Abaqus CAE y Abaqus/Standard (version 2020 o superior recomendada).
   - El ejecutable `abaqus` debe estar registrado en las variables de entorno del sistema (PATH) para poder invocarse desde terminal.
   - Nota tecnica: Versiones muy antiguas (como Abaqus 6.14) presentaban restricciones de memoria y limites en los grados de libertad enriquecidos mediante XFEM.

2. Interprete de Python:
   - No es necesario instalar un interprete de Python independiente.
   - Las simulaciones se ejecutan exclusivamente a traves del interprete embebido en Abaqus mediante el comando:
     `abaqus cae noGUI=nombre_script.py`
   - Segun la version instalada de Abaqus, el interprete interno sera Python 2.7 (en versiones 2020 a 2023) o Python 3.10+ (en Abaqus 2024 y posteriores). El codigo base de FFMcracking esta preparado para ser compatible con ambas versiones.

3. Sistema Operativo:
   - Microsoft Windows 10/11 (64 bits) o distribuciones Linux certificadas por Dassault Systemes (RHEL, Rocky Linux, SUSE, Ubuntu con dependencias oficiales).

================================================================================
2. DEPENDENCIAS DE SOFTWARE
================================================================================

FFMcracking sigue una estricta politica de CERO DEPENDENCIAS EXTERNAS:
- No requiere librerias pesadas de ciencia de datos (como SciPy, Scikit-Learn, PyTorch o Pandas).
- Todo el framework esta construido sobre las librerias nativas de Python (`math`, `random`, `os`, `sys`) y la libreria de calculo matricial `numpy` (incluida de serie en cualquier instalacion de Abaqus).

Estructura de directorios del framework:
```text
FFMcracking/
|-- src/
|   |-- FFMcracking.py         # Modulo central de calculo y enlace con Abaqus
|   +-- modOptimization.py     # Algoritmo de optimizacion Bayesiana
+-- samples/
    |-- sample1_OpenHoleCrackFixed/
    |   |-- example1.py        # Calculo con grieta de tamano prefijado
    |   +-- OpenHoleLongDEF-fmesh-00.inp
    +-- sample2_OpenHoleBasicOptimLenght/
        |-- example2.py        # Calculo con optimizacion de longitud lc
        +-- OpenHoleLongDEF-fmesh-00.inp
```

================================================================================
3. CONFIGURACION DEL ENTORNO DE EJECUCION
================================================================================

Para que el interprete de Abaqus localice los modulos en `src/` al ejecutar los ejemplos, se recomiendan dos procedimientos:

3.1. Metodo A: Configuracion de PYTHONPATH (Recomendado)
--------------------------------------------------------------------------------
Se anade la ruta absoluta del directorio `src/` a la variable de entorno `PYTHONPATH` antes de lanzar el calculo.

En Windows (PowerShell):
```powershell
$env:PYTHONPATH = "C:\ruta\al\repositorio\FFMcracking\src;" + $env:PYTHONPATH
```

En Linux (Bash):
```bash
export PYTHONPATH="/ruta/al/repositorio/FFMcracking/src:$PYTHONPATH"
```

3.2. Metodo B: Copia local de modulos
--------------------------------------------------------------------------------
Alternativamente, puede copiarse `FFMcracking.py` (y `modOptimization.py` en el caso del ejemplo 2) directamente en la misma carpeta donde reside el script `example1.py` o `example2.py`.

================================================================================
4. VERIFICACION PREVIA A LA EJECUCION
================================================================================

Antes de iniciar una simulacion, verifique en su consola que el comando `abaqus` responde correctamente:

```powershell
abaqus whereami
```

Si el sistema devuelve la ruta del ejecutable y la version de Abaqus, el entorno esta listo para proceder con las simulaciones guiadas.

================================================================================
Navegacion: [Anterior: Modulo de Optimizacion](../03-modulos/04-modulo-optimizacion.md) | [Portada Wiki](../index.md) | [Siguiente: Ejemplo 1 Grieta Fija](02-ejemplo-1-grieta-fija.md)
================================================================================
