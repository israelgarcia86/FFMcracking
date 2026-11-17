# 02. Guia para Desarrolladores y Hoja de Ruta

Navegacion: [Portada Wiki](../index.md) > [05. FAQ y Desarrollo](01-resolucion-problemas-faq.md) > Guia para Desarrolladores

================================================================================
1. REGLAS DE DESARROLLO Y CODIFICACION
================================================================================

Para asegurar la robustez, portabilidad e integridad del framework, cualquier contribucion al codigo fuente en `src/` debe respetar estrictamente los siguientes cuatro principios:

1. Compatibilidad Dual Python 2.7 y Python 3.x:
   - Abaqus 2020 a 2023 ejecuta scripts bajo Python 2.7, mientras que Abaqus 2024 y posteriores incorporan Python 3.10+.
   - No utilice caracteristicas exclusivas de Python 3.6+ (como f-strings) ni de Python 3.10+ (como `match / case`) en los modulos de `src/`.
   - Utilice formateo con `.format()` o `%`, y asegure divisiones con decimales flotantes (`from __future__ import division` o conversion explicita `float()`).

2. Regla de Cero Dependencias Externas:
   - Los modulos `src/FFMcracking.py` y `src/modOptimization.py` no deben depender de paquetes externos ajenos al entorno estandar de Abaqus (prohibido anadir dependencias de SciPy, Scikit-Learn, Torch, SymPy o Pandas).
   - Cualquier algoritmo matematico, de optimizacion o matricial debe resolverse utilizando exclusivamente `math`, `random`, `sys`, `os` y `numpy`.

3. Gestion Rigurosa de Recursos en Abaqus:
   - Cierre siempre las bases de datos de resultados ODB mediante `odb.close()` y la base de modelos `mdb.close()` al concluir cada etapa de calculo para prevenir bloqueos de ficheros en el sistema operativo.
   - Purgue los archivos temporales (.lck, .prt, .fil) antes de lanzar nuevos jobs.

4. Generalizacion y Alcance Transversal:
   - No introduzca parches rigidos pensados unicamente para un archivo `.inp` o una malla particular.
   - Asegurese de que las transformaciones de coordenadas, nombres de pasos y busquedas nodales mantengan independencia respecto a la geometria del problema.

================================================================================
2. HOJA DE RUTA Y LINEAS DE EXTENSION
================================================================================

A continuacion se detallan las lineas prioritarias de evolucion para el framework, estructuradas segun el analisis del repositorio y la tesis doctoral de Garcia (2014):

2.1. Nivel 1: Robustez y Refactorizacion del Nucleo
--------------------------------------------------------------------------------
- Invalidacion inteligente de cache:
  Implementar la verificacion automatica de fechas de modificacion o suma de verificacion (hash) del archivo `.inp` y del nombre del paso de carga (`nameStep`). Si el archivo varia, el sistema invalidara el ODB sin grieta y regenerara el calculo base de forma transparente.
- Normalizacion de rutas y portabilidad:
  Sustituir concatenaciones de cadenas por `os.path.join()` y `os.path.normpath()` para compatibilidad multiplataforma transparente entre sistemas Windows y Linux.
- Tratamiento riguroso de compresiones:
  Comprobar de forma estricta que la tension normal minima a lo largo del segmento cumpla `min(snn) > 0` antes de proceder al cociente del criterio tensional puntual.

2.2. Nivel 2: Criterios de Fallo en Modo Mixto
--------------------------------------------------------------------------------
- Activacion de propiedades cortantes:
  Activar el uso de `tauc` y `Gc2` (ya declarados en la clase `FFM_Material`).
- Formulacion tensocortante:
  Incorporar criterios tensionales combinados sobre el plano candidato, tales como el criterio de tension equivalente de Mohr-Coulomb o el criterio cuadratico de traccion y cortante:
  (sigma_nn / sigmac)^2 + (sigma_tn / tauc)^2 >= 1
- Tenacidad en modo mixto:
  Evaluar la mixicidad modal en el vertice de grieta y aplicar leyes energeticas fenomenologicas (como la relacion empirica de Hutchinson-Suo o leyes de potencia).

2.3. Nivel 3: Optimizacion Multiparametrica (Angulo y Posicion de Inicio)
--------------------------------------------------------------------------------
- Busqueda del angulo de inicio theta:
  Permitir que la orientacion del segmento de grieta sea una variable de diseno. En entallas no simetricas o estados de cortante, la grieta se inicia formando un angulo optimo respecto al eje de referencia.
- Optimizador bidimensional:
  Extender el proceso gaussiano en `modOptimization.py` a dimensiones superiores (L, theta) para encontrar simultaneamente la orientacion mas desfavorable y la longitud critica de rotura.

2.4. Nivel 4: Fisuracion en Materiales Compuestos e Interfaces
--------------------------------------------------------------------------------
- Rotura transversal en laminados:
  Modelizacion de grietas transversales de matriz en laminados cruzados tipo cross-ply (analizado exhaustivamente en el capitulo 6 de Garcia, 2014).
- Despegue en interfaces adhesivas o bimateriales:
  Aplicacion del Criterio Acoplado a la iniciacion de despegues en uniones adhesivas y contactos fibra-matriz a microescala.

================================================================================
Navegacion: [Anterior: FAQ y Resolucion](01-resolucion-problemas-faq.md) | [Portada Wiki](../index.md)
================================================================================
