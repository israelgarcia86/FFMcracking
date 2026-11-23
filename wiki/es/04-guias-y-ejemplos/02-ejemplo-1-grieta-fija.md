# Ejemplo 1: grieta fija

El ejemplo `samples/sample1_OpenHoleCrackFixed/example1.py` calcula el factor de carga para una grieta recta definida por los puntos `(1.0, 0.0, 0.0)` y `(4.5, 0.0, 0.0)`. Para esta geometría, la longitud candidata es 3,5 en las unidades del modelo. Las propiedades configuradas son `sigmac = 70.0` y `Gc1 = 0.4`.

## Preparación y ejecución

1. Compruebe que Abaqus/CAE y Abaqus/Standard pueden acceder a la licencia. Conéctese a la red de licencias o a la VPN si su instalación lo requiere.
2. Conserve `samples/sample1_OpenHoleCrackFixed/` dentro del repositorio completo. La carpeta debe contener `example1.py` y `OpenHoleLongDEF-fmesh-00.inp`. El script importa `src.FFMcracking` directamente.
3. Abra Abaqus/CAE y cree una base de modelos Standard/Explicit. Guarde antes los trabajos que necesite: el script cierra la base de modelos activa.
4. En **File → Set Work Directory**, escriba la ruta absoluta de la carpeta del ejemplo en su ordenador y pulse **OK**. Elegir el script en el paso siguiente no cambia por sí mismo el directorio de trabajo.
5. En **File → Run Script**, seleccione `example1.py` y pulse **OK**. Espere a que finalicen los análisis sin grieta y con grieta. El script imprime `The critical factor is: ...` en el área de mensajes de Abaqus; el archivo de repetición `abaqus.rpy` también puede recoger la salida.

Como alternativa, abra una terminal en la carpeta del ejemplo y ejecute `abaqus cae noGUI=example1.py`.

El script toma la ruta de entrada del directorio de trabajo y usa `compFiles/` para los resultados. Compruebe que allí está `OpenHoleLongDEF-fmesh-00.inp` y que el paso `Loading` existe en el modelo.

## Qué calcula

El programa obtiene las tensiones del modelo sin grieta, calcula el criterio tensional, ejecuta el modelo con grieta XFEM y evalúa el criterio energético. La función devuelve `[t_CC, t_SC, t_EC]`, pero el script imprime únicamente `t_CC = max(t_SC, t_EC)`.

En una ejecución con Abaqus 2024, el factor impreso fue aproximadamente **3,9020**. Es un resultado de esa configuración, no un valor universal ni una validación experimental. Revise las unidades y los datos del modelo antes de interpretar el resultado.

El cálculo sin grieta puede reutilizarse desde `compFiles/`. El script también elimina archivos anteriores que coincidan con el patrón del trabajo con grieta en esa carpeta. Para conservar resultados previos, utilice una copia separada de la carpeta del ejemplo dentro de `samples/`; si cambia el `.inp` o el paso de carga, utilice una salida limpia.

## Visualización del resultado

El script exporta `crack-s22.png` y `crack-opening-s22.png` a `compFiles/visualization/`. La primera imagen muestra S22 sobre la geometría sin deformar en el último incremento del paso `Loading`. La segunda muestra el mismo campo de tensiones sobre la geometría deformada, con una escala de visualización de 150× para apreciar la abertura XFEM. El campo corresponde a la carga de referencia del análisis. La escala solo cambia la forma mostrada.

Al ejecutar el script desde Abaqus/CAE, la segunda vista queda en pantalla. La ejecución desde la terminal también exporta las imágenes. Para inspeccionar el resultado más adelante, abra `compFiles/visualization/crack.odb` en Abaqus/CAE.

[Índice](../index.md) · [Ejemplo 2](03-ejemplo-2-optimizacion-longitud.md)
