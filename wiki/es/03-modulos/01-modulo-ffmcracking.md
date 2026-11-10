# 01. Modulo Central: FFMcracking.py

Navegacion: [Portada Wiki](../index.md) > [03. Modulos](01-modulo-ffmcracking.md) > Modulo Central

================================================================================
1. PROPOSITO DEL MODULO
================================================================================

El archivo fuente `src/FFMcracking.py` constituye el nucleo operacional del framework. Agrupa las estructuras de datos que parametrizan el problema fisico y la funcion orquestadora `Compute_crit_factor_FFM()`, encargada de interactuar con el entorno de Abaqus CAE y resolver las ecuaciones del Criterio Acoplado.

================================================================================
2. CLASES DE DATOS
================================================================================

Las tres clases principales actuan como contenedores estructurados y desacoplados:

2.1. Clase FFM_Material
--------------------------------------------------------------------------------
Almacena las propiedades mecanicas de fractura del material y define los parametros de desactivacion del modelo de dano de Abaqus:

```python
class FFM_Material:
    def __init__(self, sigmac, tauc, Gc1, Gc2):
        self.sigmac = sigmac        # Resistencia a traccion (Modo I)
        self.tauc = tauc            # Resistencia a cortadura (Modo II)
        self.Gc1 = Gc1              # Tenacidad de fractura en Modo I
        self.Gc2 = Gc2              # Tenacidad de fractura en Modo II
        self.maxsigma = 5000e10 * sigmac      # Tension artificial de inicio XFEM
        self.maxd = 1.0e10 * Gc1 / sigmac     # Desplazamiento artificial limite XFEM
```

El principio del XFEM como separador geometrico:
Abaqus exige obligatoriamente definir un modelo de dano constitutivo (MaxpsDamageInitiation y DamageEvolution) para habilitar una grieta enriquecida con XFEM. Sin embargo, en la Mecanica de la Fractura Finita, no se desea que Abaqus decida el avance del dano mediante leyes cohesivas locales; toda la fisica de fractura la gobierna el Criterio Acoplado en el post-proceso.

Para resolver esto, FFMcracking asigna una tension de dano `maxsigma` de valor deliberadamente desproporcionado (50000 billones de veces mayor que sigmac). Como consecuencia, el material nunca alcanza dicha tension de fallo en el solver, el dano no se activa y la grieta XFEM se comporta de forma exacta como una discontinuidad puramente elastica con caras libres de tension.

Nota sobre Modo Mixto:
En la version actual del codigo, `sigmac` y `Gc1` son las propiedades activas en los calculos (Modo I puro). Los atributos `tauc` y `Gc2` quedan registrados en la clase como interfaz preparada para futuras extensiones hacia criterios cuadraticos o tensocortantes.

2.2. Clase FFM_Crack
--------------------------------------------------------------------------------
Define la entidad geometrica de la grieta candidata:

```python
class FFM_Crack:
    def __init__(self, tips_coordinates):
        self.tips_coordinates = tips_coordinates
```

- `tips_coordinates`: Lista de dos tuplas con las coordenadas tridimensionales de los extremos de la grieta: `[(x1, y1, z1), (x2, y2, z2)]`.
- Aunque los analisis actuales son bidimensionales (tension plana o deformacion plana), la coordenada z debe definirse como 0.0 debido a que los objetos `Path` de Abaqus requieren especificaciones 3D.

2.3. Clase FFM_InputModel
--------------------------------------------------------------------------------
Encapsula la configuracion del modelo de calculo y proporciona metodos de acceso:

```python
class FFM_InputModel:
    def __init__(self, inpFilePath, inpFileName, nameStep, outRoute, material):
        self.inpFilePath = inpFilePath
        self.inpFileName = inpFileName
        self.nameStep = nameStep
        self.outRoute = outRoute
        self.material = material
```

Metodos auxiliares:
- `extract_nameStep()`: Devuelve el nombre del paso de carga de Abaqus donde estan aplicadas las cargas a escalar.
- `extract_outRoute()`: Devuelve la ruta absoluta del directorio donde se escribiran los resultados.
- `extract_material()`: Devuelve el objeto `FFM_Material`.
- `extract_inpFileTotal()`: Concatena la ruta y el nombre del archivo de entrada `.inp`.

================================================================================
3. FUNCION PRINCIPAL: Compute_crit_factor_FFM
================================================================================

Firma:
  `Compute_crit_factor_FFM(CrackFFM, OriginalModel)`

Parametros:
- `CrackFFM` (instancia de `FFM_Crack`): Geometria de la grieta a evaluar.
- `OriginalModel` (instancia de `FFM_InputModel`): Definicion del modelo elastico de base.

Retorno:
- Lista de tres valores flotantes: `[t_CC, t_SC, t_EC]`
  - `t_CC`: Factor de carga critico acoplado `max(t_SC, t_EC)`.
  - `t_SC`: Factor critico del criterio tensional.
  - `t_EC`: Factor critico del criterio energetico.

Secuencia operativa de la funcion:
1. Configuracion inicial del entorno:
   Importa librerias de Abaqus (`abaqus`, `odbAccess`, `caeModules`), cierra bases abiertas preventivamente (`mdb.close()`) y ajusta el directorio de trabajo a `outRoute`.
2. Evaluacion del Criterio Tensional (Modelo sin grieta):
   - Comprueba si existe `<jobName>-without.odb`. Si no existe, genera el modelo en `mdb`, inserta la solicitud de resultados `EC0` y ejecuta el job.
   - Abre la base de datos ODB, extrae las tensiones sobre el camino de la grieta y calcula `t_SC = sigmac / min(sigma_nn)`.
   - Extrae los vectores de desplazamiento y reaccion nodal en el contorno del modelo continuo (Estado 1).
3. Evaluacion del Criterio Energetico (Modelo con grieta XFEM):
   - Importa un nuevo modelo en `mdb` desde el archivo `.inp`.
   - Modifica el material asignando los umbrales de dano artificiales (`maxsigma`, `maxd`).
   - Invoca a `generate_crack()` para trazar la geometria del segmento e inyectar el enriquecimiento XFEM.
   - Pesa y ejecuta el job `<jobName>-with`.
   - Abre `<jobName>-with.odb` y extrae desplazamientos y fuerzas nodales en el contorno (Estado 2).
   - Calcula el trabajo reciproco `RE` mediante el Teorema de Betti.
   - Mide la longitud efectiva de la grieta y calcula `DE = Gc1 * longitud`.
   - Determina `t_EC = sqrt(DE / |RE|)`.
4. Acoplamiento y retorno:
   - Evalua `t_CC = max(t_SC, t_EC)`.
   - Cierra todas las bases de datos ODB y el modelo MDB.
   - Devuelve la lista `[t_CC, t_SC, t_EC]`.

================================================================================
Navegacion: [Anterior: Ciclo de Simulacion](../02-arquitectura/02-ciclo-de-simulacion.md) | [Portada Wiki](../index.md) | [Siguiente: Extraccion de Tensiones](02-extraccion-tensiones.md)
================================================================================
