# 03. Modelado XFEM y Criterio de Energia

Navegacion: [Portada Wiki](../index.md) > [03. Modulos](01-modulo-ffmcracking.md) > Modelado XFEM y Criterio de Energia

================================================================================
1. FUNCION DE XFEM EN EL FRAMEWORK
================================================================================

Para evaluar el decremento de energia potencial (-Delta Pi), es necesario conocer la respuesta elastica de la estructura tras la aparicion de la grieta candidata. 

En enfoques convencionales de elementos finitos, evaluar distintas longitudes de grieta exigiria remallar conformalmente la geometria en cada iteracion. FFMcracking elude por completo el remallado empleando el Metodo de los Elementos Finitos Extendidos (XFEM).

Mediante la particion de la unidad, XFEM enriquece los grados de libertad de los elementos que son atravesados por la discontinuidad:
- Funciones de salto de Heaviside: representan el salto cinematico de desplazamientos entre las caras opuestas de la fisura.
- Funciones asintoticas del vertice: modelan el campo singular elastoestatico en el frente de grieta.

Papel conceptual de XFEM: "Separador geometrico", no ley de dano.
El software introduce propiedades de dano ficticias deliberadamente inalcanzables (`maxsigma = 5000e10 * sigmac`). Por consiguiente:
- El material permanece rigurosamente en regimen elastico lineal.
- Las caras de la grieta quedan completamente libres de tracciones superficiales.
- XFEM actua unicamente como un mecanismo cinematico que introduce la hendidura geometrica sin coste de remallado.

================================================================================
2. RUTINA DE INSERCION: generate_crack
================================================================================

Firma:
  `generate_crack(crackObject, modelObject, session, regionToolset)`

Secuencia de operaciones en Abaqus CAE:
1. Creacion del boceto (`ConstrainedSketch`):
   Traza el segmento recto 2D delimitado por `point1` y `point2`.
2. Generacion de la pieza tipo alambre (`BaseWire`):
   Construye una entidad deformable unidimensional que representa la fisura.
3. Instanciacion en el ensamblaje:
   Inserta `Crack-Instance` en `rootAssembly`.
4. Definicion del dominio enriquecido (`crackDomain`):
   Selecciona los elementos de la malla estructural sobre los que se habilitaran las funciones enriquecidas de XFEM.
5. Creacion del Feature de ingenieria:
   Ejecuta `a.engineeringFeatures.XFEMCrack(name='Crack-1', crackDomain=crackDomain, crackLocation=crackLocation)`.

Nota tecnica sobre modelos multi-instancia:
En el codigo original, la asignacion de elementos para `crackDomain` toma los elementos de la ultima instancia iterada (`elements = e1`). Si un modelo esta compuesto por multiples instancias, el dominio debe construirse acumulando todas las instancias (`elements1`), aspecto clave al extender el framework a ensamblajes complejos.

================================================================================
3. TEOREMA DE RECIPROCIDAD DE BETTI Y CALCULO DE RE
================================================================================

El termino RE representa la energia elastica liberada a factor de carga unitario:
  RE = -Delta Pi(1)

3.1. Fundamentacion matematica
--------------------------------------------------------------------------------
En un cuerpo elastico lineal, la variacion incremental de energia potencial entre el estado continuo 1 y el estado fisurado 2 puede formularse a traves de la integral de contorno del Teorema de Reciprocidad de Betti (Garcia, 2014, cap. 3):

  RE = 0.5 * integral_{contorno} [ u_i^(2) * T_i^(1) - u_i^(1) * T_i^(2) ] dS

donde u denota desplazamiento y T traccion superficial exterior.

Puesto que sobre los contornos libres de carga las tracciones son nulas (T = 0), y en el interior del solido rige el equilibrio elastoestatico, la integral de contorno se anula en todas partes salvo en los nodos donde existen condiciones de contorno impuestas (desplazamientos prescritos o fuerzas aplicadas).

3.2. Implementacion nodal: evaluate_RE
--------------------------------------------------------------------------------
La funcion `extractDespForcefromNodes()` lee los vectores nodales de desplazamientos (U) y reacciones (RF / NFORC) en los nodos con condiciones de contorno de ambos ficheros ODB.

A continuacion, `evaluate_RE()` calcula discretamente:

  RE = 0.5 * sum_{i en BCs} [ Ux2(i)*Fx1(i) + Uy2(i)*Fy1(i) - Ux1(i)*Fx2(i) - Uy1(i)*Fy2(i) ]

Ventajas computacionales:
- No requiere integrar densidades de energia elastica en los miles o millones de elementos de la malla.
- El calculo es practicamente instantaneo (opera sobre una lista reducida de nodos de contorno).
- Proporciona una precision muy alta, dependiente unicamente de la resolucion global del modelo.

Sensibilidad numerica:
Para grietas muy cortas (l << lc), la respuesta con grieta difiere muy poco de la respuesta sin grieta (U2 prox U1 y F2 prox F1). El valor de RE resulta de la resta de magnitudes casi identicas. Por ello, es recomendable que los jobs de Abaqus se resuelvan con doble precision numerica.

================================================================================
4. ENERGIA DISIPADA Y FACTOR CRITICO t_EC
================================================================================

4.1. Energia disipada: evaluate_DE
--------------------------------------------------------------------------------
La funcion `evaluate_DE(crackObject, Gc1, ...)` calcula el coste energetico de apertura de la nueva superficie:

  DE = Gc1 * l_efectiva

Para verificar que la grieta no rebasa los limites fisicos del solido, la longitud efectiva `l_efectiva` se evalua a lo largo de un camino de inspeccion muestreado sobre el campo de resultados.

4.2. Factor critico de carga del Criterio Energetico
--------------------------------------------------------------------------------
Conocidos DE y RE, el factor critico resulta:

  t_EC = sqrt(DE / |RE|) = sqrt((Gc1 * l_efectiva) / |RE|)

================================================================================
Navegacion: [Anterior: Extraccion de Tensiones](02-extraccion-tensiones.md) | [Portada Wiki](../index.md) | [Siguiente: Modulo de Optimizacion](04-modulo-optimizacion.md)
================================================================================
