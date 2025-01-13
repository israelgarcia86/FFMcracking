# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:59:37 2016
@author: israel
HOW TO USE:

"""

##################################################   CONFIGURATION   ##########
# Path and name to the input file
inpFilePath='/home/israel/OHT_UD/'
inpFileName='OHT_L185W56d14_UD.inp'
nameStep='Step-1'
# Path where the output files are saved
#outRuta='/usr/simulia/Commands/'
outRuta=inpFilePath

# Material failure properties
sigmac=10.0   # Tensile strength expressed in the same units that those used in input file
Gc=5.0        # Fracture toughness expressed in the same units that those used in input file
###

###
numItervals = 100
###

# Damage parameters for the damage model (they should be very large to avoid crack growth)
maxsigma = 5000e10
maxd = 10e10

#####

#### Crack geometries to be taken into account
# Generate List of cracks to test
CracksList=[]
import numpy as np
for L in np.arange(0.3,26.0,0.1):
  a=[(-7.0-L, 2.5, 0.0),(7.0+L, 2.5, 0.0)]
  CracksList+=[a]

lenth_nomat = 14.0      # To remove from the estimation of the crack length the space without material (inside a hole for example)
################################################# END OF CONFIGURATION ########
#
#
# IMPORT MODULES
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from caeModules import *
import numpy as np
import os
import math

mdb.close()

### CONFIGURATION TO TEST
# For a straight crack

# IMPORT THE MODEL
# Changing the working directory
os.chdir(inpFilePath)

# Total path to the original input file
inpFileTotal=inpFilePath+inpFileName

# Name given to the original model
modelName=inpFileName[:-4]+'-Original'

# Name given to the original model for the EC
jobNameEC=inpFileName[:-4]+'-EC'

# CONFIGURE THE SESSION
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON)
Mdb()
session.viewports['Viewport: 1'].setValues(displayedObject=None)

# IMPORTING THE MODEL
mdb.ModelFromInputFile(name=modelName, inputFileName=inpFileTotal)
a = mdb.models[modelName].rootAssembly


# EXPLORING THE NODES WHERE A FORCE OR DISPLACEMENT IS PRESCRIBED, LIST OF NODES
####
####
# Forces
listaNodesForce=[]
for cond in mdb.models[modelName].loads.keys():
    reg = mdb.models[modelName].loads[cond].region[0]

    if reg in mdb.models[modelName].rootAssembly.surfaces:
        objectReg = mdb.models[modelName].rootAssembly.surfaces[reg]
    elif reg in mdb.models[modelName].rootAssembly.sets:
        objectReg = mdb.models[modelName].rootAssembly.sets[reg]

    for n in objectReg.nodes:
            if n.label not in listaNodesForce:
                listaNodesForce.append(n.label)


# Displacement
listaNodesDesp=[]
for cond in mdb.models[modelName].boundaryConditions.keys():
    reg = mdb.models[modelName].boundaryConditions[cond].region[0]

    if reg in mdb.models[modelName].rootAssembly.surfaces:
        objectReg = mdb.models[modelName].rootAssembly.surfaces[reg]
    elif reg in mdb.models[modelName].rootAssembly.sets:
        objectReg = mdb.models[modelName].rootAssembly.sets[reg]

    for n in objectReg.nodes:
            if n.label not in listaNodesDesp:
                listaNodesDesp.append(n.label)

# Function to join two lists
def union(a,b):
    for e in b:
        if e not in a:
            a.append(e)
    return a

# Joining the two lists because the formula to obtain the energy can be applied directly
lista_nodes_BCs = union(listaNodesForce[:],listaNodesDesp[:])


# INCLUDING THE FIELD OUTPUT NFORCE AND U EN EL ANALISIS PARA TENERLOS DESPUES
mdb.models[modelName].FieldOutputRequest(name='EC0', createStepName=nameStep, variables=('U', 'TF', 'NFORC','RF','CDISP','CSTRESS','CF','PHILSM'))

# CREAR UN JOB QUE SE CORRESPONDE AL CASO SIN GRIETA
mdb.Job(name=jobNameEC+'-without', model=modelName, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=DOUBLE, nodalOutputPrecision=FULL, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',     scratch='', multiprocessingMode=DEFAULT, numCpus=1)

# HACER ANALISIS
# Submit the job for the EC
mdb.jobs[jobNameEC+'-without'].submit(consistencyChecking=OFF)

# Waiting for the job to be completed
mdb.jobs[jobNameEC+'-without'].waitForCompletion()

# Importing the results
odbECwithout = openOdb(outRuta+jobNameEC+'-without.odb')

# RECUPERAR NFORCE Y U EN ESOS NODOS
# Generating empty arrays
Ux1_BC = np.zeros(len(lista_nodes_BCs))
Uy1_BC = np.zeros(len(lista_nodes_BCs))
Fx1_BC = np.zeros(len(lista_nodes_BCs))
Fy1_BC = np.zeros(len(lista_nodes_BCs))

# Exploring the results
UList=odbECwithout.steps[nameStep].frames[-1].fieldOutputs['U'].values
RFList=odbECwithout.steps[nameStep].frames[-1].fieldOutputs['RF'].values
#    NF1List=odbECwithout.steps[nameStep].frames[-1].fieldOutputs['NFORC1'].values
#    NF2List=odbECwithout.steps[nameStep].frames[-1].fieldOutputs['NFORC2'].values

for n in UList:
    if n.nodeLabel in listaNodesDesp or n.nodeLabel in listaNodesForce:
        i = lista_nodes_BCs.index(n.nodeLabel)
        Ux1_BC[i] = n.dataDouble[0]
        Uy1_BC[i] = n.dataDouble[1]

for n in RFList:
    if n.nodeLabel in listaNodesDesp:
        i = lista_nodes_BCs.index(n.nodeLabel)
        Fx1_BC[i] = n.dataDouble[0]
        Fy1_BC[i] = n.dataDouble[1]

#    for n in NF1List:
#        if n.nodeLabel in listaNodesForce:
#            i = lista_nodes_BCs.index(n.nodeLabel)
#            Fx1_BC[i] += n.data
#
#    for n in NF2List:
#        if n.nodeLabel in listaNodesForce:
#            i = lista_nodes_BCs.index(n.nodeLabel)
#            Fy1_BC[i] += n.data

    # REMOVE JOB

# For a straight crack
# Initial point for the crack
point1 = crack[0]
# Final point por the crack
point2 = crack[1]

auxX = np.linspace(point1[0],point2[0],num=numItervals)
auxY = np.linspace(point1[1],point2[1],num=numItervals)
auxZ = np.linspace(point1[2],point2[2],num=numItervals)

ListPointsPathArray = np.transpose(np.array([auxX,auxY,auxZ]))
ListPointsPath = tuple(map(tuple,ListPointsPathArray))

############################## EVALUATING THE STRESS CRITERION
odbSC = odbECwithout

# CONFIGURE THE SESSION
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON)
Mdb()
session.viewports['Viewport: 1'].setValues(displayedObject=None)

# EXPLORING THE RESULTS
session.viewports['Viewport: 1'].setValues(displayedObject=odbSC)
session.Path(name='Path-1', type=POINT_LIST, expression=ListPointsPath)

# Generating the path
pth = session.paths['Path-1']

# Modifying the coordinate system
# Tangent vector
aux = (np.array(point2) - np.array(point1))/np.linalg.norm((np.array(point2) - np.array(point1)))
t_vector = np.array([aux[0],aux[1]])

# Normal vector
n_vector = np.array([-t_vector[1],t_vector[0]])

scratchOdb = session.ScratchOdb(odbSC)
scratchOdb.rootAssembly.DatumCsysByThreePoints(name='CSYS-1',
    coordSysType=CARTESIAN, origin=(0.0, 0.0, 0.0), point1=(aux[0],aux[1], 0.0),
    point2=(-t_vector[1],t_vector[0], 0.0))
dtm = session.scratchOdbs[outRuta+jobNameEC+'-without.odb'].rootAssembly.datumCsyses['CSYS-1']
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    transformationType=USER_SPECIFIED, datumCsys=dtm)

# Extracting the results for S11, S22 and S12
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S11'))
try:
    dataS11 = session.XYDataFromPath(name='XYData-1', path=pth, includeIntersections=False, shape=UNDEFORMED, labelType=TRUE_DISTANCE)
    #stt = np.array(dataS11.data)[:numItervals,1]      # Chapuza para sortear el erro de Abaqus 11
    stt = np.array(dataS11.data)[:,1]      # Codigo Correcto.
    del session.xyDataObjects['XYData-1']
except VisError:
    stt = np.array([[0.0,0.0],[0.0,0.0]])

session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S22'))
try:
    dataS22 = session.XYDataFromPath(name='XYData-1', path=pth, includeIntersections=False, shape=UNDEFORMED, labelType=TRUE_DISTANCE)
    #snn = np.array(dataS22.data)[:min(numItervals,np.size(sxx)),1]    # Chapuza para sortear el error de Abaqus 11
    snn = np.array(dataS22.data)[:,1]    # Codigo Correcto.
    del session.xyDataObjects['XYData-1']
except VisError:
    dataS22 = np.array([[0.0,0.0],[0.0,0.0]])
    snn = np.array([[0.0,0.0],[0.0,0.0]])


session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S12'))
try:
    dataS12 = session.XYDataFromPath(name='XYData-1', path=pth, includeIntersections=False, shape=UNDEFORMED, labelType=TRUE_DISTANCE)
    #stn = np.array(dataS12.data)[:min(numItervals,np.size(sxx)),1]    # Chapuza para sortear el error de Abaqus 11
    stn = np.array(dataS12.data)[:,1]   # Codigo Correcto.
    del session.xyDataObjects['XYData-1']
except VisError:
    stn = np.array([[0.0,0.0],[0.0,0.0]])

## Calculating the multiplier for the initial loads
if np.amax(snn) > 0:
    crit_mult_SC = sigmac/np.amin(snn)
else:
    crit_mult_SC = float('inf')
    #
    #
critical_mult_array_SC[n_test]=crit_mult_SC

############################# EVALUATING THE ENERGY CRITERION

# IMPORTING THE MODEL
mdb.ModelFromInputFile(name=modelName, inputFileName=inpFileTotal)
a = mdb.models[modelName].rootAssembly

# INCLUIR GRIETA XFEM
# Introducing damage model in material
for mat in mdb.models[modelName].materials.keys():
    mdb.models[modelName].materials[mat].MaxpsDamageInitiation(table=((maxsigma, ), ))
    mdb.models[modelName].materials[mat].maxpsDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((maxd, ), ))


# Generating the geometry for the crack
###############
approx_size = math.sqrt(np.sum((np.array(point2) - np.array(point1))**2))
s = mdb.models[modelName].ConstrainedSketch(name='__profile__', sheetSize=approx_size)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=point1[0:2], point2=point2[0:2])
#s.VerticalConstraint(entity=g.findAt((0.0, 7.5)), addUndoState=False)
p = mdb.models[modelName].Part(name='Crack', dimensionality=TWO_D_PLANAR,type=DEFORMABLE_BODY)
p = mdb.models[modelName].parts['Crack']
p.BaseWire(sketch=s)
s.unsetPrimaryObject()
p = mdb.models[modelName].parts['Crack']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models[modelName].sketches['__profile__']

################
# Generating the crack
################
# Obtaining the instances before generating the crack
listInstances = []
for nameinst in mdb.models[modelName].rootAssembly.instances.keys():
    listInstances.append(nameinst)

# Generaing the crack instance
a = mdb.models[modelName].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models[modelName].rootAssembly
p = mdb.models[modelName].parts['Crack']
a.Instance(name='Crack-Instance', part=p, dependent=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, constraints=ON, connectors=ON, engineeringFeatures=ON)

# Defining the Cracked Domain
a = mdb.models[modelName].rootAssembly
elements1=[]
for nameinst in listInstances:
    e1 = a.instances[nameinst].elements
    elements1 += e1

crackDomain = regionToolset.Region(elements=e1)

# Defining the Crack Location
a = mdb.models[modelName].rootAssembly
#e1 = a.instances['Part-2-1'].edges
#edges1 = e1.findAt(((0.0, 3.75, 0.0), ))
edges1 = a.instances['Crack-Instance'].edges
crackLocation = regionToolset.Region(edges=edges1)

# Implementing the crack
a = mdb.models[modelName].rootAssembly
a.engineeringFeatures.XFEMCrack(name='Crack-1', crackDomain=crackDomain, crackLocation=crackLocation)

# Defining the interaction
mdb.models[modelName].XFEMCrackGrowth(name='Int-1', createStepName='Initial', crackName='Crack-1')
#: The interaction "Int-1" has been created.

# Visualazing the crack
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF,  constraints=OFF, connectors=OFF, engineeringFeatures=OFF)
###############

# INCLUDING THE FIELD OUTPUT NFORCE AND U EN EL ANALISIS PARA TENERLOS DESPUES
mdb.models[modelName].FieldOutputRequest(name='EC0', createStepName=nameStep, variables=('U', 'TF', 'NFORC','RF','CDISP','CSTRESS','CF','PHILSM'))

# Generatin the job
#################
mdb.Job(name=jobNameEC+'-with', model=modelName, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', multiprocessingMode=DEFAULT, numCpus=1)
################

# SUbmitting the job
mdb.jobs[jobNameEC+'-with'].submit(consistencyChecking=OFF)

# WAITING FOR FINALIZATION
# Waiting for the job to be completed
mdb.jobs[jobNameEC+'-with'].waitForCompletion()

# RECUPERAR NFORCE AND U EN EL ANALISIS
# Importing the results
odbECwith = openOdb(outRuta+jobNameEC+'-with.odb')

# RECUPERAR NFORCE Y U EN ESOS NODOS
# Generating empty arrays
Ux2_BC = np.zeros(len(lista_nodes_BCs))
Uy2_BC = np.zeros(len(lista_nodes_BCs))
Fx2_BC = np.zeros(len(lista_nodes_BCs))
Fy2_BC = np.zeros(len(lista_nodes_BCs))

# Exploring the results
UList=odbECwith.steps[nameStep].frames[-1].fieldOutputs['U'].values
RFList=odbECwith.steps[nameStep].frames[-1].fieldOutputs['RF'].values
#    NF1List=odbECwith.steps[nameStep].frames[-1].fieldOutputs['NFORC1'].values
#    NF2List=odbECwith.steps[nameStep].frames[-1].fieldOutputs['NFORC2'].values

for n in UList:
    if n.nodeLabel in listaNodesDesp or n.nodeLabel in listaNodesForce:
        i = lista_nodes_BCs.index(n.nodeLabel)
        Ux2_BC[i] = n.dataDouble[0]
        Uy2_BC[i] = n.dataDouble[1]

for n in RFList:
    if n.nodeLabel in listaNodesDesp:
        i = lista_nodes_BCs.index(n.nodeLabel)
        Fx2_BC[i] = n.dataDouble[0]
        Fy2_BC[i] = n.dataDouble[1]

#    for n in NF1List:
#        if n.nodeLabel in listaNodesForce:
#            i = lista_nodes_BCs.index(n.nodeLabel)
#            Fx2_BC[i] += n.data
#
#    for n in NF2List:
#        if n.nodeLabel in listaNodesForce:
#            i = lista_nodes_BCs.index(n.nodeLabel)
#            Fy2_BC[i] += n.data


# CALCULAR EL CAMBIO DE ENERGIA A PARTIR DEL CAMBIO
# Released energy
RE = 0.5*np.sum(Ux2_BC*Fx1_BC + Uy2_BC*Fy1_BC - Ux1_BC*Fx2_BC - Uy1_BC*Fy2_BC)

# Dissipated energy
# EXPLORING THE RESULTS
# To calculate the length when intersecting the crack edge and one part is out
session.viewports['Viewport: 1'].setValues(displayedObject=odbECwith)
session.Path(name='Path-1', type=POINT_LIST, expression=ListPointsPath)

# Generating the path
pth = session.paths['Path-1']

# Extracting the results for S11, S22 and S12
# To calculate the length when intersecting the crack edge and one part is out
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S11'))
try:
    dataS11 = session.XYDataFromPath(name='XYData-1', path=pth, includeIntersections=False, shape=UNDEFORMED, labelType=TRUE_DISTANCE)
    cracklength = np.array(dataS11.data)[-1,0] - np.array(dataS11.data)[0,0]
    del session.xyDataObjects['XYData-1']
except VisError:
    cracklength=float('inf')

# Calculating the dissipated energy
DE = Gc*(cracklength-lenth_nomat)

# Since we know that if the boundary conditions are multiplied by t the released energy is also multiplied by t we can obtain the critical t
if RE==0:
    crit_mult_EC = np.inf
else:
    crit_mult_EC = math.sqrt(math.fabs(DE/RE))

crit_mult_CC = max(crit_mult_SC, crit_mult_EC)

print crit_mult_CC


odbECwith.close()
odbECwithout.close()
mdb.close()
