# -*- coding: utf-8 -*-
"""Module containing all the definitions necessary to precit a crack onset using FFM
Created on Fri Apr  8 09:59:37 2016
@author: israelgarcia86
HOW TO USE: Please see Readme in the same repository
"""

# Definitions of classes (to be moved to another file)
class FFM_Material:
    """Class for objects containng material information necessary for
        finire fracture mechanics analysis

    Attributes:
        sigmac: Critical tensile strenght of the material
        tauc:   Critical shear strength of the material
        Gc1:    Fracture toughness in pure mode 1
        Gc2:    Fracture toughness in pure mode 2
        maxsigma:   Maximum stress for the XFEM model
        maxd:       Maximum displacement for the XFEM model 
    """
    def __init__(self,sigmac,tauc,Gc1,Gc2):
        self.sigmac = sigmac    
        self.tauc = tauc
        self.Gc1 = Gc1
        self.Gc2 = Gc2
        self.maxsigma = 5000e10*sigmac
        self.maxd = 1.0e10*Gc1/sigmac

class FFM_Crack:
    """Class defining a crack object in the context of FFM

    Attributes:
        added_crack_geom: List for every segment of the coordinates defining the geometry 
                        of the segment for the new crack. In case of straight segments: Two coordinates per segment.
                        Coordinates are given as tuples (x,y)
        curr_crack_geom: List for every segment of the coordinates defining the geometry
                        of the segment for the current crack. In case of straight segments: Two coordinates per segment.
                        Coordinates are given as tuples (x,y)
        added_crack_type: List of strings defining the type of every segment for the new crack.
                        Options for now: 'straight', more to be added
        curr_crack_type: List of strings defining the type of every segment for the current crack.
                        Options for now: 'straight', more to be added
    """
    def __init__(self, added_crack_geom = [], 
                    curr_crack_geom = [], 
                    added_crack_type = ['straight']*len(added_crack_geom), 
                    curr_crack_type = ['straight']*len(curr_crack_geom)):
        self.added_crack_geom = added_crack_geom
        self.curr_crack_geom = curr_crack_geom 
        self.added_crack_type = added_crack_type
        self.curr_crack_type = curr_crack_type

    def consolidade_segment(self,segment_geom,segment_type):
        self.curr_crack_geom.append(segment_geom)
        self.curr_crack_type.append(segment_type)
        

class FFM_InputModel:
    """Class for objects containng the input model and the problem to solve

    Attributes:
        inpFilePath: String with the path to the directory containing the Abaqus imput file
        inpFileName: String with the name of the Abaqus imput file
        nameStep:    String with the name of the step for which the load are applied
        outRoute:    String with the path where the outputfiles want to be saved
        material_model:   Object containing the data about the material
    """
    def __init__(self,inpFilePath,inpFileName,nameStep,outRoute,material_model):
        self.inpFilePath = inpFilePath
        self.inpFileName = inpFileName
        self.nameStep = nameStep
        self.outRoute = outRoute
        self.material_model = material_model
    def extract_sigmac(self):
        return self.material_model.sigmac
    def extract_Gc(self):
        return self.material_model.Gc1
    def extract_maxsigma(self):
        return self.material_model.maxsigma
    def extract_maxd(self):
        return self.material_model.maxd

# Function to extract the nodes where force BCs are applied
def extractNodesForce(modelObject):
    listaNodesForce = []
    for cond in modelObject.loads.keys():
        reg = modelObject.loads[cond].region[0]

        if reg in modelObject.rootAssembly.surfaces:
            objectReg = modelObject.rootAssembly.surfaces[reg]
        elif reg in modelObject.rootAssembly.sets:
            objectReg = modelObject.rootAssembly.sets[reg]

        for n in objectReg.nodes:
                if n.label not in listaNodesForce:
                    listaNodesForce.append(n.label)
    return listaNodesForce

# Function to extract the nodes where disp BCs are applied
def extractNodesDesp(modelObject):
    listaNodesDesp = []
    for cond in modelObject.boundaryConditions.keys():
        reg = modelObject.boundaryConditions[cond].region[0]

        if reg in modelObject.rootAssembly.surfaces:
            objectReg = modelObject.rootAssembly.surfaces[reg]
        elif reg in modelObject.rootAssembly.sets:
            objectReg = modelObject.rootAssembly.sets[reg]

        for n in objectReg.nodes:
                if n.label not in listaNodesDesp:
                    listaNodesDesp.append(n.label)
    return listaNodesDesp

# Function to join two lists
def union(a,b):
    for e in b:
        if e not in a:
            a.append(e)
    return a

# Function to extract Displacement and Force from Nodes
def  extractDespForcefromNodes(resultsObject,listaNodesDesp,listaNodesForce,lista_nodes_BCs):
    import numpy as np
    Ux_BC = np.zeros(len(lista_nodes_BCs))
    Uy_BC = np.zeros(len(lista_nodes_BCs))
    Fx_BC = np.zeros(len(lista_nodes_BCs))
    Fy_BC = np.zeros(len(lista_nodes_BCs))

    # Exploring the results
    UList=resultsObject.fieldOutputs['U'].values
    RFList=resultsObject.fieldOutputs['RF'].values
    NF1List=resultsObject.fieldOutputs['NFORC1'].values
    NF2List=resultsObject.fieldOutputs['NFORC2'].values   

    for n in UList:
        if n.nodeLabel in listaNodesDesp or n.nodeLabel in listaNodesForce:
            i = lista_nodes_BCs.index(n.nodeLabel)
            Ux_BC[i] = n.dataDouble[0]
            Uy_BC[i] = n.dataDouble[1]

    for n in RFList:
        if n.nodeLabel in listaNodesDesp:
            i = lista_nodes_BCs.index(n.nodeLabel)
            Fx_BC[i] = n.dataDouble[0]
            Fy_BC[i] = n.dataDouble[1]

    for n in NF1List:
        if n.nodeLabel in listaNodesForce:
            i = lista_nodes_BCs.index(n.nodeLabel)
            Fx_BC[i] -= n.data
            
    for n in NF2List:
        if n.nodeLabel in listaNodesForce:
            i = lista_nodes_BCs.index(n.nodeLabel)
            Fy_BC[i] -= n.data

    return [Ux_BC,Uy_BC,Fx_BC,Fy_BC]

def extractStressesBetween2points(session,odbObject,point1,point2):
    import numpy as np
    from abaqusConstants import *
    
    # Number of intervals for the path where the stresses are extracted
    numItervals = 100

    auxX = np.linspace(point1[0],point2[0],num=numItervals)
    auxY = np.linspace(point1[1],point2[1],num=numItervals)
    auxZ = np.linspace(point1[2],point2[2],num=numItervals)

    ListPointsPathArray = np.transpose(np.array([auxX,auxY,auxZ]))
    ListPointsPath = tuple(map(tuple,ListPointsPathArray))

    # CONFIGURE THE SESSION
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON)
    session.viewports['Viewport: 1'].setValues(displayedObject=None)

    # EXPLORING THE RESULTS
    session.viewports['Viewport: 1'].setValues(displayedObject=odbObject)
    session.Path(name='Path-1', type=POINT_LIST, expression=ListPointsPath)

    # Generating the path
    pth = session.paths['Path-1']

    # Modifying the coordinate system
    # Tangent vector
    aux = (np.array(point2) - np.array(point1))/np.linalg.norm((np.array(point2) - np.array(point1)))
    t_vector = np.array([aux[0],aux[1]])
    # Normal vector
    n_vector = np.array([-t_vector[1],t_vector[0]])

    scratchOdb = session.ScratchOdb(odbObject)
    scratchOdb.rootAssembly.DatumCsysByThreePoints(name='CSYS-1',
        coordSysType=CARTESIAN, origin=(0.0, 0.0, 0.0), point1=(aux[0],aux[1], 0.0),
        point2=(-t_vector[1],t_vector[0], 0.0))
    #dtm = session.scratchOdbs[outRoute+jobNameEC+'-without.odb'].rootAssembly.datumCsyses['CSYS-1']
    dtm = scratchOdb.rootAssembly.datumCsyses['CSYS-1']
    session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(transformationType=USER_SPECIFIED, datumCsys=dtm)

    # Extracting the results for S11, S22 and S12
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S11'))
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

    return [stt, snn, stn]

def evaluate_stress_criterion(stt,snn,stn,ModelObject):
    import numpy as np
    sigmac = ModelObject.extract_sigmac()
    if np.amax(snn) > 0:
        crit_mult_SC = sigmac/np.amin(snn)
    else:
        crit_mult_SC = float('inf')
    
    return crit_mult_SC

def generate_crack(crackObject,modelObject,session,regionToolset):
    import math
    import numpy as np
    from abaqusConstants import *
    ###############
    # Generating the geometry for the crack
    ###############
    # Initial point por the crack
    point1 = crackObject.tips_coordinates[0]
    # Final point por the crack
    point2 = crackObject.tips_coordinates[1]
    approx_size = math.sqrt(np.sum((np.array(point2) - np.array(point1))**2))
    s = modelObject.ConstrainedSketch(name='__profile__', sheetSize=approx_size)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=point1[0:2], point2=point2[0:2])
    p = modelObject.Part(name='Crack', dimensionality=TWO_D_PLANAR,type=DEFORMABLE_BODY)
    p = modelObject.parts['Crack']
    p.BaseWire(sketch=s)
    s.unsetPrimaryObject()
    p = modelObject.parts['Crack']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del modelObject.sketches['__profile__']

    ################
    # Generating the crack
    ################
    # Obtaining the instances before generating the crack
    listInstances = []
    for nameinst in modelObject.rootAssembly.instances.keys():
        listInstances.append(nameinst)

    # Generaing the crack instance
    a = modelObject.rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a = modelObject.rootAssembly
    p = modelObject.parts['Crack']
    a.Instance(name='Crack-Instance', part=p, dependent=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, constraints=ON, connectors=ON, engineeringFeatures=ON)

    # Defining the Cracked Domain
    a = modelObject.rootAssembly
    elements1 = []
    for nameinst in listInstances:
        e1 = a.instances[nameinst].elements
        elements1 += e1

    crackDomain = regionToolset.Region(elements=e1)

    # Defining the Crack Location
    a = modelObject.rootAssembly
    edges1 = a.instances['Crack-Instance'].edges
    crackLocation = regionToolset.Region(edges=edges1)

    # Implementing the crack
    a = modelObject.rootAssembly
    a.engineeringFeatures.XFEMCrack(name='Crack-1', crackDomain=crackDomain, crackLocation=crackLocation)

    # Defining the interaction
    modelObject.XFEMCrackGrowth(name='Int-1', createStepName='Initial', crackName='Crack-1')

    # Visualazing the crack
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF,  constraints=OFF, connectors=OFF, engineeringFeatures=OFF)

def evaluate_RE(Ux1_BC,Ux2_BC,Uy1_BC,Uy2_BC,Fx1_BC,Fx2_BC,Fy1_BC,Fy2_BC):
    import numpy as np
    RE = 0.5*np.sum(Ux2_BC*Fx1_BC + Uy2_BC*Fy1_BC - Ux1_BC*Fx2_BC - Uy1_BC*Fy2_BC)
    return RE

def evaluate_DE(crackObject,OriginalModel,odbECwith,session):
    import numpy as np
    from abaqusConstants import *
    # EXPLORING THE RESULTS
    # To calculate the length when intersecting the crack edge and one part is out
    # Initial point por the crack
    point1 = crackObject.tips_coordinates[0]
    # Final point por the crack
    point2 = crackObject.tips_coordinates[1]
    # Number of intervals for the path where the stresses are extracted
    numItervals = 100

    auxX = np.linspace(point1[0],point2[0],num=numItervals)
    auxY = np.linspace(point1[1],point2[1],num=numItervals)
    auxZ = np.linspace(point1[2],point2[2],num=numItervals)

    ListPointsPathArray = np.transpose(np.array([auxX,auxY,auxZ]))
    ListPointsPath = tuple(map(tuple,ListPointsPathArray))

    session.viewports['Viewport: 1'].setValues(displayedObject=odbECwith)
    session.Path(name='Path-1', type=POINT_LIST, expression=ListPointsPath)
    
    # Generating the path
    pth = session.paths['Path-1']

    # Extracting the results for S11, S22 and S12
    # To calculate the length when intersecting the crack edge and one part is out
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S11'))
    try:
        dataS11 = session.XYDataFromPath(name='XYData-1', path=pth, includeIntersections=False, shape=UNDEFORMED, labelType=TRUE_DISTANCE)
        # To understand next line
        cracklength = np.array(dataS11.data)[-1,0] - np.array(dataS11.data)[0,0]
        del session.xyDataObjects['XYData-1']
    except VisError:
        cracklength=float('inf')

    # Calculating the dissipated energy
    Gc = OriginalModel.extract_Gc()
    
    DE = Gc*cracklength

    return DE

def extractStresses_crackpath(session,odbObject,CrackFFM):
    # For a straight crack (can be generalized for others)
    # Initial point for the crack
    point1 = CrackFFM.tips_coordinates[0]
    # Final point por the crack
    point2 = CrackFFM.tips_coordinates[1]

    [stt, snn, stn] = extractStressesBetween2points(session,odbObject,point1,point2)

    return [stt, snn, stn]

def remove_files(directory,word):
    import os
    for f in os.listdir(directory):
        if word in f:
            os.remove(directory + f)


def Compute_crit_factor_FFM(CrackFFM,OriginalModel):
    """
    Compute_crit_factor_FFM computes the critical factor to the current loads in the model which would lead
                            to crack initiation for a certain crack geometry detailed in CrackFFM object

    :object CrackFFM of class FFM_Crack. This object details the data of the crack to be tested.
    :object OriginalModel of class FFM_InputModel: This object included the data about the original input model
                                                    and the material properties
    
    :return: a float containing the critical factor to the current loads for which a crack onset is predicted
                according to the Coupled Criterion of the Finite Fracture Mechanics
                see D. Leguillon, Europ. Journal of Mech  A/Solids, Vol 21, Issue 1, Page 61-72. 
                https://doi.org/10.1016/S0997-7538(01)01184-6
    """ 
    from abaqus import *
    from abaqusConstants import *
    from odbAccess import *
    from caeModules import *
    import numpy as np
    import os
    import math

    # Close previous mdb opened - preventive 
    mdb.close()

    # IMPORTING THE MODEL
    # Changing the working directory
    os.chdir(OriginalModel.inpFilePath)

    # Total path to the original input file
    inpFileTotal = OriginalModel.inpFilePath+OriginalModel.inpFileName

    # Name given to the original model
    modelName = OriginalModel.inpFileName[:-4]+'-Original'

    # Name given to the original model for the EC
    jobNameEC = OriginalModel.inpFileName[:-4]+'-EC'

    # Name given to the original model for the step with the loads
    nameStep = OriginalModel.nameStep

    # Route to keep the files of the simulations
    outRoute = OriginalModel.outRoute

    # Generate the output directory in case it does not exist
    if not os.path.isdir(outRoute):
        os.mkdir(outRoute[:-1])

    # Defining work directory
    os.chdir(outRoute)

    # CONFIGURE THE SESSION
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=ON)
    Mdb()
    session.viewports['Viewport: 1'].setValues(displayedObject=None)

    if jobNameEC+'-without.odb' not in os.listdir(outRoute):

        # IMPORTING THE MODEL
        mdb.ModelFromInputFile(name=modelName, inputFileName=inpFileTotal)
        a = mdb.models[modelName].rootAssembly

        # Including the field output U and NFORC to extract after the results
        mdb.models[modelName].FieldOutputRequest(name='EC0', createStepName=nameStep, variables=('U', 'TF', 'NFORC','RF','CDISP','CSTRESS','CF','PHILSM'))

        # Generate a job for the case without crack
        mdb.Job(name=jobNameEC+'-without', model=modelName, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=DOUBLE, nodalOutputPrecision=FULL, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',     scratch='', multiprocessingMode=DEFAULT, numCpus=1)

        # SUBMITTING THE ANALYSIS
        # Submit the job for the Stress Criterion
        mdb.jobs[jobNameEC+'-without'].submit(consistencyChecking=OFF)

        # Waiting for the job to be completed
        mdb.jobs[jobNameEC+'-without'].waitForCompletion()

    # Importing the results
    odbECwithout = openOdb(outRoute+jobNameEC+'-without.odb')

    [stt, snn, stn] = extractStresses_crackpath(session,odbECwithout,CrackFFM)

    ## Calculating the multiplier for the initial loads
    crit_mult_SC = evaluate_stress_criterion(stt,snn,stn,OriginalModel)

    ### EVALUATING THE ENERGY CRITERION

    # IMPORTING THE MODEL
    mdb.ModelFromInputFile(name=modelName, inputFileName=inpFileTotal)
    a = mdb.models[modelName].rootAssembly

    # Generate list of nodes with force control
    listaNodesForce = extractNodesForce(mdb.models[modelName])
    
    # Generate list of nodes with disp control
    listaNodesDesp = extractNodesDesp(mdb.models[modelName])
        
    # Joining the two lists because the formula to obtain the energy can be applied directly during the EC evaluation
    lista_nodes_BCs = union(listaNodesForce[:],listaNodesDesp[:])

     # Extracting the results of forces and displacement before the crack onset
    resultsObjectwithout = odbECwithout.steps[nameStep].frames[-1]
    [Ux1_BC,Uy1_BC,Fx1_BC,Fy1_BC] = extractDespForcefromNodes(resultsObjectwithout,listaNodesDesp,listaNodesForce,lista_nodes_BCs)

    # Data for the XFEM model
    maxsigma = OriginalModel.extract_maxsigma()
    maxd = OriginalModel.extract_maxd()

    # INCLUIR GRIETA XFEM
    # Introducing damage model in material
    for mat in mdb.models[modelName].materials.keys():
        mdb.models[modelName].materials[mat].MaxpsDamageInitiation(table=((maxsigma, ), ))
        mdb.models[modelName].materials[mat].maxpsDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((maxd, ), ))

    generate_crack(CrackFFM,mdb.models[modelName],session,regionToolset)

    # INCLUDING THE FIELD OUTPUT NFORCE AND U EN EL ANALISIS PARA TENERLOS DESPUES
    mdb.models[modelName].FieldOutputRequest(name='EC0', createStepName=nameStep, variables=('U', 'TF', 'NFORC','RF','CDISP','CSTRESS','CF','PHILSM'))

    # Delete all the files with the word 'with.' from a previous analysis to avoid conflicts
    remove_files(outRoute,'with.')

    # Generatin the job
    mdb.Job(name=jobNameEC+'-with', model=modelName, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', multiprocessingMode=DEFAULT, numCpus=1)

    # SUbmitting the job
    mdb.jobs[jobNameEC+'-with'].submit(consistencyChecking=OFF)

    # WAITING FOR FINALIZATION
    # Waiting for the job to be completed
    mdb.jobs[jobNameEC+'-with'].waitForCompletion()

    # RECUPERAR NFORCE AND U EN EL ANALISIS
    # Importing the results
    odbECwith = openOdb(outRoute+jobNameEC+'-with.odb')

    resultsObjectwith = odbECwith.steps[nameStep].frames[-1]
    [Ux2_BC,Uy2_BC,Fx2_BC,Fy2_BC] = extractDespForcefromNodes(resultsObjectwith,listaNodesDesp,listaNodesForce,lista_nodes_BCs)

    # CALCULAR EL CAMBIO DE ENERGIA A PARTIR DEL CAMBIO
    # Change in potential elastic energy
    RE = evaluate_RE(Ux1_BC,Ux2_BC,Uy1_BC,Uy2_BC,Fx1_BC,Fx2_BC,Fy1_BC,Fy2_BC)

    # Dissipated energy
    DE = evaluate_DE(CrackFFM,OriginalModel,odbECwith,session)

    # Evaluating the energy criterion
    # Since we know that if the boundary conditions are multiplied by t the released energy is also multiplied by t we can obtain the critical t
    if RE==0:
        crit_mult_EC = np.inf
    else:
        crit_mult_EC = math.sqrt(math.fabs(DE/RE))

    ## EVALUATE THE COUPLED CRITERION
    crit_mult_CC = max(crit_mult_SC, crit_mult_EC)

    # Closing the result files and the model database
    odbECwith.close()
    odbECwithout.close()
    mdb.close()

    ## RETURNING THE CRITICAL FACTOR ACCORDING TO THE COUPLED CRITERION
    return [crit_mult_CC, crit_mult_SC, crit_mult_EC]