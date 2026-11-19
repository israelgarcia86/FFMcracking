###########################   CONFIGURATION   ################################
# Input files, path and mat properties
import os
import sys

inpFilePath = os.getcwd().replace('\\', '/') + '/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = inpFilePath + 'compFiles/'
sigmac = 70.0
Gc1 = 0.4
##############################################################################

##############################################################################
# Main code (not necessary to modify)
repository_root = os.path.abspath(os.path.join(inpFilePath, os.pardir, os.pardir))
if not os.path.isfile(os.path.join(repository_root, 'src', '__init__.py')):
    raise RuntimeError(
        'Set the Abaqus working directory to this example folder under '
        'the FFMcracking repository samples directory.'
    )
if sys.path[:1] != [repository_root]:
    sys.path.insert(0, repository_root)

from src import FFMcracking as FFM

# Generation of the objects for the materual and the model
material_1 = FFM.FFM_Material(sigmac = sigmac, tauc = 0.0, Gc1 = Gc1, Gc2 = 0.0)
model_1 = FFM.FFM_InputModel(inpFilePath,inpFileName,nameStep,outRoute,material_1)

# In this example, the test corresponds to 
# Crack geometry
coordinates_1 = [(1.0, 0.0, 0.0),(4.5, 0.0, 0.0)]
crack_1 = FFM.FFM_Crack(coordinates_1)
################################################################################

################################################################################
# Run the code
[critical_factor_CC, critical_factor_SC, critical_factor_EC] = FFM.Compute_crit_factor_FFM(crack_1,model_1)
print('The critical factor is: ' + str(critical_factor_CC))
################################################################################

