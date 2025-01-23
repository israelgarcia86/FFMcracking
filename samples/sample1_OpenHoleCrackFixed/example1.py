###########################   CONFIGURATION   ################################
# Input files, path and mat properties
inpFilePath = '/home/israel/Calculos/abaqus/prueba5/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = '/home/israel/Calculos/abaqus/prueba5/compFiles/'
sigmac = 70.0
Gc1 = 0.4
##############################################################################

##############################################################################
# Main code (not necessary to modify)
import FFMcracking as FFM

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
critical_factor = FFM.Compute_crit_factor_FFM(crack_1,model_1)
print('The critical factor is: ' + str(critical_factor))
################################################################################
