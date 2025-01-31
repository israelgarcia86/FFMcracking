###########################   CONFIGURATION   ################################
# Input files, path and mat properties
inpFilePath = '/home/israel/Calculos/abaqus/prueba6/'
inpFileName = 'OpenHoleLongDEF-fmesh-00.inp'
nameStep = 'Loading'
outRoute = '/home/israel/Calculos/abaqus/prueba6/compFiles/'
sigmac = 70.0
Gc1 = 0.4
L_min = 1.0e-1
L_max = 5.0
##############################################################################

##############################################################################
# Main code (not necessary to modify)
import FFMcracking as FFM
import modOptimization as modOpt

# Generation of the objects for the materual and the model
material_1 = FFM.FFM_Material(sigmac = sigmac, tauc = 0.0, Gc1 = Gc1, Gc2 = 0.0)
model_1 = FFM.FFM_InputModel(inpFilePath,inpFileName,nameStep,outRoute,material_1)
################################################################################

################################################################################
# Run the code
def objective(L):
    coordinates_1 = [(1.0, 0.0, 0.0),(1.0+L, 0.0, 0.0)]
    crack_1 = FFM.FFM_Crack(coordinates_1)
    [critical_factor_CC, critical_factor_SC, critical_factor_EC] = FFM.Compute_crit_factor_FFM(crack_1,model_1)
    print('Evaluated for L = ' + str(L) + 'and got crit factor = ' + str(critical_factor_CC))
    return [critical_factor_CC, critical_factor_SC, critical_factor_EC]

bounds = (L_min, L_max)
n_iterations = 10
best_x, best_y, x_train, y_train = modOpt.bayesian_optimization(objective, bounds, n_iterations)
print('The critical factor is: ' + str(best_y) + ' for a increment of crack length of ' +  str(best_x))
print('Training data')
print('xtrain: ', str(x_train))
print('ytrain: ', str(y_train))
################################################################################
