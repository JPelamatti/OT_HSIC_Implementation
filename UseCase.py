import numpy as np
import openturns as ot

##########################################################
#Example: Conditional SA, V-stat estimator, exponential weight function, permutation p-value estimation
loaded_sample = ot.Sample.ImportFromCSVFile('sample.csv')
d = 5
input_sample = loaded_sample[:,0:d]
output_sample = loaded_sample[:,d]


x_covariance_collection = []
for i in range(d): 
    cov = ot.SquaredExponential()
    cov.setScale(1/np.var(input_sample[:,i])) #Gaussian kernel parameterization
    x_covariance_collection.append(cov)
    
y_covariance = ot.SquaredExponential()
y_covariance.setScale(1/np.var(output_sample))  #Gaussian kernel parameterization

CovarianceList = [x_covariance_collection,y_covariance]

C = [[-3.,-1.6],[5.3,9.9]]
weightFunctionParameters = [5.]
weightFunction = ot.HSICSAExponentialWeightFunction(C, weightFunctionParameters)

Estimator = ot.CSAHSICEstimator(CovarianceList, input_sample, output_sample, ot.HSICEstimator.Vstat, weightFunction)

Estimator.computeIndices()
Estimator.drawR2HSICIIndices()

B = 1200
Estimator.parameterizePValueEstimator(ot.HSICEstimator.PermutationPValuesEstimator, B)
Estimator.computePValues()
Estimator.drawPValues()



#############################################################
#Example: Global SA, U-stat estimator, asymptotic p-value estimation

loaded_sample = ot.Sample.ImportFromCSVFile('sample.csv')
d = 5
input_sample = loaded_sample[:,0:d]
output_sample = loaded_sample[:,d]


x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(1/np.var(input_sample[:,i])) #Gaussian kernel parameterization
    x_covariance_collection.append(cov)
    
y_covariance = ot.SquaredExponential()
y_covariance.setScale(1/np.var(output_sample))  #Gaussian kernel parameterization

CovarianceList = [x_covariance_collection,y_covariance]

Estimator = ot.GSAHSICEstimator(CovarianceList, input_sample, output_sample, ot.HSICEstimator.Ustat)

Estimator.computeIndices()
Estimator.drawR2HSICIIndices()

Estimator.parameterizePValueEstimator(ot.HSICEstimator.AsymptoticPValuesEstimator)
Estimator.computePValues()
Estimator.drawPValues()



################################################################
#Example: Target SA, V-stat estimator, step weight function, asymptotic p-value estimation


loaded_sample = ot.Sample.ImportFromCSVFile('sample.csv')
d = 5
input_sample = loaded_sample[:,0:d]
output_sample = loaded_sample[:,d]


x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(1/np.var(input_sample[:,i]))  #Gaussian kernel parameterization
    x_covariance_collection.append(cov)
    
    
#Definition of kronecker covariance object
def kroncov(s, t):
    if s == t:
        return 1.
    else:
        return 0.

myMesh = ot.IntervalMesher([2]).build(ot.Interval(0., 1.))
myCovariance = ot.CovarianceMatrix(myMesh.getVerticesNumber())
for k in range(myMesh.getVerticesNumber()):
    t = myMesh.getVertices()[k]
    for l in range(k + 1):
        s = myMesh.getVertices()[l]
        myCovariance[k, l] = kroncov(s[0], t[0])

y_covariance = ot.UserDefinedCovarianceModel(myMesh, myCovariance)
y_covariance.setActiveParameter([])

y_covariance.setScale(1/np.var(output_sample))

CovarianceList = [x_covariance_collection,y_covariance]

C = [[-3.,-1.6],[5.3,9.9]]
weightFunction = ot.HSICSAStepWeightFunction(C)

Estimator = ot.TSAHSICEstimator(CovarianceList, input_sample, output_sample, ot.HSICEstimator.Vstat, weightFunction)

Estimator.computeIndices()
Estimator.drawR2HSICIIndices()

Estimator.parameterizePValueEstimator(ot.HSICEstimator.AsymptoticPValuesEstimator)
Estimator.computePValues()
Estimator.drawPValues()