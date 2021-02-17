import numpy as np
import openturns as ot
import HSICEstimators
import matplotlib.pyplot as plt
import HSICSAWeightFunctions

plt.close('all')

ot_HSICEstimator_Vstat = 1
ot_HSICEstimator_Ustat = 2
ot_HSICEstimator_AsymptoticPValuesEstimator = 1
ot_HSICEstimator_PermutationPValuesEstimator = 2

N = 10 
X1 = ot.Uniform(-np.pi,np.pi)
X2 = ot.Uniform(-np.pi,np.pi)
X3 = ot.Uniform(-np.pi,np.pi)
X = ot.ComposedDistribution([X1,X2,X3])
X = ot.RandomVector(X)

fun = ot.SymbolicFunction(['x1','x2','x3'],['sin(x1) + 5*sin(x2)^2 + 0.1*(x3)^4*sin(x1)'])

def f(x):
    [x1,x2,x3] = x
    return np.sin(x1) + 5*np.sin(x2)**2 + 0.1*(x3)**4*np.sin(x1)

d = 3

Sample = X.getSample(N)
outputSample = fun(Sample)
Sample.stack(outputSample)
inputSample = Sample[:,0:d]
Sample.exportToCSVFile('sample.csv', ',')

C = [[5.,np.inf]]

# v = ot.CompositeRandomVector(fun, X)
# sample = v.getSample(N)
# loaded_sample = sample
# loaded_sample = ot.Sample.ImportFromCSVFile('sample.csv')
# inputSample = loaded_sample[:,0:d]
# outputSample = loaded_sample[:,d]

x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale([inputSample[:,i].computeStandardDeviation()[0,0]]) #Gaussian kernel parameterization
    cov.setNuggetFactor(0.)
    x_covariance_collection.append(cov)
    
y_covariance = ot.SquaredExponential()
y_covariance.setScale([outputSample.computeStandardDeviation()[0,0]])  #Gaussian kernel parameterization

CovarianceList = [x_covariance_collection,y_covariance]
 
''' GSA test'''
# Estimator = HSICEstimators.GSAHSICEstimator(CovarianceList, inputSample, outputSample, ot_HSICEstimator_Vstat)

# Estimator.computeIndices()
# print(Estimator.getR2HSICIIndices())
# print(Estimator.HSIC_XY)

# Estimator.drawR2HSICIIndices()

# Estimator.parameterizePValueEstimator(ot_HSICEstimator_PermutationPValuesEstimator, 1000 )
# Estimator.computePValues()
# print(Estimator.getPValues())

''' TSA test'''
weightFunction = HSICSAWeightFunctions.HSICSAExponentialWeightFunction(C, [0.5,outputSample.computeStandardDeviation()[0,0]])

yw = np.empty(N)
for i in range(N):
    yw[i] = weightFunction.function(outputSample[i])
    
CovarianceList[-1].setScale([np.std(yw, ddof=1)])  #Gaussian kernel parameterization

# yw = ot.Sample(100,yw)
# s = ot.Sample(100,[yw])
# s.BuildFromPoint(yw)
# CovarianceList[-1].setScale([yw.computeStandardDeviation()[0,0]])  #Gaussian kernel parameterization
   
    
Estimator = HSICEstimators.TSAHSICEstimator(CovarianceList, inputSample, outputSample, ot_HSICEstimator_Vstat,weightFunction)


Estimator.computeIndices()
print(Estimator.getR2HSICIIndices())
print(Estimator.HSIC_XY)

Estimator.drawR2HSICIIndices()

Estimator.parameterizePValueEstimator(ot_HSICEstimator_AsymptoticPValuesEstimator)
Estimator.computePValues()
print(Estimator.getPValues())



