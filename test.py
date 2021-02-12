import numpy as np
import openturns as ot
import HSICEstimators
import matplotlib.pyplot as plt
plt.close('all')

ot_HSICEstimator_Vstat = 1
ot_HSICEstimator_Ustat = 2

N = 100 
X1 = ot.Uniform(-np.pi,np.pi)
X2 = ot.Uniform(-np.pi,np.pi)
X3 = ot.Uniform(-np.pi,np.pi)
X = ot.ComposedDistribution([X1,X2,X3])
X = ot.RandomVector(X)

fun = ot.SymbolicFunction(['x1','x2','x3'],['sin(x1) + 5*sin(x2)^2 + 0.1*(x3)^4*sin(x1)'])

d = 3

Sample = X.getSample(N)
outputSample = fun(Sample)
Sample.stack(outputSample)
inputSample = Sample[:,0:d]
Sample.exportToCSVFile('sample.csv', ',')

# v = ot.CompositeRandomVector(fun, X)
# sample = v.getSample(N)
# loaded_sample = sample
# loaded_sample = ot.Sample.ImportFromCSVFile('sample.csv')
# inputSample = loaded_sample[:,0:d]



x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    # cov.setScale([1/inputSample[:,i].computeVariance()[0]]) #Gaussian kernel parameterization
    cov.setScale([inputSample[:,i].computeStandardDeviation()[0,0]]) #Gaussian kernel parameterization
    cov.setNuggetFactor(0.)
    x_covariance_collection.append(cov)
    
y_covariance = ot.SquaredExponential()
# y_covariance.setScale([1/outputSample.computeVariance()[0]])  #Gaussian kernel parameterization
y_covariance.setScale([outputSample.computeStandardDeviation()[0,0]])  #Gaussian kernel parameterization

CovarianceList = [x_covariance_collection,y_covariance]

Estimator = HSICEstimators.GSAHSICEstimator(CovarianceList, inputSample, outputSample, ot_HSICEstimator_Ustat)

Estimator.computeIndices()
print(Estimator.getR2HSICIIndices())
print(Estimator.HSIC_XY)

Estimator.drawR2HSICIIndices()

# Estimator.parameterizePValueEstimator(ot.HSICEstimator.AsymptoticPValuesEstimator)
# Estimator.computePValues()
# Estimator.drawPValues()

