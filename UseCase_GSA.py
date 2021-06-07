import numpy as np
import openturns as ot
import HSICEstimators
import matplotlib.pyplot as plt
import HSICSAWeightFunctions
from HSICStat import HSICvStat, HSICuStat
from openturns.viewer import View

"""Test -case definition"""
X1 = ot.Uniform(-np.pi, np.pi)
X2 = ot.Uniform(-np.pi, np.pi)
X3 = ot.Uniform(-np.pi, np.pi)
X = ot.ComposedDistribution([X1, X2, X3])
X = ot.RandomVector(X)
fun = ot.SymbolicFunction(
    ["x1", "x2", "x3"], ["sin(x1) + 5*sin(x2)^2 + 0.1*(x3)^4*sin(x1)"]
)


"""Load data"""
plt.close("all")

loaded_sample = ot.Sample.ImportFromCSVFile("sample.csv", ",")
d = loaded_sample.getDimension() - 1
N = loaded_sample.getSize()

inputSample = loaded_sample[:, 0:d]
outputSample = loaded_sample[:, d]

"""Covariance models associated to the inputs. An empyrical parameterization rule is used for the lenghtscale parameters"""
x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(
        [inputSample[:, i].computeStandardDeviation()[0]]
    )  # Empirical Gaussian kernel parameterization, ATTENTION, THIS VARIES DEPENDING ON THE PACKAGE VERSION!
    cov.setNuggetFactor(0.0)
    x_covariance_collection.append(cov)

y_covariance = ot.SquaredExponential()
y_covariance.setScale(
    [outputSample.computeStandardDeviation()[0]]
)  # Empirical Gaussian kernel parameterization

CovarianceList = [x_covariance_collection, y_covariance]


"""test parameters"""
# Estimatortype = HSICuStat() #U-stat HSIC estimator
Estimatortype = HSICvStat() #V-stat HSIC estimator


Estimator = HSICEstimators.GSAHSICEstimator(
    CovarianceList, inputSample, outputSample, Estimatortype
)

"""Testing"""

print(Estimator.getR2HSICIndices())
print(Estimator.getHSICIndices())

B = 1000  # Only used for permutatio p-value estimation

#Permutation-based estimation of p-values
Estimator.setPermutationBootstrapSize(B)
print(Estimator.getPValuesPermutation())

#Asymptotic estimation of p-values
print(Estimator.getPValuesAsymptotic())

View(Estimator.drawR2HSICIndices())
View(Estimator.drawPValuesAsymptotic())
