import numpy as np
import openturns as ot
import HSICEstimators
import matplotlib.pyplot as plt
import HSICSAWeightFunctions
from HSICStat import HSICvStat, HSICuStat

"""Test -case definition"""
X1 = ot.Uniform(-np.pi, np.pi)
X2 = ot.Uniform(-np.pi, np.pi)
X3 = ot.Uniform(-np.pi, np.pi)
X = ot.ComposedDistribution([X1, X2, X3])
X = ot.RandomVector(X)
fun = ot.SymbolicFunction(
    ["x1", "x2", "x3"], ["sin(x1) + 5*sin(x2)^2 + 0.1*(x3)^4*sin(x1)"]
)


"""Definition of the kronecker covariance function"""


def rho(t):
    x = t[0]
    if x == 0:
        return [1.0]
    else:
        return [0.0]


rho = ot.PythonFunction(1, 1, rho)
kronCov = ot.StationaryFunctionalCovarianceModel([1.0], [1.0], rho)

"""Load data"""
plt.close("all")

loaded_sample = ot.Sample.ImportFromCSVFile("sample.csv", ",")
d = loaded_sample.getDimension() - 1
N = loaded_sample.getSize()

inputSample = loaded_sample[:, 0:d]
outputSample = loaded_sample[:, d]

C = [[5.0, np.inf]]

x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(
        inputSample[:, i].computeStandardDeviation()
    )  # Gaussian kernel parameterization
    cov.setNuggetFactor(0.0)
    x_covariance_collection.append(cov)


"""test parameters"""
# Estimatortype = HSICuStat()
Estimatortype = HSICvStat()

B = 1000  # Only used for permutatio p-value estimation

weightf = "Exp"  # Only used for CSA and TSA
# weightf = 'Ind' #Only used for CSA and TSA

# OutputCov = 'Kron' #Only used for TSA and Ind weight function
OutputCov = "Exp"

# SA = 'GSA'
# SA = 'TSA'
SA = "CSA"

"""Initialization"""
if weightf == "Exp":
    weightFunction = HSICSAWeightFunctions.HSICSAExponentialWeightFunction(
        C, [0.5, outputSample.computeStandardDeviation()[0]]
    )
elif weightf == "Ind":
    weightFunction = HSICSAWeightFunctions.HSICSAStepWeightFunction(C)


if SA == "GSA" or "CSA":
    y_covariance = ot.SquaredExponential()
    y_covariance.setScale(
        outputSample.computeStandardDeviation()
    )  # Gaussian kernel parameterization


if SA == "TSA":
    if OutputCov == "Exp":
        y_covariance = ot.SquaredExponential()
        yw = np.empty(N)
        for i in range(N):
            yw[i] = weightFunction.function(outputSample[i])
        y_covariance.setScale([np.std(yw, ddof=1)])  # Gaussian kernel parameterization
    elif OutputCov == "Ind":
        weightFunction = HSICSAWeightFunctions.HSICSAStepWeightFunction(C)


CovarianceList = [x_covariance_collection, y_covariance]


if SA == "GSA":
    Estimator = HSICEstimators.GSAHSICEstimator(
        CovarianceList, inputSample, outputSample, Estimatortype
    )

if SA == "TSA":
    Estimator = HSICEstimators.TSAHSICEstimator(
        CovarianceList, inputSample, outputSample, weightFunction, Estimatortype
    )

if SA == "CSA":
    Estimator = HSICEstimators.CSAHSICEstimator(
        CovarianceList, inputSample, outputSample, weightFunction, Estimatortype
    )


"""Testing"""

Estimator.computeIndices()
print(Estimator.getR2HSICIIndices())
print(Estimator.HSIC_XY)

Estimator.setPermutationBootstrapSize(100)

print(Estimator.getPValuesPermutation())
