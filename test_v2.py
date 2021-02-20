import numpy as np
import openturns as ot
import HSICEstimators
import matplotlib.pyplot as plt
import HSICSAWeightFunctions

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

ot_HSICEstimator_Vstat = 1
ot_HSICEstimator_Ustat = 2
ot_HSICEstimator_AsymptoticPValuesEstimator = 1
ot_HSICEstimator_PermutationPValuesEstimator = 2

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
        [inputSample[:, i].computeStandardDeviation()[0, 0]]
    )  # Gaussian kernel parameterization
    cov.setNuggetFactor(0.0)
    x_covariance_collection.append(cov)


"""test parameters"""
# Estimatortype = ot_HSICEstimator_Ustat
Estimatortype = ot_HSICEstimator_Vstat

pValueEstimation = ot_HSICEstimator_PermutationPValuesEstimator
# pValueEstimation = ot_HSICEstimator_AsymptoticPValuesEstimator

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
        C, [0.5, outputSample.computeStandardDeviation()[0, 0]]
    )
elif weightf == "Ind":
    weightFunction = HSICSAWeightFunctions.HSICSAStepWeightFunction(C)


if SA == "GSA" or "CSA":
    y_covariance = ot.SquaredExponential()
    y_covariance.setScale(
        [outputSample.computeStandardDeviation()[0, 0]]
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
        CovarianceList, inputSample, outputSample, Estimatortype, weightFunction
    )

if SA == "CSA":
    Estimator = HSICEstimators.CSAHSICEstimator(
        CovarianceList, inputSample, outputSample, Estimatortype, weightFunction
    )


"""Testing"""

Estimator.computeIndices()
print(Estimator.getR2HSICIIndices())
print(Estimator.HSIC_XY)

if pValueEstimation == ot_HSICEstimator_PermutationPValuesEstimator:
    Estimator.parameterizePValueEstimator(pValueEstimation, 10000)
else:
    Estimator.parameterizePValueEstimator(pValueEstimation)

Estimator.computePValues()
print(Estimator.getPValues())
