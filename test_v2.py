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


"""Definition of the kronecker covariance function"""
mesh  = ot.Mesh([[0.],[1.]]) 
covMat = ot.CovarianceMatrix(2)
kronCov = ot.UserDefinedCovarianceModel(mesh,covMat)


"""Load data"""
plt.close("all")

loaded_sample = ot.Sample.ImportFromCSVFile("sample.csv", ",")
d = loaded_sample.getDimension() - 1
N = loaded_sample.getSize()

inputSample = loaded_sample[:, 0:d]
outputSample = loaded_sample[:, d]

C = [[5.0, np.inf]] #Critical domain, this is used for TSA and CSA. In case multiple critical domains are considered, one can define C = [[-10.0,1.5],[33.1,46.0]]

"""Covariance models associated to the inputs. An empyrical parameterization rule is used for the lenghtscale parameters"""
x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(
        [inputSample[:, i].computeStandardDeviation()[0,0]]
    )  # Gaussian kernel parameterization, ATTENTION, THIS VARIES DEPENDING ON THE PACKAGE VERSION!
    cov.setNuggetFactor(0.0)
    x_covariance_collection.append(cov)


"""test parameters"""
# Estimatortype = HSICuStat()
Estimatortype = HSICvStat()

B = 1000  # Only used for permutatio p-value estimation

# weightf = "Exp"  # Only used for CSA and TSA
weightf = 'Ind' #Only used for TSA

# OutputCov = 'Kron' #Only used for TSA combined with Ind weight function
OutputCov = "Exp"

# SA = 'GSA' #Global sensitivity analysis
# SA = 'TSA' #Target sensitivity analysis
SA = "CSA" #Conditional sensitivity analysis

"""Initialization"""
if weightf == "Exp":
    weightFunction = HSICSAWeightFunctions.HSICSAExponentialWeightFunction(
        C, [0.5, outputSample.computeStandardDeviation()[0,0]] # ATTENTION, THIS VARIES DEPENDING ON THE PACKAGE VERSION!
    )
elif weightf == "Ind":
    weightFunction = HSICSAWeightFunctions.HSICSAStepWeightFunction(C)


if SA == "GSA" or "CSA":
    y_covariance = ot.SquaredExponential()
    y_covariance.setScale(
        [outputSample.computeStandardDeviation()[0,0]]
    )  # Gaussian kernel parameterization


if SA == "TSA":
    if OutputCov == "Exp":
        y_covariance = ot.SquaredExponential()
        yw = weightFunction.function(outputSample)
        y_covariance.setScale([np.std(yw, ddof=1)])  # Gaussian kernel parameterization
    elif OutputCov == "Kron":
        y_covariance = kronCov


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

print(Estimator.getR2HSICIndices())
print(Estimator.getHSICIndices())

Estimator.setPermutationBootstrapSize(1000)
print(Estimator.getPValuesPermutation())

# print(Estimator.getPValuesAsymptotic())

View(Estimator.drawR2HSICIndices())