import numpy as np
import openturns as ot
import HSICEstimators
import HSICSAWeightFunctions
from HSICStat import HSICvStat, HSICuStat
from KroneckerCov import KroneckerKernel

"""Definition of the kronecker covariance function"""
kronCov = KroneckerKernel()

"""Test -case definition"""
X1 = ot.Normal()
X2 = ot.Uniform()
X = ot.ComposedDistribution([X1, X2])
X = ot.RandomVector(X)
fun = ot.SymbolicFunction(
    ["x1", "x2"], ["min(x1,x2)"]
)
d = 2
N = 100

ot.RandomGenerator.SetSeed(0)
inputSample = X.getSample(N)
outputSample = fun(inputSample)


inputSample = ot.Sample([[0.608202,-0.632279],[-1.26617,-0.983758],[-0.438266,-0.567843],
                         [1.20548,0.196576],[-2.18139,0.297528],[0.350042,0.712356],
                         [-0.355007,-0.833663],[1.43725,-0.0922891],[0.810668,0.0103994],
                         [0.793156,-0.959137],[-0.470526,-0.653801],[0.261018,-0.629697],
                         [-2.29006,0.691573],[-1.28289,0.00556723],[-1.31178,-0.456731],
                         [-0.0907838,0.282551],[0.995793,0.127445],[-0.139453,-0.889423],
                         [-0.560206,0.602221],[0.44549,-0.708005],[0.322925,0.560746],
                         [0.445785,0.210698],[-1.03808,-0.231704],[-0.856712,-0.0757071],
                         [0.473617,-0.747651],[-0.125498,-0.717586],[0.351418,0.533389],
                         [1.78236,-0.601014],[0.0702074,-0.283657],[-0.781366,0.421392],
                         [-0.721533,0.97622],[-0.241223,0.649341],[-1.78796,-0.447644],
                         [0.40136,0.524658],[1.36783,-0.310752],[1.00434,0.466132],
                         [0.741548,0.318871],[-0.0436123,0.45667],[0.539345,0.419467],
                         [0.29995,-0.785465],[0.407717,-0.957343],[-0.485112,-0.888291],
                         [-0.382992,-0.13238],[-0.752817,0.881545],[0.257926,0.230244],
                         [1.96876,-0.408034],[-0.671291,0.74598],[1.85579,-0.624525],
                         [0.0521593,0.790653],[0.790446,0.359935],[0.716353,0.868061],
                         [-0.743622,-0.28315],[0.184356,-0.605466],[-1.53073,0.975779],
                         [0.655027,0.415187],[0.538071,0.0840439],[1.73821,-0.278904],
                         [-0.958722,0.803063],[0.377922,0.745595],[-0.181004,-0.359175],
                         [1.67297,0.992755],[-1.03896,-0.376385],[-0.353552,-0.558697],
                         [1.21381,-0.300297],[-0.777033,0.898571],[-1.36853,-0.904539],
                         [0.103474,-0.841734],[-0.89182,-0.878618],[0.905602,-0.904866],
                         [0.334794,-0.209856],[-0.483642,0.742799],[0.677958,-0.984481],
                         [1.70938,0.926643],[1.07062,-0.699162],[-0.506925,0.536564],
                         [-1.66086,0.0400396],[2.24623,-0.325119],[0.759602,0.639517],
                         [-0.510764,-0.182467],[-0.633066,-0.975622],[-0.957072,0.00729759],
                         [0.544047,-0.411715],[0.814561,-0.540526],[-0.734708,0.685758],
                         [-0.111461,0.252678],[0.994482,-0.765926],[-0.160625,-0.991518],
                         [-0.938771,-0.808189],[-1.96869,-0.171358],[-0.657603,-0.516637],
                         [0.338751,0.396634],[1.01556,0.586587],[0.637167,-0.369344],
                         [-0.0899071,0.0029418],[-0.855886,0.866255],[1.27128,0.942399],
                         [-0.238253,-0.569602],[1.3263,-0.27135],[2.11968,0.427897],
                         [-0.901581,0.276064]])

outputSample = ot.Sample([[-0.632279],[-1.26617],[-0.567843],[0.196576],[-2.18139],
                          [0.350042],[-0.833663],[-0.0922891],[0.0103994],[-0.959137],
                          [-0.653801],[-0.629697],[-2.29006],[-1.28289],[-1.31178],
                          [-0.0907838],[0.127445],[-0.889423],[-0.560206],[-0.708005],
                          [0.322925],[0.210698],[-1.03808],[-0.856712],[-0.747651],
                          [-0.717586],[0.351418],[-0.601014],[-0.283657],[-0.781366],
                          [-0.721533],[-0.241223],[-1.78796],[0.40136],[-0.310752],
                          [0.466132],[0.318871],[-0.0436123],[0.419467],[-0.785465],
                          [-0.957343],
                          [-0.888291],[-0.382992],[-0.752817],[0.230244],[-0.408034],
                          [-0.671291],[-0.624525],[0.0521593],[0.359935],[0.716353],
                          [-0.743622],[-0.605466],[-1.53073],[0.415187],[0.0840439],
                          [-0.278904],[-0.958722],[0.377922],[-0.359175],[0.992755],
                          [-1.03896],[-0.558697],[-0.300297],[-0.777033],[-1.36853],
                          [-0.841734],[-0.89182],[-0.904866],[-0.209856],[-0.483642],
                          [-0.984481],[0.926643],[-0.699162],[-0.506925],[-1.66086],
                          [-0.325119],[0.639517],[-0.510764],[-0.975622],[-0.957072],
                          [-0.411715],[-0.540526],[-0.734708],[-0.111461],[-0.765926],
                          [-0.991518],[-0.938771],[-1.96869],[-0.657603],[0.338751],
                          [0.586587],[-0.369344],[-0.0899071],[-0.855886],[0.942399],
                          [-0.569602],[-0.27135],[0.427897],[-0.901581]])

C = [[0.62, np.inf]] 

"""Covariance models associated to the inputs. An empyrical parameterization rule is used for the lenghtscale parameters"""
x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(
        [inputSample[:, i].computeStandardDeviation()[0]]
    )  # Gaussian kernel parameterization, ATTENTION, THIS VARIES DEPENDING ON THE PACKAGE VERSION!
    cov.setNuggetFactor(0.0)
    x_covariance_collection.append(cov)


def computeValues(Estimatortype,OutputCov,SA, weightf = None):
    """Initialization"""
    if weightf == "Exp":
        weightFunction = HSICSAWeightFunctions.HSICSAExponentialWeightFunction(
            C, [0.5, outputSample.computeStandardDeviation()[0]] # ATTENTION, THIS VARIES DEPENDING ON THE PACKAGE VERSION!
        )
    elif weightf == "Ind":
        weightFunction = HSICSAWeightFunctions.HSICSAStepWeightFunction(C)
    
    
    if SA == "GSA" or "CSA":
        y_covariance = ot.SquaredExponential()
        y_covariance.setScale(
            [outputSample.computeStandardDeviation()[0]]
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
    
    print('R2' , Estimator.getR2HSICIndices())
    print('HSIC' , Estimator.getHSICIndices())
    
    if SA != 'CSA':
        print('p-val asymptotic', Estimator.getPValuesAsymptotic())

    Estimator.setPermutationBootstrapSize(1000)
    print('p-val permutation',  Estimator.getPValuesPermutation())



print('GSA V-stat, Covariance Gaussienne ')
Estimatortype = HSICvStat()
OutputCov = "Exp"
SA = 'GSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,)


print('GSA U-stat, Covariance Gaussienne ')
Estimatortype = HSICuStat()
OutputCov = "Exp"
SA = 'GSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,)


print('TSA V-stat, Seuillage dur, Kronecker covariance ')
Estimatortype = HSICvStat()
OutputCov = "Kron"
weightf = 'Ind' #Only used for TSA
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('TSA U-stat, Seuillage dur, Kronecker covariance ')
Estimatortype = HSICuStat()
OutputCov = "Kron"
weightf = 'Ind' #Only used for TSA
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('TSA V-stat, Fonction de poids exponentielle, Covariance Gaussienne ')
Estimatortype = HSICvStat()
OutputCov = "Exp"
weightf = 'Exp' 
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('TSA U-stat, Fonction de poids exponentielle, Covariance Gaussienne ')
Estimatortype = HSICuStat()
OutputCov = "Exp"
weightf = 'Exp' 
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('CSA V-stat, Fonction de poids exponentielle, Covariance Gaussienne ')
Estimatortype = HSICvStat()
OutputCov = "Exp"
weightf = 'Exp' 
SA = 'CSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('CSA V-stat, Seuillage dur, Covariance Gaussienne ')
Estimatortype = HSICvStat()
OutputCov = "Exp"
weightf = 'Ind' 
SA = 'CSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


