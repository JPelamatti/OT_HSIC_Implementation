import numpy as np
import openturns as ot
import HSICEstimators
import HSICSAWeightFunctions
from HSICStat import HSICvStat, HSICuStat
from KroneckerCov import KroneckerKernel

"""Definition of the kronecker covariance function"""
kronCov = KroneckerKernel()

"""Test -case definition"""
X1 = ot.Uniform(-np.pi, np.pi)
X2 = ot.Uniform(-np.pi, np.pi)
X3 = ot.Uniform(-np.pi, np.pi)
X = ot.ComposedDistribution([X1, X2, X3])
X = ot.RandomVector(X)
fun = ot.SymbolicFunction(
    ["x1", "x2", "x3"], ["sin(x1) + 5*sin(x2)^2 + 0.1*(x3)^4*sin(x1)"]
)

d = 3
N = 100

# ot.RandomGenerator.SetSeed(0)
# inputSample = X.getSample(N)
# outputSample = fun(inputSample)

inputSample = ot.Sample([[0.816038,0.00821179,-2.33097],[2.40524,0.0904902,2.66682],
                         [-2.29163,2.58692,0.152173],[-2.93737,1.38302,3.04918],
                         [-0.960969,0.546645,3.07094],[2.94947,2.03963,-1.1917],
                         [2.64321,1.33298,-1.68788],[0.0191018,-2.20727,-0.347263],
                         [-2.74446,2.03611,-1.98636],[-1.30215,0.376116,-3.09057],
                         [1.347,0.96388,-1.78393],[-0.732858,-3.01627,0.617561],
                         [-0.793142,1.39837,0.934712],[1.4908,1.53625,2.23793],
                         [2.40962,-1.34499,-2.61903],[-1.28809,2.24783,-0.289935],
                         [2.69241,0.155106,0.0326708],[2.01571,2.8642,-3.01322],
                         [1.15972,-1.59375,-2.05398],[2.06106,-1.72058,-1.97825],
                         [-0.880889,2.86803,2.17264],[2.85725,0.650095,0.01749],
                         [0.556787,-0.0612079,-1.43486],[-1.99781,-0.382635,0.887661],
                         [-2.60259,-2.32225,0.40038],[1.00988,2.68503,-2.79421],
                         [-1.81935,-2.50383,1.89193],[-0.714846,-1.09471,-2.22426],
                         [-2.98728,-0.625538,1.76164],[-0.509449,-0.592749,0.661928],
                         [3.0275,-0.634769,-0.727921],[2.61463,2.47638,-0.237841],
                         [2.86264,-2.41144,-2.34882],[-0.16996,-1.78677,-2.25436],
                         [-1.50884,-2.37174,1.67569],[1.01373,-2.21592,-1.88814],
                         [-0.0642485,-0.952468,-0.891133],[-0.197767,0.154282,1.32384],
                         [-2.44367,-2.2469,3.06688],[-0.97189,-1.30398,2.03996],
                         [0.980033,0.190456,-1.40631],[1.10011,1.36325,1.64826],
                         [3.00952,-1.59859,-0.976256],[0.919445,0.489522,1.4644],
                         [-2.73814,2.41795,1.00176],[1.71543,-1.29261,1.43467],
                         [3.10515,2.07528,1.31779],[-2.61963,1.1862,-2.46761],
                         [-1.72871,-2.76117,-3.00758],[-1.24018,1.90013,-2.79065],
                         [-1.77682,-0.934969,-0.415883],[0.814707,1.82601,2.76945],
                         [0.503618,-0.0419387,0.723332],[-0.206793,-1.16566,-1.28188],
                         [1.23147,1.41855,2.34357],[1.51487,-3.08316,-1.962],
                         [2.74461,0.36155,2.48391],[2.33388,2.65318,1.13077],
                         [-0.671169,-0.969187,2.7271],[0.81209,1.66555,-0.889543],
                         [2.59828,0.729746,-1.90213],[2.59748,-2.01657,3.0655],
                         [-2.14014,-0.76217,1.30435],[0.279213,-2.64467,0.264032],
                         [-1.86201,0.406361,-0.876202],[2.23553,-1.64634,2.5229],
                         [-0.407915,1.07561,2.34236],[-0.869188,0.993014,-1.12838],
                         [-3.12835,-0.358906,3.11883],[-1.63719,1.53796,-1.18245],
                         [-1.5384,2.21936,-1.7552],[0.575127,-1.62607,-0.943412],
                         [-2.07837,-2.91323,2.82294],[2.99918,-0.284413,-2.84169],
                         [-0.106867,1.97919,-2.64439],[1.31512,2.7585,-2.76026],
                         [2.87735,2.19212,-2.84272],[1.2756,-0.362291,-0.659281],
                         [2.01174,-1.12515,2.33357],[-0.113845,-1.56037,-3.09284],
                         [-1.84369,0.282831,2.91113],[1.5783,-3.10369,-2.19648],
                         [2.67369,2.26279,1.68567],[0.854031,-0.932592,0.125788],
                         [0.641903,1.71654,-1.02139],[1.26609,0.679145,2.0091],
                         [-2.64261,-0.117584,-0.573238],[-1.55614,1.90126,-3.06501],
                         [1.11197,0.787193,0.0229261],[-0.889369,1.4796,-1.29344],
                         [-0.501843,0.64429,-1.69811],[-1.12246,-0.0947317,2.15437],
                         [1.87616,-2.22588,0.79381],[0.474772,-1.37366,-2.40623],
                         [-1.07619,-1.69232,-3.11495],[2.96367,2.05846,-2.539],
                         [0.110757,-0.690349,-0.538338],[-2.46414,-1.57203,-1.62306],
                         [2.56221,-1.65231,1.24606],[1.56919,1.61971,1.84282]])

outputSample = ot.Sample([[2.87928],[4.1093],[0.6356],[2.86985],[-6.75928],[4.20863],
                         [5.5885],[3.2528],[3.00441],[-9.08559],[5.33616],[-0.600603],
                         [4.08585],[8.49117],[8.56224],[2.07663],[0.553557],[8.71878],
                         [7.54562],[7.12201],[-2.12497],[2.11223],[0.771174],[-0.269706],
                         [2.15498],[6.98033],[-0.43859],[1.68987],[1.41252],[1.06335],
                         [1.87524],[2.40792],[3.33775],[4.16436],[0.637245],[5.11982],
                         [3.25164],[-0.138756],[-3.28573],[2.3961],[1.33455],[6.33677],
                         [5.13979],[2.26644],[1.75987],[6.03174],[3.87926],[1.94899],
                         [-8.37859],[-2.20522],[2.25497],[9.68867],[0.504597],[3.96251],
                         [8.67249],[2.49501],[2.48402],[1.94173],[-0.663231],[5.72641],
                         [3.41603],[9.15963],[1.29785],[1.41205],[-0.233168],[8.94734],
                         [2.28006],[2.62087],[0.478347],[3.80175],[1.22758],[5.57177],
                         [-6.16756],[1.46112],[3.58318],[7.28235],[5.27253],[1.60288],
                         [7.65712],[3.8464],[-7.48986],[3.33471],[3.77946],[3.97934],
                         [5.55843],[4.48094],[-0.414885],[-5.35064],[3.40555],[3.96448],
                         [0.922755],[-2.79772],[4.13582],[6.79779],[-4.23996],[4.81475],
                         [2.13907],[3.93819],[5.64635],[7.14131]])
C = [[5.0, np.inf]] 

"""Covariance models associated to the inputs. An empyrical parameterization rule is used for the lenghtscale parameters"""
x_covariance_collection = []
for i in range(d):
    cov = ot.SquaredExponential()
    cov.setScale(
        [inputSample[:, i].computeStandardDeviation()[0,0]]
    )  # Gaussian kernel parameterization, ATTENTION, THIS VARIES DEPENDING ON THE PACKAGE VERSION!
    cov.setNuggetFactor(0.0)
    x_covariance_collection.append(cov)


def computeValues(Estimatortype,OutputCov,SA, weightf = None):
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
    
    print('R2-HSIC : ' , Estimator.getR2HSICIndices())
    print('\n')
    print('HSIC : ' , Estimator.getHSICIndices())
    print('\n')
    if SA != 'CSA':
        print('p-valeurs asymptotiques : ', Estimator.getPValuesAsymptotic())
        print('\n')
    Estimator.setPermutationBootstrapSize(1000)
    print('p-valeurs permutation : ',  Estimator.getPValuesPermutation())
    print('\n')


print('GSA V-statistic, Covariance Gaussienne ')
print('\n')
Estimatortype = HSICvStat()
OutputCov = "Exp"
SA = 'GSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,)


print('GSA U-statistic, Covariance Gaussienne ')
print('\n')
Estimatortype = HSICuStat()
OutputCov = "Exp"
SA = 'GSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,)


print('TSA V-statistic, Seuillage dur, Noyau de Kronecker  ')
print('\n')
Estimatortype = HSICvStat()
OutputCov = "Kron"
weightf = 'Ind' #Only used for TSA
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('TSA U-statistic, Seuillage dur, Noyau de Kronecker ')
print('\n')
Estimatortype = HSICuStat()
OutputCov = "Kron"
weightf = 'Ind' #Only used for TSA
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('TSA V-statistic, Fonction de poids exponentielle, Covariance Gaussienne ')
print('\n')
Estimatortype = HSICvStat()
OutputCov = "Exp"
weightf = 'Exp' 
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('TSA U-statistic, Fonction de poids exponentielle, Covariance Gaussienne ')
print('\n')
Estimatortype = HSICuStat()
OutputCov = "Exp"
weightf = 'Exp' 
SA = 'TSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('CSA V-stat, Fonction de poids exponentielle, Covariance Gaussienne ')
print('\n')
Estimatortype = HSICvStat()
OutputCov = "Exp"
weightf = 'Exp' 
SA = 'CSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


print('CSA V-statistic, Seuillage dur, Covariance Gaussienne ')
print('\n')
Estimatortype = HSICvStat()
OutputCov = "Exp"
weightf = 'Ind' 
SA = 'CSA' #Global sensitivity analysis

computeValues(Estimatortype,OutputCov,SA,weightf)


