import numpy as np
import openturns as ot
from HSICStat import HSICvStat
import copy

class CSAHSICEstimator:
    """
    Conditional sensitivity analysis estimator, only V-stat estimator and asymptotic p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, weightFunction, HSICstat=HSICvStat()):
        if not HSICstat._isCSACompatible():
            raise TypeError(
                "Chosen {}-stat estimator not available for CSA".format(
                    HSICstat.getStatLetter()
                )
            )
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICstat = HSICstat
        self.weightFunction = weightFunction
        self.n = X.getSize()
        self.d = X.getDimension()
        self.PValuesAsymptotic = ot.Point()
        self.HSIC_XY = ot.Point()
        self.R2HSICIndices = ot.Point()
        self.setPermutationBootstrapSize(1000)

    def getStatLetter(self):
        """Return the letter of the statistic used as HSIC estimator."""
        return self.HSICstat.getStatLetter()

    def _computeWeightMatrix(self, Y):
        
        if self.weightFunction == None:  # GSA case
            W = np.eye(self.n)
            
        else:            
            W = np.zeros((self.n, self.n))
    
            w = np.array(self.weightFunction.function(Y))
            w = w / np.mean(w)
            np.fill_diagonal(W, w)

        return W

    def _computeHSICIndex(self, V1, V2, Cov1, Cov2, W):
        return self.HSICstat._computeHSICIndex(V1, V2, Cov1, Cov2, W)

    def _computeIndices(self):
        W = self._computeWeightMatrix(self.Y)

        self.HSIC_XX = ot.Point()

        for dim in range(self.d):

            self.HSIC_XY.add(
                self._computeHSICIndex(
                    self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W
                )
            )
            self.HSIC_XX.add(
                self._computeHSICIndex(
                    self.X[:, dim], self.X[:, dim], self.CovX[dim], self.CovX[dim], W
                )
            )

        self.HSIC_YY = self._computeHSICIndex(self.Y, self.Y, self.CovY, self.CovY, W)

        self.R2HSICIndices = ot.Point()
        for dim in range(self.d):
            self.R2HSICIndices.add(
                self.HSIC_XY[dim] / np.sqrt(self.HSIC_XX[dim] * self.HSIC_YY)
            )

        return 0

    def setPermutationBootstrapSize(self, B):
        self.PValuesPermutation = ot.Point()
        self.PermutationBootstrapSize = B

    def getPermutationBootstrapSize(self):
        return self.PermutationBootstrapSize

    def _computePValuesPermutation(self):

        W_obs = self._computeWeightMatrix(self.Y)
        self.PValuesPermutation = ot.Point()

        N_permutations = ot.KPermutationsDistribution(self.n, self.n).getSample(self.PermutationBootstrapSize)
        permuted_values = [[int(x) for x in N_permutations[j]] for j in range(self.PermutationBootstrapSize)]

        for dim in range(self.d):
            HSIC_obs = self._computeHSICIndex(
                self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W_obs
            )
            HSIC_l = []
            for perm in permuted_values:
                Y_p = self.Y[perm]

                W = self._computeWeightMatrix(Y_p)

                HSIC_l.append(
                    self._computeHSICIndex(
                        self.X[:, dim], Y_p, self.CovX[dim], self.CovY, W
                    )
                )

            p = np.count_nonzero(np.array(HSIC_l) > HSIC_obs) / (
                self.PermutationBootstrapSize + 1
            )

            self.PValuesPermutation.add(p)
        return 0

    def _computePValuesAsymptotic(self):
        raise ValueError(
            "Asymptotic p-value estimator not available for CSA, a permutation-based estimator must be used"
        )

    def getHSICIndices(self):
        if self.HSIC_XY.getDimension() == 0:
            self._computeIndices()
        return ot.Point(self.HSIC_XY)

    def getR2HSICIndices(self):
        if self.R2HSICIndices.getDimension() == 0:
            self._computeIndices()                  
        return ot.Point(self.R2HSICIndices)

    def getPValuesPermutation(self):
        if self.PValuesPermutation.getDimension() == 0:
            self._computePValuesPermutation()

        return ot.Point(self.PValuesPermutation)

    def getPValuesAsymptotic(self):
        if self.PValuesAsymptotic.getDimension() == 0:
            self._computePValuesAsymptotic()
         
        return ot.Point(self.PValuesAsymptotic)
    


    def drawHSICIndices(self):
        HSICIndices =  self.getHSICIndices()
        graph = ot.SobolIndicesAlgorithm().DrawCorrelationCoefficients(HSICIndices,self.X.getDescription(),'HSIC Indices')
        graph.setAutomaticBoundingBox(True)
        return graph

        
    def drawR2HSICIndices(self):
        R2HSICIndices =  self.getR2HSICIndices()
        graph = ot.SobolIndicesAlgorithm().DrawCorrelationCoefficients(R2HSICIndices,self.X.getDescription(),'R2-HSIC Indices')
        graph.setAutomaticBoundingBox(True)
        return graph

        
    def drawPValuesPermutation(self):
        PValuesPermutation =  self.getPValuesPermutation()
        graph = ot.SobolIndicesAlgorithm().DrawCorrelationCoefficients(PValuesPermutation,self.X.getDescription(),'Permutation-based p-values')
        graph.setAutomaticBoundingBox(True)
        return graph


    def drawPValuesAsymptotic(self):
        PValuesAsymptotic =  self.getPValuesAsymptotic()
        graph = ot.SobolIndicesAlgorithm().DrawCorrelationCoefficients(PValuesAsymptotic,self.X.getDescription(),'Asymptotic p-values')
        graph.setAutomaticBoundingBox(True)
        return graph




class GSAHSICEstimator(CSAHSICEstimator):
    """
    Global sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, HSICstat=HSICvStat()):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICstat = HSICstat
        self.weightFunction = None
        self.n = X.getSize()
        self.d = X.getDimension()
        self.PValuesAsymptotic = ot.Point()
        self.HSIC_XY = ot.Point()
        self.R2HSICIndices = ot.Point()
        self.setPermutationBootstrapSize(1000)

    def _computePValuesAsymptotic(self):
        W = self._computeWeightMatrix(self.Y)

        self.PValuesAsymptotic = ot.Point()

        H = np.eye(self.n) - 1 / self.n * np.ones((self.n, self.n))
        Ky = np.array(self.CovY.discretize(self.Y))
        
        Ky_mdiag = copy.deepcopy(Ky)
        np.fill_diagonal(Ky_mdiag,0.)
            
        Ey = 1 / self.n / (self.n - 1) * np.sum(Ky_mdiag)
        By = H @ Ky @ H

        for dim in range(self.d):
            HSIC_obs = self._computeHSICIndex(
                self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W
            )

            Kx = np.array(self.CovX[dim].discretize(self.X[:, dim]))
            
            Kx_mdiag = copy.deepcopy(Kx)
            np.fill_diagonal(Kx_mdiag,0.)

            Ex = 1 / self.n / (self.n - 1) * np.sum(Kx_mdiag)

            Bx = H @ Kx @ H
            B = np.multiply(Bx, By)
            B = B ** 2
            np.fill_diagonal(B,0.)

            mHSIC = 1 / self.n * (1 + Ex * Ey - Ex - Ey)
            varHSIC = (
                2
                * (self.n - 4)
                * (self.n - 5)
                / self.n
                / (self.n - 1)
                / (self.n - 2)
                / (self.n - 3)
                * np.ones((1, self.n))
                @ B
                @ np.ones((self.n, 1))
                / self.n
                / (self.n - 1)
            )
            varHSIC = varHSIC[0, 0]

            alpha = mHSIC ** 2 / varHSIC
            beta = self.n * varHSIC / mHSIC

            gamma = ot.Gamma(alpha, 1 / beta)
            p = self.HSICstat._computePValue(gamma, self.n, HSIC_obs, mHSIC)

            self.PValuesAsymptotic.add(p)
        return 0


class TSAHSICEstimator(GSAHSICEstimator):
    """
    Target sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, filterFunction, HSICstat=HSICvStat()):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICstat = HSICstat
        self.weightFunction = None
        self.filterFunction = filterFunction
        self.n = X.getSize()
        self.d = X.getDimension()
        self.PValuesAsymptotic = ot.Point()
        self.HSIC_XY = ot.Point()
        self.R2HSICIndices = ot.Point()
        self.setPermutationBootstrapSize(1000)

        self.Y = self.filterFunction.function(self.Y)