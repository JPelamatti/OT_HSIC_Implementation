import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from HSICStat import HSICvStat


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
        self._alreadyComputedPValuesAsymptotic = False
        self.setPermutationBootstrapSize(10000)

    def getStatLetter(self):
        """Return the letter of the statistic used as HSIC estimator."""
        return self.HSICstat.getStatLetter()

    def _computeWeightMatrix(self, Y):

        W = np.zeros((self.n, self.n))

        w = np.empty(self.n)
        for i in range(self.n):
            w[i] = self.weightFunction.function(Y[i])
        w = w / np.mean(w)
        np.fill_diagonal(W, w)

        return W

    def computeHSICIndex(self, V1, V2, Cov1, Cov2, W):
        return self.HSICstat._computeHSICIndex(V1, V2, Cov1, Cov2, W)

    def computeIndices(self):
        W = self._computeWeightMatrix(self.Y)

        self.HSIC_XY = []
        self.HSIC_XX = []

        for dim in range(self.d):

            self.HSIC_XY.append(
                self.computeHSICIndex(
                    self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W
                )
            )
            self.HSIC_XX.append(
                self.computeHSICIndex(
                    self.X[:, dim], self.X[:, dim], self.CovX[dim], self.CovX[dim], W
                )
            )

        self.HSIC_YY = self.computeHSICIndex(self.Y, self.Y, self.CovY, self.CovY, W)

        self.R2HSICIndices = []
        for dim in range(self.d):
            self.R2HSICIndices.append(
                self.HSIC_XY[dim] / np.sqrt(self.HSIC_XX[dim] * self.HSIC_YY)
            )

        return 0

    def setPermutationBootstrapSize(self, B):
        self._alreadyComputedPValuesPermutation = False
        self.PermutationBootstrapSize = B

    def getPermutationBootstrapSize(self):
        return self.PermutationBootstrapSize

    def _computePValuesPermutation(self):
        if self.weightFunction == None:  # GSA case
            W_obs = np.eye(self.n)
        else:
            W_obs = self._computeWeightMatrix(self.Y)
        self.PValuesPermutation = []

        # permutations = list[ot.KPermutations(self.n).generate()] ###Does not work, too computationaly intensive
        # perm_selected = random.sample(permutations,self.PermutationBootstrapSize)

        for dim in range(self.d):
            HSIC_obs = self.computeHSICIndex(
                self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W_obs
            )
            HSIC_l = []
            for b in range(self.PermutationBootstrapSize):
                Y_p = np.random.permutation(self.Y)
                # Y_p = self.Y[[perm_selected[b]]]

                if self.weightFunction == None:  # GSA case
                    W = np.eye(self.n)
                else:
                    W = self._computeWeightMatrix(Y_p)

                HSIC_l.append(
                    self.computeHSICIndex(
                        self.X[:, dim], Y_p, self.CovX[dim], self.CovY, W
                    )
                )

            p = np.count_nonzero(np.array(HSIC_l) > HSIC_obs) / (
                self.PermutationBootstrapSize + 1
            )

            self.PValuesPermutation.append(p)
        self._alreadyComputedPValuesPermutation = True
        return 0

    def _computePValuesAsymptotic(self):
        raise ValueError(
            "Asymptotic p-value estimator not available for CSA, a permutation-based estimator must be used"
        )

    def getHSICIndices(self):
        return self.HSIC_XY

    def getR2HSICIIndices(self):
        return self.R2HSICIndices

    def getPValuesPermutation(self):
        if not self._alreadyComputedPValuesPermutation:
            self._computePValuesPermutation()
        return ot.Point(self.PValuesPermutation)

    def getPValuesAsymptotic(self):
        if not self._alreadyComputedPValuesAsymptotic:
            self._computePValuesAsymptotic()
        return ot.Point(self.PValuesAsymptotic)

    def drawHSICIndices(self):
        plt.figure()
        plt.plot(np.arange(1, self.d + 1), self.HSIC_XY, "*")
        plt.xticks(np.arange(1, self.d + 1))
        plt.xlabel("Variable index")
        plt.ylabel("HSIC indices")

        return 0

    def drawR2HSICIIndices(self):
        plt.figure()
        plt.plot(np.arange(1, self.d + 1), self.R2HSICIndices, "*")
        plt.xticks(np.arange(1, self.d + 1))
        plt.xlabel("Variable index")
        plt.ylabel("R2-HSIC indices")

        return 0

    def drawPValuesPermutation(self):
        plt.figure()
        plt.plot(np.arange(1, self.d + 1), self.getPValuesPermutation(), "*")
        plt.xticks(np.arange(1, self.d + 1))
        plt.xlabel("Variable index")
        plt.ylabel("p-values")

        return 0

    def drawPValuesAsymptotic(self):
        plt.figure()
        plt.plot(np.arange(1, self.d + 1), self.getPValuesAsymptotic(), "*")
        plt.xticks(np.arange(1, self.d + 1))
        plt.xlabel("Variable index")
        plt.ylabel("p-values")

        return 0


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
        self._alreadyComputedPValuesAsymptotic = False
        self.setPermutationBootstrapSize(10000)

    def _computeWeightMatrix(self, Y):

        W = np.eye(self.n)

        return W

    def _computePValuesAsymptotic(self):
        W = np.eye(self.n)

        self.PValuesAsymptotic = []

        H = np.eye(self.n) - 1 / self.n * np.ones((self.n, self.n))
        Ky = self.CovY.discretize(self.Y)
        Ey = 1 / self.n / (self.n - 1) * np.sum(Ky - np.diag(np.diag(Ky)))
        By = H @ Ky @ H

        for dim in range(self.d):
            HSIC_obs = self.computeHSICIndex(
                self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W
            )

            Kx = self.CovX[dim].discretize(self.X[:, dim])

            Ex = 1 / self.n / (self.n - 1) * np.sum(Kx - np.diag(np.diag(Kx)))

            Bx = H @ Kx @ H
            B = np.multiply(Bx, By)
            B = B ** 2

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
                @ (B - np.diag(np.diag(B)))
                @ np.ones((self.n, 1))
                / self.n
                / (self.n - 1)
            )
            varHSIC = varHSIC[0, 0]

            alpha = mHSIC ** 2 / varHSIC
            beta = self.n * varHSIC / mHSIC

            gamma = ot.Gamma(alpha, 1 / beta)
            p = self.HSICstat._computePValue(gamma, self.n, HSIC_obs, mHSIC)

            self.PValuesAsymptotic.append(p)
        self._alreadyComputedPValuesAsymptotic = True
        return 0


class TSAHSICEstimator(GSAHSICEstimator):
    """
    Target sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, weightFunction, HSICstat=HSICvStat()):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICstat = HSICstat
        self.weightFunction = weightFunction
        self.n = X.getSize()
        self.d = X.getDimension()
        self._alreadyComputedPValuesAsymptotic = False
        self.setPermutationBootstrapSize(10000)

        for i in range(self.n):
            self.Y[i] = [self.weightFunction.function(self.Y[i])]
