import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

ot_HSICEstimator_Vstat = 1
ot_HSICEstimator_Ustat = 2
ot_HSICEstimator_AsymptoticPValuesEstimator = 1
ot_HSICEstimator_PermutationPValuesEstimator = 2


class CSAHSICEstimator:
    """
    Conditional sensitivity analysis estimator, only V-stat estimator and asymptotic p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, HSICEstimatorType, weightFunction):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICEstimatorType = HSICEstimatorType
        self.weightFunction = weightFunction
        self.PValueEstimatorType = None
        self.n = X.getSize()
        self.d = X.getDimension()

    def computeGramMatrix(self, sample, Cov):
        m = ot.Mesh(sample)
        K = ot.CovarianceMatrix(Cov.discretize(m))

        return K

    def _computeWeightMatrix(self, Y):

        W = np.zeros((self.n, self.n))

        w = np.empty(self.n)
        for i in range(self.n):
            w[i] = self.weightFunction.function(Y[i])
        w = w / np.mean(w)
        np.fill_diagonal(W, w)

        return W

    def computeHSICIndex(self, V1, V2, Cov1, Cov2, W):
        if self.HSICEstimatorType == ot_HSICEstimator_Vstat:
            return self._VStatEstimator(V1, V2, Cov1, Cov2, W)
        elif self.HSICEstimatorType == ot_HSICEstimator_Ustat:
            return self._UStatEstimator(V1, V2, Cov1, Cov2, W)
        else:
            raise ValueError("undefined estimator type")

    def _VStatEstimator(self, V1, V2, Cov1, Cov2, W):

        U = np.ones((self.n, self.n))
        H1 = np.eye(self.n) - 1 / self.n * U @ W
        H2 = np.eye(self.n) - 1 / self.n * W @ U

        Kv1 = self.computeGramMatrix(V1, Cov1)

        Kv2 = self.computeGramMatrix(V2, Cov2)

        HSIC = 1 / self.n ** 2 * np.trace(W @ Kv1 @ W @ H1 @ Kv2 @ H2)

        return HSIC

    def _UStatEstimator(self, V1, V2, Cov1, Cov2, W):
        raise ValueError("U-stat estimator not available for CSA, V-stat must be used")

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

    def parameterizePValueEstimator(self, PValueEstimatorType, B=None):
        self.PValueEstimatorType = PValueEstimatorType
        self.PermutationBootstrapSize = (
            B
        )  # If PValueEstimatorType == ot.HSICEstimator.PermutationPValuesEstimator

    def computePValues(self):
        if self.PValueEstimatorType == ot_HSICEstimator_PermutationPValuesEstimator:
            self._computePValuesPermutation()
        elif self.PValueEstimatorType == ot_HSICEstimator_AsymptoticPValuesEstimator:
            self._computePValuesAsymptotic()
        else:
            raise ValueError("Invalid p-value estimator type")
        return 0

    def _computePValuesPermutation(self):
        if self.weightFunction == None:  # GSA case
            W_obs = np.eye(self.n)
        else:
            W_obs = self._computeWeightMatrix(self.Y)
        self.PValues = []

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

            self.PValues.append(p)
        return 0

    def _computePValuesAsymptotic(self):
        raise ValueError(
            "Asymptotic p-value estimator not available for CSA, a permutation-based estimator must be used"
        )

    def getHSICIndices(self):
        return self.HSIC_XY

    def getR2HSICIIndices(self):
        return self.R2HSICIndices

    def getPValues(self):
        return self.PValues

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

    def drawPValues(self):
        plt.figure()
        plt.plot(np.arange(1, self.d + 1), self.PValues, "*")
        plt.xticks(np.arange(1, self.d + 1))
        plt.xlabel("Variable index")
        plt.ylabel("p-values")

        return 0


class GSAHSICEstimator(CSAHSICEstimator):
    """
    Global sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, HSICEstimatorType):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICEstimatorType = HSICEstimatorType
        self.weightFunction = None
        self.PValueEstimatorType = None
        self.n = X.getSize()
        self.d = X.getDimension()

    def _computeWeightMatrix(self, Y):

        W = np.eye(self.n)

        return W

    def _UStatEstimator(self, V1, V2, Cov1, Cov2, W=None):

        # W is a mute parameters which allows to call in the same fashion both estimators
        # Kv1 = self.computeGramMatrix(V1,Cov1)
        # Kv2 = self.computeGramMatrix(V2,Cov2)

        # HSIC = 0
        # for i in range(n):
        #     for j in range(n):
        #         Aij = Kv1[i,j] - np.mean(Kv1[i,:]) - np.mean(Kv1[:,j])  + np.mean(Kv1)
        #         Bij = Kv2[i,j] - np.mean(Kv2[i,:]) - np.mean(Kv2[:,j])  + np.mean(Kv2)
        #         HSIC += Aij*Bij
        # HSIC = 1/n**2*HSIC

        Kv1 = self.computeGramMatrix(V1, Cov1)
        Kv1_ = Kv1 - np.diag(np.diag(Kv1))
        Kv2 = self.computeGramMatrix(V2, Cov2)
        Kv2_ = Kv2 - np.diag(np.diag(Kv1))
        One = np.ones((self.n, 1))

        HSIC = (
            1
            / self.n
            / (self.n - 3)
            * (
                np.trace(Kv1_ @ Kv2_)
                - 2 / (self.n - 2) * One.T @ Kv1_ @ Kv2_ @ One
                + One.T @ Kv1_ @ One * One.T @ Kv2_ @ One / (self.n - 1) / (self.n - 2)
            )
        )

        return HSIC[0, 0]

    def _computePValuesAsymptotic(self):
        W = np.eye(self.n)

        self.PValues = []

        H = np.eye(self.n) - 1 / self.n * np.ones((self.n, self.n))
        Ky = self.computeGramMatrix(self.Y, self.CovY)
        Ey = 1 / self.n / (self.n - 1) * np.sum(Ky - np.diag(np.diag(Ky)))
        By = H @ Ky @ H

        for dim in range(self.d):
            HSIC_obs = self.computeHSICIndex(
                self.X[:, dim], self.Y, self.CovX[dim], self.CovY, W
            )

            Kx = self.computeGramMatrix(self.X[:, dim], self.CovX[dim])

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

            Gamma = ot.Gamma(alpha, 1 / beta)

            if self.HSICEstimatorType == ot_HSICEstimator_Vstat:
                p = Gamma.computeComplementaryCDF(HSIC_obs * self.n)
            elif self.HSICEstimatorType == ot_HSICEstimator_Ustat:
                p = Gamma.computeComplementaryCDF(
                    HSIC_obs * self.n + mHSIC * self.n
                )  # Why?!
            else:
                raise ValueError(
                    "Unknown estimator type for asymptotic p-value estimation"
                )

            self.PValues.append(p)
        return 0


class TSAHSICEstimator(GSAHSICEstimator):
    """
    Target sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation

    """

    def __init__(self, CovarianceList, X, Y, HSICEstimatorType, weightFunction):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X
        self.Y = Y
        self.HSICEstimatorType = HSICEstimatorType
        self.weightFunction = weightFunction
        self.PValueEstimatorType = None
        self.n = X.getSize()
        self.d = X.getDimension()

        for i in range(self.n):
            self.Y[i] = [self.weightFunction.function(self.Y[i])]
