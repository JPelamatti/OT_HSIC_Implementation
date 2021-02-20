#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:36:28 2021

@author: Julien PELAMATTI, Joseph MURÉ
"""

from numpy import trace, diag, ones, eye


class HSICStat:
    """
    All HSIC stats inherit from this class.

    """

    def __init__(self):
        raise NotImplementedError("Use an actual statistic to compute HSIC indices.")

    def _computeHSICIndex(self, V1, V2, Cov1, Cov2, W):
        pass

    def _isCSACompatible(self):
        return False

    def getStatLetter(self):
        """Return the letter of the statistic used as HSIC estimator."""
        return self._stat

    def _computePValue(self, gamma, n, HSIC_obs, mHSIC):
        return gamma.computeComplementaryCDF(HSIC_obs * n + mHSIC * n)


class HSICvStat(HSICStat):
    """
    Compute the HSIC V-stat estimator.

    """

    def __init__(self):
        self._stat = "V"

    def _computeHSICIndex(self, V1, V2, Cov1, Cov2, W):
        n = W.shape[0]
        U = ones((n, n))
        H1 = eye(n) - 1 / n * U @ W
        H2 = eye(n) - 1 / n * W @ U

        Kv1 = Cov1.discretize(V1)

        Kv2 = Cov2.discretize(V2)

        HSIC = 1 / n ** 2 * trace(W @ Kv1 @ W @ H1 @ Kv2 @ H2)

        return HSIC

    def _isCSACompatible(self):
        return True

    def _computePValue(self, gamma, n, HSIC_obs, mHSIC):
        return gamma.computeComplementaryCDF(HSIC_obs * n)


class HSICuStat(HSICStat):
    """
    Compute the HSIC U-stat estimator.

    Note
    ----

    The U-stat is incompatible with the `CSAHSICEstimator`.
    Use `HSICvStat` with `CSAHSICEstimator`.

    """

    def __init__(self):
        self._stat = "U"

    def _computeHSICIndex(self, V1, V2, Cov1, Cov2, W):

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

        n = W.shape[0]
        Kv1 = Cov1.discretize(V1)
        Kv1_ = Kv1 - diag(diag(Kv1))
        Kv2 = Cov2.discretize(V2)
        Kv2_ = Kv2 - diag(diag(Kv1))
        One = ones((n, 1))

        HSIC = (
            1
            / n
            / (n - 3)
            * (
                trace(Kv1_ @ Kv2_)
                - 2 / (n - 2) * One.T @ Kv1_ @ Kv2_ @ One
                + One.T @ Kv1_ @ One * One.T @ Kv2_ @ One / (n - 1) / (n - 2)
            )
        )

        return HSIC[0, 0]
