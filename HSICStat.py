#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:36:28 2021

@author: Julien PELAMATTI, Joseph MURÉ
"""

from numpy import trace, ones, eye, fill_diagonal, array


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

        n = W.shape[0]
        Kv1 = array(Cov1.discretize(V1))
        fill_diagonal(Kv1,0)
        Kv2 = array(Cov2.discretize(V2))
        fill_diagonal(Kv2,0)
        One = ones((n, 1))

        HSIC = (
            1
            / n
            / (n - 3)
            * (
                trace(Kv1 @ Kv2)
                - 2 / (n - 2) * One.T @ Kv1 @ Kv2 @ One
                + One.T @ Kv1 @ One * One.T @ Kv2 @ One / (n - 1) / (n - 2)
            )
        )

        return HSIC[0, 0]
