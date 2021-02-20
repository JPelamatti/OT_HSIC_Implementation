import numpy as np

#### Weight function class for conditional and target sensitivity analysis
class HSICSAWeightFunction:
    def __init__(self, C, weightFunctionParameters=None):
        self.C = C
        self.weightFunctionParameters = weightFunctionParameters

    def _distanceFromC(self, y):
        for c in self.C:
            if y[0] >= c[0] and y[0] <= c[1]:
                dmin = 0.0
                return dmin

        dmin = np.inf

        for c in self.C:
            if np.min([np.abs(y[0] - c[0]), np.abs(y[0] - c[1])]) < dmin:
                dmin = np.min([np.abs(y[0] - c[0]), np.abs(y[0] - c[1])])

        return dmin


class HSICSAExponentialWeightFunction(HSICSAWeightFunction):
    def function(self, y):
        return np.exp(
            -self._distanceFromC(y)
            / (
                self.weightFunctionParameters[0]
                * self.weightFunctionParameters[1]
                / 5.0
            )
        )


class HSICSAStepWeightFunction(HSICSAWeightFunction):
    def function(self, y):
        distanceFromC = self._distanceFromC(y)
        if distanceFromC == 0:
            return 1.0
        elif distanceFromC > 0:
            return 0.0
        else:
            raise ValueError(
                "Error while computing distance between y and the critical domain"
            )
