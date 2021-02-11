import numpy as np

#### Weight function class for conditional and target sensitivity analysis
class HSICSAWeightFunction:

    def __init__(self, C, weightFunctionParameters = None):
        self.C = C
        self.weightFunctionParameters = weightFunctionParameters
        
    def _distanceFromC(self,y):
        for c in self.C:
            if y >= c[0] and y <= c[1]:
                d = 0.
               
        dmin = np.inf
        if d != 0.:
            for c in self.C:
                if np.min([np.abs(y-c[0]),np.abs(y-c[1])]) < dmin:
                    dmin = np.min([np.abs(y-c[0]),np.abs(y-c[1])])
                    
        return dmin

            
class HSICSAExponentialWeightFunction(HSICSAWeightFunction):

    def function(self,y):
        return np.exp(-self._distanceFromC(y) /self.weightFunctionParameters[0])


class HSICSAStepWeightFunction(HSICSAWeightFunction):
    
    def function(self,y):
        distanceFromC = self._distanceFromC(y)
        if distanceFromC == 0:
            return 1.
        elif distanceFromC > 0:
            return 0.
        else:
            raise ValueError('Error while computing distance between y and the critical domain')