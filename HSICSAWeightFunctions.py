import numpy as np
import openturns as ot

#%%
# Auxiliary functions
#%% 
class FunctionWrapper(ot.OpenTURNSPythonFunction):

    def __init__(self,C, pythonfunction, weightFunctionParameters = None):
        super(FunctionWrapper, self).__init__(1, 1)
        self.C = C
        self.pythonfunction = pythonfunction
        self.weightFunctionParameters = weightFunctionParameters
        
    def _exec(self, y):
        return [self.pythonfunction(y,self.C,self.weightFunctionParameters)]
    

def _distanceFromC(y, C):
    for c in C:
        if y[0] >= c[0] and y[0] <= c[1]:
            dmin = 0.0
            return dmin

    dmin = np.inf

    for c in C:
        if np.min([np.abs(y[0] - c[0]), np.abs(y[0] - c[1])]) < dmin:
            dmin = np.min([np.abs(y[0] - c[0]), np.abs(y[0] - c[1])])

    return dmin

def _stepWeightfunction(y,C, weightFunctionParameters = None):
    
     distanceFromC = _distanceFromC(y,C)
     if distanceFromC == 0:
         return 1.0
     elif distanceFromC > 0:
         return 0.0
     else:
         raise ValueError(
             "Error while computing distance between y and the critical domain"
     )
              
def _exponentialWeightfunction(y,C,weightFunctionParameters):
    return np.exp(
        -_distanceFromC(y,C)
        / (
            weightFunctionParameters[0]
            * weightFunctionParameters[1]
            / 5.0
        )
    )
    
#%%
#  Weight function classes for conditional and target sensitivity analysis
#%%
class HSICSAWeightFunction:
    def __init__(self, C):
        self.C = C

    def setWeightFunction(self,function,weightFunctionParameters = None):
        self.function = FunctionWrapper(self.C,function, weightFunctionParameters)

class HSICSAExponentialWeightFunction(HSICSAWeightFunction):
    def __init__(self,C,weightFunctionParameters):
        super().__init__(C)
        super().setWeightFunction(_exponentialWeightfunction,weightFunctionParameters)


class HSICSAStepWeightFunction(HSICSAWeightFunction):
    def __init__(self,C):
        super().__init__(C)
        super().setWeightFunction(_stepWeightfunction)


