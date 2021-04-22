import openturns as ot

class KroneckerKernel(ot.StationaryFunctionalCovarianceModel):
    def __init__(self):
        rho = ot.PythonFunction(1,1,self.rho)
        super().__init__([1.0], [1.0], rho)
        self.setActiveParameter([])

    def rho(self,t):
        return [int(t[0]==0.)]

