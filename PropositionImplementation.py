import numpy as np
import openturns as ot
import matplotlib.pyplot as plt


####Conditional sensitivity analysis estimator, only V-stat estimator and asymptotic p-value estimation
class CSAHSICEstimator:

    def __init__(self, CovarianceList, X, Y, HSICEstimatorType, weightFunction):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X 
        self.Y = Y 
        self.HSICEstimatorType = HSICEstimatorType
        self.weightFunction = weightFunction
        self.PValueEstimatorType = None


    def computeGramMatrix(self,sample,Cov):
        n = sample.size
        K = ot.CovarianceMatrix(n)
        for i in range(n):
            for j in range(i+1,n):
                K[i,j] = Cov(sample[i],sample[j])
        K = K+K.T
        for i in range(n):
            K[i,i] = Cov(sample[i],sample[i]) #Can be made more efficient in case we are sure to have a stationary covariance function
        return K
    
    
    def _computeWeightMatrix(self,Y):
        n = len(Y)
        
        if self.weightFunction == None: #GSA case
            W = np.eye(n)
        else:
            W = np.zeros((n,n))
    
            w = np.empty(n)
            for i in range(n):
                w[i] = self.weightFunction.function(Y[i])
            w = w/np.mean(w)
            np.fill_diagonal(W, w)
            
        return W
    
    
    def z(self, V1, V2, Cov1, Cov2, W):
        if self.HSICEstimatorType == ot.HSICEstimator.Vstat:
            return self._VStatEstimator(V1, V2, Cov1, Cov2, W)
        elif self.HSICEstimatorType == ot.HSICEstimator.Ustat:
            return self._UStatEstimator(V1, V2, Cov1, Cov2, W)
        else:
            raise ValueError('undefined estimator type')
        
        
    def _VStatEstimator(self, V1, V2, Cov1, Cov2, W): 
        n = len(V1)
            
        U = np.ones((n,n))
        H1 = np.eye(n) - 1/n*U@W
        H2 = np.eye(n) - 1/n*W@U
        
        Kv1 = self.computeGramMatrix(V1,Cov1)

        Kv2 = self.computeGramMatrix(V2,Cov2)
        
        HSIC = 1/n**2*np.trace(W @ Kv1 @ W @ H1 @ Kv2 @ H2)

        return HSIC
    
          
    def _UStatEstimator(self, V1, V2, Cov1, Cov2, W): 
        raise ValueError('U-stat estimator not available for CSA, V-stat must be used')
        
    

    def computeIndices(self):
        d = self.X.shape[1] #NB, does not work if X is one-dimensional

        W = self._computeWeightMatrix(self.Y)
            
        self.HSIC_XY = []
        self.HSIC_XX = []
        
        for dim in range(d):
            
            self.HSIC_XY.append( self.computeHSICIndex(self, self.X[:,d], self.Y, self.CovX[dim], self.CovY, W))
            self.HSIC_XX.append( self.computeHSICIndex(self, self.X[:,d], self.X[:,d], self.CovX[dim], self.CovX[dim], W))
            
        self.HSIC_YY.append( self.computeHSICIndex(self, self.Y, self.Y, self.CovY, self.CovY, W))
        
        self.R2HSICIndices =[]
        for dim in range(d):
            self.R2HSICIndices.append(self.HSIC_XY/np.sqrt(self.HSIC_XX*self.HSIC_YY))

        return 0
        
        
    def parameterizePValueEstimator(self,PValueEstimatorType, B = None):
        self.PValueEstimatorType = PValueEstimatorType
        self.PermutationBootstrapSize = B #If PValueEstimatorType == ot.HSICEstimator.PermutationPValuesEstimator
        
        
    def computePValues(self):
        if self.PValueEstimatorType == ot.HSICEstimator.AsymptoticPValuesEstimator:
            self._computePValuesPermutation()
        elif self.PValueEstimatorType == ot.HSICEstimator.PermutationPValuesEstimator:
            self._computePValuesAsymptotic()
        else:
            raise ValueError('Invalid p-value estimator type')
        return 0

        
    def _computePValuesPermutation(self):
        n = self.X.shape[0] #NB, does not work if X is one-dimensional
        d = self.X.shape[1] #NB, does not work if X is one-dimensional
        
        if self.weightFunction == None: #GSA case
            W_obs = np.eye(n)
        else:
            W_obs = self._computeWeightMatrix(self.Y)
        self.PValues = []
        
        for dim in d:
            HSIC_obs = self.computeHSICIndex(self.X[:,dim],self.Y, self.CovX[dim], self.CovY, W_obs) 
            HSIC_l = [HSIC_obs]
            for b in range(self.PermutationBootstrapSize):
                Y_p = np.random.permutation(self.Y)
                
                if self.weightFunction == None: #GSA case
                    W = np.eye(n)
                else:
                    W = self._computeWeightMatrix(Y_p)
                    
                HSIC_l.append(self.computeHSICIndex(self.X[:,dim],Y_p, self.CovX[dim], self.CovY, W)) 
            
            p = 0
            for index in HSIC_l:
                if index > HSIC_obs:
                    p += 1
            p = p/(self.PermutationBootstrapSize+1)
            
            self.PValues.append(p)
        return 0
        
    
    def _computePValuesAsymptotic(self):
        raise ValueError('Asymptotic p-value estimator not available for CSA, a permutation-based estimator must be used')


    def getHSICIndices(self):
        return self.HSIC_XY
        
    
    def getR2HSICIIndices(self):
        return self.R2HSICIndices


    def getPValues(self):
        return self.PValues


    def drawHSICIndices(self):
        d = len(self.HSIC_XY)
        plt.figure()
        plt.plot(np.arange(1,d+1),self.HSIC_XY,'*')
        plt.xticks(np.arange(1,d+1))
        plt.xlabel('Variable index')
        plt.ylabel('HSIC indices')
        
        return 0
    
    def drawR2HSICIIndices(self):
        d = len(self.R2HSICIndices)
        plt.figure()
        plt.plot(np.arange(1,d+1),self.R2HSICIndices,'*')
        plt.xticks(np.arange(1,d+1))
        plt.xlabel('Variable index')
        plt.ylabel('R2-HSIC indices')
        
        return 0

    def drawPValues(self):
        d = len(self.PValues)
        plt.figure()
        plt.plot(np.arange(1,d+1),self.PValues,'*')
        plt.xticks(np.arange(1,d+1))
        plt.xlabel('Variable index')
        plt.ylabel('p-values')
        
        return 0

####Global sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation

class GSAHSICEstimator(CSAHSICEstimator):
    
    def __init__(self, CovarianceList, X, Y, HSICEstimatorType):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X 
        self.Y = Y
        self.HSICEstimatorType = HSICEstimatorType
        self.weightFunction = None
        self.PValueEstimatorType = None


    def UStatEstimator(self, V1, V2, Cov1, Cov2, W = None): #W is a mute parameters which allows to call in the same fashion both estimators
        n = len(V1)
        
        Kv1 = self.computeGramMatrix(V1,Cov1)
        Kv2 = self.computeGramMatrix(V2,Cov2)
        
        HSIC = 0
        for i in range(n):
            for j in range(n):
                Aij = Kv1[i,j] - np.mean(Kv1[i,:]) - np.mean(Kv1[:,j])  + np.mean(Kv1) 
                Bij = Kv2[i,j] - np.mean(Kv2[i,:]) - np.mean(Kv2[:,j])  + np.mean(Kv2) 
                HSIC += Aij*Bij
        HSIC = 1/n**2*HSIC
        
        return HSIC
    

    def computePValuesAsymptotic(self):

        n = self.X.shape[0]
        d = self.X.shape[1] #NB, does not work if X is one-dimensional
        
        W = np.eye(n)

        self.PValues = []
        
        Ky = self.computeGramMatrix(self.Y,self.CovY)

        
        for dim in d:
            HSIC_obs = self.computeHSICIndex(self.X[:,dim],self.Y, W)
            Kx = self.computeGramMatrix(self.X[:,dim],self.CovX[dim])

            Ex = 1/n/(n-1)*np.sum(np.sum(Kx - np.diag(np.diag(Kx))))
            Ey = 1/n/(n-1)*np.sum(np.sum(Ky - np.diag(np.diag(Ky))))

            H = np.eye(n) - 1/n*np.ones((n,n))

            B = np.multiply(H @ Kx @ H, H @ Ky @ H)
            B = B**2
            
            V = 2*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3)*np.ones((1,n)) @ (B - np.diag(np.diag(B))) @ np.ones((n,1))
            
            gamma = 1/n**2*(1+Ex*Ey-Ex-Ey)**2/V
            beta = n**2*V/(1+Ex*Ey-Ex-Ey)
            Gamma = ot.Gamma(gamma,beta)
            p = Gamma.computeComplementaryCDF(HSIC_obs)

            self.PValues.append(p)
        return 0
    
####Target sensitivity analysis estimator, both V-stat and U-stat estimator as well as asymptotic and permutation p-value estimation
class TSAHSICEstimator(GSAHSICEstimator):
    
    def __init__(self, CovarianceList, X, Y, HSICEstimatorType, weightFunction):
        self.CovX = CovarianceList[0]
        self.CovY = CovarianceList[1]
        self.X = X 
        self.Y = Y 
        self.HSICEstimatorType = HSICEstimatorType
        self.weightFunction = weightFunction
        self.PValueEstimatorType = None
        
        
        n = len(self.Y)
        for i in range(n):
            self.Y[i] = self.weightFunction.function(self.Y[i])



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
        
        
