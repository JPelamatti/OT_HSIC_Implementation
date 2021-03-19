rm(list = ls())
library(R.utils)
library(sensitivity)
# source('/home/f37280/Documents/Openturns/CDC HSIC/Codes/sensitivity-master/R/sensiHSIC.R')
# source('/home/f37280/Documents/Openturns/CDC HSIC/Codes/sensitivity-master/R/base.R')
# source('/home/f37280/Documents/Openturns/CDC HSIC/Codes/sensitivity-master/R/weightTSA.R')

#Import des donnees
Data = read.csv("sample.csv", sep = ',')  # read csv file 
namev = names(Data)
n = dim(Data)[1]
d = dim(Data)[2]

y = Data[dim(Data)[2]][,1]
X = Data[,1:(dim(Data)[2]-1)]

kernelX = c()
for(i in 1:dim(X)[2])
{
  kernelX = c(kernelX,'rbf')
}

# x <- sensiHSIC(model = NULL, X = X, kernelX = kernelX, kernelY = 'rbf', estimator.type = 'V-stat', test.method = "Permutation", B = 10000)
x <- sensiHSIC(model = NULL, X = X, kernelX = kernelX, kernelY = 'rbf', estimator.type = 'U-stat', test.method = "Asymptotic")
# x <- sensiHSIC(model = NULL, X = X, target =  list(c = 5., upper = TRUE,type = "exp1side", param = 0.5), kernelX = kernelX, kernelY = 'rbf', estimator.type = 'V-stat', test.method = "Asymptotic")
# x <- sensiHSIC(model = NULL, X = X, target =  list(c = 5., upper = TRUE,type = "indicTh", param = 0.5), kernelX = kernelX, kernelY = 'categ', estimator.type = 'U-stat', test.method = "Permutation", B = 10000)
# x <- sensiHSIC(model = NULL, X = X, target =  list(c = 5., upper = TRUE,type = "indicTh", param = 0.5), kernelX = kernelX, kernelY = 'categ', estimator.type = 'U-stat', test.method = "Asymptotic")
# x <- sensiHSIC(model = NULL, X = X, kernelX = kernelX, kernelY = 'rbf',cond = list(c = 5., upper = TRUE,type = "exp1side", param = 0.5), estimator.type = 'V-stat', test.method = "Permutation", B = 1000)

               
R = tell(x,y)
print(R$S[[1]])
print(R$HSICXY[[1]])
print(R$Pvalue[[1]])