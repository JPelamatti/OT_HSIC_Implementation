rm(list = ls())
library(R.utils)
library(sensitivity)
# source('sensiHSIC.R')
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

# x <- sensiHSIC(model = NULL, X = X, kernelX = kernelX, kernelY = 'rbf', estimator.type = 'V-stat', test.method = "Permutation", B = 1000)
# x <- sensiHSIC(model = NULL, X = X, target =  list(c = 5., upper = TRUE,type = "exp1side", param = 1.), kernelX = kernelX, kernelY = 'rbf', estimator.type = 'V-stat', test.method = "Asymptotic")
x <- sensiHSIC(model = NULL, X = X, kernelX = kernelX, kernelY = 'rbf',target = list(c = 5., upper = TRUE,type = "exp1side", param = 0.5), estimator.type = 'V-stat', test.method = "Asymptotic")

               
R = tell(x,y)
print(R$S[[1]])
print(R$HSICXY[[1]])
print(R$Pvalue[[1]])

# paramX = x$paramX
# paramY = x$paramY

# asymp_test_HSIC(X,y,kernelX = "rbf", paramX =ramX, kernelY = "rbf", paramY = paramY)

param = 0.5
c = 5.
Y = y
wY <- exp( - (c-Y)*((c-Y)>0) / (param * sd(Y)/5) )
