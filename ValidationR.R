rm(list = ls())
library(R.utils)
library(sensitivity)

#Import des donnees
Data = read.csv("sample.csv", sep = ',')  # read csv file 
namev = names(Data)
n = dim(Data)[1]
d = dim(Data)[2]

y = data.frame(Data[dim(Data)[2]])
X = Data[,1:(dim(Data)[2]-1)]

kernelX = c()
for(i in 1:dim(X)[2])
{
  kernelX = c(kernelX,'rbf')
}

x <- sensiHSIC(model = NULL, X = X, kernelX = kernelX, kernelY = 'rbf', estimator.type = 'U-stat')
R = tell(x,y)

# plot(R$S[[1])]
print(R$S[[1]])
print(R$HSICXY[[1]])
