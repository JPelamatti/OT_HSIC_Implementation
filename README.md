# Implementation prototype for the HSIC sensitivity indices on the OpenTURNS platform

The following code was developed by EDF R&D based on publicly available scientific literature.

Dependencies :
  - `openturns 1.17`
  - `numpy 1.20.2`

Files :
  - `HSICEstimators.py` : contains the classes associated to the HSIC estimators
  - `HSICSAWeightFunctions.py` : contains the weight function class
  - `HSICStat.py` : contains the statistics class, providing the different estimators expressions
  - `UseCase_GSA.py` : contains a simple example of Global sensitivity analysis on the Ishigami function 
  - `Validation1.py` : contains several working examples of sensitivity analysis on the Ishigami function
  - `Validation2.py` : contains several working examples of sensitivity analysis on the minX1X2 function
  - `sample.csv` : contains the data used by UseCase_GSA.py
