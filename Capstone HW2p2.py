import numpy as np
from numpy import linalg
import math as m
#import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from scipy.optimize import minimize
#import statsmodels.api as sm
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d
#from scipy.stats import linregress

#Setting names/variables for each column of the data
t, x1, x2, x3 = [], [], [], []

#Creating the array with a loop
for line in open(r"C:\Users\Kevin\OneDrive - Northeastern University\Senior Spring LAST SEM\CAPSTONE\Misc\hw2_p2.txt", 'r'):
    values = [float(s) for s in line.split()] 
    t.append(values[0])
    x1.append(values[1])
    x2.append(values[2])
    x3.append(values[3])

x = np.stack((x1,x2,x3), axis=1)
x_sum = np.zeros([len(t),1], dtype= float)

#Fit the data to  𝑏(𝑡) = 𝑏_𝑒𝑞 − [𝑏_𝑒𝑞 − 0.355]𝑒^−𝑘𝑡
xm = np.mean(x, axis=1)
xSEM = np.std(x, axis=1)/m.sqrt(25)

tarr = np.array(t)
#Minimum χ2 (χ2min). Sum of (xm - the function)/ xSEM^2
def chisquared(vars):
    return sum(((xm[0:]-(vars[0] - (vars[0]-0.355)*np.power(m.e,-1*vars[1]*tarr[0:])))/(xSEM[0:]))**2)

minchi = minimize(chisquared, np.array([1,1]), method='nelder-mead')

#Reduced χ2, dividing χ2 by the length of x minus number of paramters (b_eq,k)
reducedchi = minchi.fun / (25-2)

 
#Find the uncertainties in the parameters using the χ2min +1 method in two dimensions




#Plot the data with error bars. Show the fitted curve. Comment on the goodness of the fit. 


