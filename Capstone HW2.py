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
x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = [], [], [], [], [], [], [], [], [], [], []

#Creating the array with a loop
for line in open(r"C:\Users\Kevin\OneDrive - Northeastern University\Senior Spring LAST SEM\CAPSTONE\Misc\hw2_p1.txt", 'r'):
    values = [float(s) for s in line.split()] 
    x.append(values[0])
    y1.append(values[1])
    y2.append(values[2])
    y3.append(values[3])
    y4.append(values[4])
    y5.append(values[5])
    y6.append(values[6])
    y7.append(values[7])
    y8.append(values[8])
    y9.append(values[9])
    y10.append(values[10])

y = np.stack((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10), axis = 1)
y_sum = np.zeros([len(x),1], dtype = float)

#-----------------------------------------------------------------------------------------------------

#a) For each x value, determine the average y value, <y>, and its uncertainty. 
#Determine the best power-law fit to the (x,<y>) data by finding the minimum 
#χ2 (χ2min). Report χ2min, χ2red (reduced χ2), A, and p. Find the A and p 
#uncertainties using the χ2min +1 method in 2D (see sect 11.5 in the book 
#and Curve Fit examples in MATLAB). 

#Average y for each x value. ym[0] is the average of x[0]
ym = np.mean(y, axis=1)

#Finding SEM as uncertainty in the average y value. ySEM[0] is the SEM for x[0]
ySEM = np.std(y, axis=1)/m.sqrt(10)

#Minimum χ2 (χ2min)
def chisquared(vars):
    return sum(((ym[0:]-(vars[0]*np.power(x[0:],vars[1])))/(ySEM[0:]))**2)

minchi = minimize(chisquared, np.array([13,1.5]), method='nelder-mead')

#Reduced χ2, dividing χ2 by the length of x minus number of paramters (A,p)
reducedchi = minchi.fun / (15-2)

#Finding Uncertainties in A,p with graphing
A = minchi.x[0]
p = minchi.x[1]

ZdA = []
Zdp = []
dA = np.arange(A - 1, A + 1, 0.01)
dp = np.arange(p - 1, p + 1, 0.01)
for dA in np.arange(A - 1, A + 1, 0.01):
        for dp in np.arange(p - 1, p + 1, 0.01):
            if sum(((ym-(dA*np.power(x,dp)))/(ySEM))**2) - minchi.fun > 0.95 and sum(((ym-(dA*np.power(x,dp)))/(ySEM))**2) - minchi.fun < 1.05:
                ZdA.append(dA)
                Zdp.append(dp)
plt.scatter(minchi.x[0],minchi.x[1],label = 'χ2 min')
plt.scatter(ZdA,Zdp,label = 'contour')
plt.legend()
plt.xlabel("A")
plt.ylabel('p')
plt.title("|1| Contour around minimum χ2")
plt.show()

#-----------------------------------------------------------------------------------------------------

#b) Linearize the data and fit it using ordinary least squares (OLS). Find A and p. 
#y=Ax^p becomes ln(y)=ln(A)+pln(x) or y= a0 + a1x
logx = np.log(x)
logym = np.log(ym)

#Need to minimize r**2 = r**T r ... r = Xa - y or plnx - lnym
ones = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
Xmatrix = np.concatenate((ones, logx[:, np.newaxis]), axis=1)
XmatrixT = Xmatrix.T
alpha = np.matmul(XmatrixT, Xmatrix)
invalpha = linalg.inv(alpha)

amatrix = np.matmul(np.matmul(invalpha,XmatrixT), logym)
Avalue = np.exp(amatrix[0])

#-----------------------------------------------------------------------------------------------------

#c) Use the weighted least squares method (WLS) (as discussed in class) to 
#fit the linearized data using an analytical function (matrix algebra). 
#Find A and p and their uncertainties. 
ySEMc = ySEM.reshape((15,1))
XmatrixWLS = Xmatrix / ySEMc
XmatrixTWLS = XmatrixWLS.T
logymWLS = np.divide(logym, ySEM)
alphaWLS = np.matmul(XmatrixTWLS, XmatrixWLS)
invalphaWLS = linalg.inv(alphaWLS)

amatrixWLS = np.matmul(np.matmul(invalphaWLS,XmatrixTWLS), logymWLS)
AvalueWLS = np.exp(amatrixWLS[0])

#Finding Uncertainties in A,p


#-----------------------------------------------------------------------------------------------------

#d) Plot the x, <y> data points with error bars on a log scale and the three fits as lines.
#Compare and discuss the results and any differences between OLS and WLS. Include a comment 
#on when WLS is most appropriate versus OLS. 

xerror = 0.005
plt.scatter(x,ym)
plt.errorbar(x,ym, ls = 'none', xerr = xerror, yerr = ySEM, label= "Exp Data")

#Part A Power Law Best Fit
#bestfitax = np.linspace(0,10,15)
bestfitay = minchi.x[0] * x**minchi.x[1]
plt.plot(x, bestfitay, label = 'Power Law')

#OLS Best Fit
#bestfitbx = np.linspace(0,10,15)
#bestfitby = amatrix[0] + np.multiply(logx,amatrix[1])
bestfitby = Avalue * x**amatrix[1]
plt.plot(x, bestfitby, label = 'OLS')

#WLS Best Fit
#bestfitcx = np.linspace(0,10,15)
#bestfitcy = AvalueWLS + np.multiply((logx/ySEM), amatrixWLS[1])
bestfitcy = AvalueWLS * x**amatrixWLS[1]
plt.plot(x, bestfitcy, label = 'WLS')

#Plot Parameters
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('x date')
plt.ylabel('<y> data')
plt.title('<y> as a function of x')
plt.show()



