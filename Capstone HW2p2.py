import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
xSEM = np.std(x, axis=1)/m.sqrt(3)

tarr = np.array(t)
#Minimum χ2 (χ2min). Sum of (xm - the function)/ xSEM^2
def chisquared(vars):
    return sum(((xm - (vars[0] - vars[0]*np.power(m.e,vars[1]*-1*tarr) + 0.355*np.power(m.e,vars[1]*-1*tarr)))/xSEM)**2)
    #return sum(((xm-(vars[0] - (vars[0]-0.355)*np.power(m.e,-1*vars[1]*tarr)))/(xSEM))**2)

minchi = minimize(chisquared, np.array([1,1]), method='nelder-mead')

#Reduced χ2, dividing χ2 by the length of x minus number of paramters (b_eq,k)
reducedchi = minchi.fun / (25-2)

 
#Find the uncertainties in the parameters using the χ2min +1 method in two dimensions
beq = minchi.x[0]
k = minchi.x[1]

Zdbeq = []
Zdk = []
dbeq = np.arange(beq - 0.01, beq + 0.01, 0.0001)
dk = np.arange(k - 0.01, k + 0.01, 0.0001)
for dbeq in np.arange(beq - 0.01, beq + 0.01, 0.0001):
        for dk in np.arange(k - 0.01, k + 0.01, 0.0001):
            if sum(((xm-(dbeq - (dbeq-0.355)*np.power(m.e,-1*dk*tarr)))/(xSEM))**2) - minchi.fun > 0.90 and sum(((xm-(dbeq - (dbeq-0.355)*np.power(m.e,-1*dk*tarr)))/(xSEM))**2) - minchi.fun < 1.10:
                Zdbeq.append(dbeq)
                Zdk.append(dk)
plt.scatter(minchi.x[0],minchi.x[1], label = 'χ2 min')
plt.scatter(Zdbeq,Zdk, label = 'contour')
plt.legend()
plt.xlabel("b_eq")
plt.ylabel('k')
plt.title("|1| Contour around minimum χ2")
plt.show()


#Plot the data with error bars. Show the fitted curve. Comment on the goodness of the fit. 
terror = 0.005
plt.scatter(t,xm)
plt.errorbar(t,xm, ls = 'none', xerr = terror, yerr = xSEM, label= "Exp Data")

#Best Fit Line from b_eq, k
bestfitx = minchi.x[0] - (minchi.x[0] - 0.355)*np.power(m.e,-1*minchi.x[1]*tarr)
plt.plot(t, bestfitx, label = 'Line of Best Fit')

#Plot Parameters
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('t data')
plt.ylabel('<x> data')
plt.title('<x> as a function of t')
plt.show()


