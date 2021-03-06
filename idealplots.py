import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def bestfit2(x, mag, n, A): #arbitrary power law times gaussian at 0
    return mag*x**n*np.exp(-A*x**2)

def bestfit3(x, mag, n, A, B): #arbitrary power law times exponential sq & cu
    return mag*x**n*np.exp(-A*x**2-B*x**3)

def bestfit4(x, mag, n, A, B): #arbitrary power law times exponential sq & qu
    return mag*x**n*np.exp(-A*x**2-B*x**4)

def goodness(xdata, ydata, res): #Goodness-of-fit function, lower result = better
    sqintegral = 0
    sqresint = 0
    for i in range(len(xdata)-1):
        sqintegral += ((xdata[i+1]-xdata[i])*ydata[i])**2/ydata[i]
        sqresint += ((xdata[i+1]-xdata[i])*res[i])**2/ydata[i]
    return sqresint/sqintegral

#Collecting data from file
data = np.genfromtxt('output/13dec/nodist/mcnodist', comments = ';', skip_header=19)
xdatauncut = data[:,0]
ydatauncut = data[:,1]

#cutting out unimportant data
xdata = []
ydata = []
ymax = np.amax(ydatauncut)
for i in range(len(xdatauncut)):
    if ydatauncut[i] >= ymax*1e-2:
        xdata = np.append(xdata, [xdatauncut[i]])
        ydata = np.append(ydata, [ydatauncut[i]])

#normalising data
ydata = ydata/ymax
xdata = xdata/30

#plotting actual data
plt.subplot(2, 1, 1)
plt.loglog(xdata, ydata, 'b-', label='Emission curve')
plt.ylabel('Relative emissivity density')
plt.title('Molecular Clouds idealised case')#Change this for each plot

sp0 = [5e-2, 4, 2e-3]

try: #fitting function to data
    popt2, pcov2= curve_fit(bestfit2, xdata, ydata,
                             p0=sp0, maxfev=100000,
                             method='lm')
    
except Exception as e: #if data doesn't fit, plot initial guess instead
    plt.plot(xdata, bestfit2(xdata, *sp0), 'g-')#Green means something went wrong
    print(e)

else: #plotting properly fitted data
    plt.plot(xdata, bestfit2(xdata, *popt2), 'r-',
             label='Fit function: mag=%5.3e, n=%5.3f, A=%5.3f' % tuple(popt2))
    plt.legend()
    
    #plotting subplot with residual
    res = ydata-bestfit2(xdata, *popt2)
    plt.subplot(2, 1, 2)
    plt.plot(xdata, res)
    plt.plot(xdata, np.zeros(len(xdata)), 'k--')
    plt.xlabel('Frequency/30GHz')
    plt.ylabel('Residual')
    plt.xscale("log")
    print('Goodness of fit = '+ str(goodness(xdata, ydata, res)))

finally:
    plt.show()
