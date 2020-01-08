import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import codecs

def doublepower(x, mag, n, A, B): #power law times exponential sq & qu
    return mag*x**n*np.exp(-A*x**2-B*x**4)

def cubepower(x, mag, n, A, B):
    return mag*x**n*np.exp(-A*x**2-B*x**3)

def goodnessf(xdata, ydata, res): #Goodness-of-fit function, lower result = better
    sqintegral = 0
    sqresint = 0
    for i in range(len(xdata)-1):
        sqintegral += ((xdata[i+1]-xdata[i])*ydata[i])**2/ydata[i]
        sqresint += ((xdata[i+1]-xdata[i])*res[i])**2/ydata[i]
    return sqresint/sqintegral

parameters = [] #to store the parameter
goodnessd = [] #goodness-of-fit for that parameter using doublepower
goodnessc = []


#Change below line to proper folder, and don't forget paramstr
for file in os.listdir('output/parameters/Tgrid'):
    filepath = 'output/parameters/Tgrid/'+file
    if file == '.DS_Store': #This auto-generated file was getting in the way.
        continue
    print(file)
    data = np.genfromtxt(filepath, comments = ';', skip_header=20)
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

    #extracting parameter
    paramstr = file[5:100] #first number = length of prefix
    if paramstr[0] == 'e':
        paramstr = '1' + paramstr
    if paramstr[-1] == 't':
        paramstr = paramstr[:-4]
    paramflt = float(paramstr)
    parameters = np.append(parameters, [paramflt])
    #print(paramflt)

    #############Finding goodness-of-fit: doublepower##############

    try:
        dpopt, dpcov = curve_fit(doublepower, xdata, ydata,
                                 p0=[5e-2, 4, 2e-3, 0], maxfev=100000,
                                 method='lm')

    except Exception as e: #if data doesn't fit, put '0' as the error
        goodnessd = np.append(goodnessd, [0])
        print(e)

    else: #add error to goodness-of-fit list
        res = ydata-doublepower(xdata, *dpopt)
        goodnessd = np.append(goodnessd, [goodnessf(xdata, ydata, res)])
        
    ############Finding goodness-of-fit: cubeopt#################

    try:
        cpopt, cpcov = curve_fit(cubepower, xdata, ydata,
                                 p0=[1e-3, 4, 2e-3, 0], maxfev=100000,
                                 method='lm')

    except Exception as e: #if data doesn't fit, put '0' as the error
        goodnessc = np.append(goodnessc, [0])
        print(e)

    else: #add error to goodness-of-fit list
        res = ydata-cubepower(xdata, *cpopt)
        goodnessc = np.append(goodnessc, [goodnessf(xdata, ydata, res)])


plt.plot(parameters, goodnessc, 'v', label='$J_3$ goodness-of-fit')
plt.plot(parameters, goodnessd, '<', label='$J_4$ goodness-of-fit')
plt.title("Goodness-of-fit of $J_3$ and $J_4$ across a wide range of temperatures")
plt.xlabel("Temperature in Kelvin")
plt.ylabel("Goodness-of-fit")
plt.legend()
plt.xscale("log")
plt.yscale("log")
#print(parameters)
#print(goodness)
plt.show()
