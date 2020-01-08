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

#Arrays to fill
situations = ['All delta\nfunctions', 'Charge distribution\nonly', 'Dipole moment\ndistribution only',
              'Shape distribution\nonly', 'Size distribution\nonly', 'All proper\ndistributions']
j2s = []
j3s = []
j4s = []

#Collecting data from files - replace with path to respective folder
files = os.listdir('output/13dec_environment/wim')
files.sort() #we keep track of the situation by taking it in alphabetical order

for file in files:
    if file == '.DS_Store':
        continue
    
    data = np.genfromtxt('output/13dec_environment/wim/'+file, comments = ';', skip_header=19)
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

    #initial guess (J_2)
    sp20 = [5e-2, 4, 2e-3]
    #initial guess (J_3, J_4)
    sp0 = [5e-2, 4, 2e-3, 0]


    ########J_2#######

    try: #fitting function to data
        popt2, pcov2= curve_fit(bestfit2, xdata, ydata,
                                 p0=sp20, maxfev=100000,
                                 method='lm')

    except Exception as e: #if data doesn't fit, throw something
        print(e)
        j2s = np.append(j2s, 1) #instantly noticeably poor if it shows up

    else: #find, print and plot goodness-of-fit
        res2 = ydata-bestfit2(xdata, *popt2)
        j2s = np.append(j2s, goodness(xdata, ydata, res2))

    ########J_3#######

    try: #fitting function to data
        popt3, pcov3= curve_fit(bestfit3, xdata, ydata,
                                 p0=sp0, maxfev=100000,
                                 method='lm')

    except Exception as e: #if data doesn't fit, throw something
        print(e)
        j3s = np.append(j3s, 1)

    else: #find, print and plot goodness-of-fit
        res3 = ydata-bestfit3(xdata, *popt3)
        j3s = np.append(j3s, goodness(xdata, ydata, res3))

    ########J_4#######

    try: #fitting function to data
        popt4, pcov4= curve_fit(bestfit4, xdata, ydata,
                                 p0=sp0, maxfev=100000,
                                 method='lm')

    except Exception as e: #if data doesn't fit, throw something
        print(e)
        j4s = np.append(j4s, 1)

    else: #find, print and plot goodness-of-fit
        res4 = ydata-bestfit4(xdata, *popt4)
        j4s = np.append(j4s, goodness(xdata, ydata, res4))

    
###########Making the plot#############

#reordering arrays to make more sense
myorder = [3, 1, 2, 4, 5, 0] #alphabetical -> order in 'situations'
j2s = [j2s[i] for i in myorder]
j3s = [j3s[i] for i in myorder]
j4s = [j4s[i] for i in myorder]
print(j2s)
print(j3s)
print(j4s)

fig, ax = plt.subplots()
x = np.arange(len(situations))
width = 0.25

rectsj2 = ax.bar(x-width, j2s, width, label='$J_2$ goodness-of-fit', log=False)
rectsj3 = ax.bar(x, j3s, width, label='$J_3$ goodness-of-fit', log=False)
rectsj4 = ax.bar(x+width, j4s, width, label='$J_4$ goodness-of-fit', log=False)

ax.set_ylabel('Goodness-of-fit (lower is better)')
#Change this for each new graph
ax.set_title('Goodness-of-fit of $J_2$, $J_3$ and $J_4$ for the WIM')
ax.set_xticks(x)
ax.set_xticklabels(situations)
ax.legend()

fig.tight_layout()

plt.show()
