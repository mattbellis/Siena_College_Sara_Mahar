import numpy as np
import matplotlib.pylab as plt
import math as math
from iminuit import Minuit, describe, Struct



Nstudents = 200
fake_mean = 80
fake_width = 5
scores = np.random.normal(fake_mean,fake_width,Nstudents)

# Here's a very simple plot of your data
#plt.figure()
#plt.hist(scores,bins=25)#,range=(60,100))

def Gaussian(mean,x,width):
        y = (1.0/(width*np.sqrt(2*np.pi)))*(np.exp(-(x-mean)**2/(2*(width**2))))
        return -np.log(y).sum()


m=Minuit(Gaussian, mean=70, error_mean=0.1, limit_mean=(-20.,20.), width= 1, fix_width=True, x=scores, fix_x=True, errordef = 0.5)


m.print_param()

m.migrad()

