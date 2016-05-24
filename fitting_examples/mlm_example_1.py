import numpy as np
import matplotlib.pylab as plt
import math as math

################################################################################
# This is how we define our own function
################################################################################
def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*(width**2)))

    return y


################################################################################
# First thing I'm going to do is to generate some fake data for us to work with.
# 
# This is going to be more like a particle physics experiment. Suppose I'm looking
# at the mass calculated by combining two particles. Sometimes those two particles
# came from some random process (background), but sometimes they came from 
# some new particle we are hunting for (signal)!
#
# Let's generate these data!
################################################################################

# So here's your signal data!
Nsig = 100
sig_mean = 10.1
sig_width = 0.05
signal = np.random.normal(sig_mean,sig_width,Nsig)

# So here's your background data!
Nbkg = 950
background = 9.0+(2*np.random.random(Nbkg))

# Combine the background and signal, because when we run the experiment, we actually
# don't know which is which!
data = signal.copy()
data = np.append(data,background.copy())

# Here's a very simple plot of our data.
plt.figure()
plt.hist(data,bins=50)


################################################################################
# Question #1
################################################################################
# Your total probability *for each event* will now be composed of two separate 
# probabilities:
#
# 1) the probability that an event came from signal
# 2) the probability that an event came from bacground
# 
# The first part you know how to do (Gaussian). The second part is a bit more challenging, 
# but because the background looks ``flat", it means that every value is equally likely
# over that range (from 9.0 to 11.0). So the probability for an event to come from background
# will be 
#
#    P(bkg) = 1/(range) = 1/(11-9) = 1/2
#
# So even though it's a bit weird, the probability for an event to come from background is
# just 0.5, regardless of where it comes from (so long as it is between 9.0 and 11.0.
#
# You also need to account for the fraction (frac) of the numbers that come from signal or background.
# So your total probability for each event will be....
#
#   P = frac*P(sig)  +  (1-frac)*P(bkg)
#
# Your goal is to vary the mean and width of the Gaussian *and* the fractional amount (frac)
# of signal and background, as described above, to find the values that give you the 
# maximum likelihood. 
#
# Good luck!


# YOUR WORK HERE.


plt.show()

