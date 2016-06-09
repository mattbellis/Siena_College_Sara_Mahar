import numpy as np
import matplotlib.pylab as plt
import math as math
%matplotlib  notebook

################################################################################
# This is how we define our own function
################################################################################
def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*(width**2)))

    return y


################################################################################
# First thing I'm going to do is to generate some fake data for us to work with.
# 
# Suppose I have test scores from 200 students. 
################################################################################

# So here's your fake data!
Nstudents = 200
fake_mean = 80
fake_width = 5
scores = np.random.normal(fake_mean,fake_width,Nstudents)

# Here's a very simple plot of your data
plt.figure()
plt.hist(scores,bins=25)#,range=(60,100))

# Functional form for a Gaussian
# y = e^{(x-mean)^2/(2*width)^2
plt.figure()
x = np.linspace(60,100,1000)
y = Gaussian(x,fake_mean,fake_width)
plt.plot(x,y)


################################################################################
# Question #1
################################################################################
# Calculate the probabilities of measuring all these points if they came from a 
# Gaussian of mean=90 and width=10.

# YOUR WORK HERE.

def calc_probability(x, mean, width):
	form1= (1/(((2*np.pi)**.5)*((width)**.5)))
	form2= np.exp(-((x-mean)**2)/(2*width))
	return form1*form2
	
probabilities=[]
for num in scores:
	probabilities.append(calc_probability(num, 80, 10))

print 'Question 1: Calculate the probabilities of obtaining these scores from a gaussian with a width of 10 and mean of 90. \n \n'


################################################################################
# Question #2
################################################################################
# What is the product of those probabilities?

# YOUR WORK HERE.
def summation(probability_input):

	find_summation=0
	for num in probability_input:
		find_summation+=np.log(num)

print summation(probabilities)

print '\n \nQuestion 2: Taking the product of the probabilities is difficult because all the numbers are so small.  So the way to to take the product is to take the natural log and the sum all of the factors.'

################################################################################
# Question #3
################################################################################
# Vary the mean and width to find the maximum probability of measuring those
# particular test scores.

# YOUR WORK HERE.
lower_mean=75
upper_mean=90
lower_width=1
upper_width=10
steps=.1

max_probability=-99999999999999999999
best_mean=0
best_width=0
for i in np.arange(lower_mean,upper_mean,steps):
    for j in np.arange(lower_width,upper_width,steps):
	test_probability=0
	for num in scores:
		test_probability+=np.log(Gaussian(num, i, j))
    	if max_probability<test_probability:
        	max_probability=test_probability
        	best_mean=i
		best_width=j

print max_probability
print best_mean
print best_width

print '\n \nQuestion 3:  Explore the parameter space!  Very the parameters to maximise the sum of the log of the product of probabilities.'
