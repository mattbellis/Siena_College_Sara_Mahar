import numpy as np
import matplotlib.pylab as plt
import math as math
from scipy.optimize import fmin
from iminuit import Minuit, describe, Struct
from scipy.spatial import distance
from scipy.sparse import vstack
import time
import seaborn as sns
#%matplotlib  notebook

nbkg = 1000
sigmeans = [5.0,7.0]
bkglos = [3.5,5]
bkghis = [6,9]
num_bootstrapping_samples=100


def calc_pull(iterations, nsig, nMC, nneigh,cdist_bool,sigwidths):
    
    pull_frac_list=[]
    average_best_frac = 0
    frac = []
    fit_frac = []
    fit_frac_uncert = []
    frac_org = nsig/float(nsig+nbkg)

    for num in range(iterations):
        
        nsig_iteration = np.random.poisson(nsig)
        nbkg_iteration = np.random.poisson(nbkg)
        data = gen_sig_and_bkg([nsig_iteration,nbkg_iteration],sigmeans,sigwidths,bkglos,bkghis)
        signal_points= signal_2D(nMC,sigmeans,sigwidths)
        background_points = background_2D(nMC,bkglos,bkghis)
        frac_iteration = float(nsig_iteration)/(float(nbkg_iteration+nsig_iteration))
        frac.append(frac_iteration)
        
        if cdist_bool:
            signal_prob=nncdist(data,signal_points, nneighbors=nneigh)
            background_prob= nncdist(data,background_points, nneighbors=nneigh)
        else:
            signal_prob=nn(data,signal_points, nneighbors=nneigh)
            background_prob = nn(data,background_points, nneighbors=nneigh)

        def tot_prob(frac):
            tot_prob=[]
            tot_prob.append(frac*signal_prob + ((1-frac)*background_prob))
            return np.array(tot_prob)
        
        def probability(frac):
            prob=tot_prob(frac)
            return -np.log(prob[prob>0]).sum()
        
        m1=Minuit(probability,frac= 0.2,limit_frac=(0.001,1),error_frac=0.001,errordef = 0.5,print_level=0)
        m1.migrad()

        if (m1.get_fmin().is_valid):
            param=m1.values
            err=m1.errors
            fit_frac.append(param["frac"])
            fit_frac_uncert.append(err["frac"])
            pull_frac=(frac_org-param["frac"])/err["frac"]
            pull_frac_list.append(pull_frac)
            
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,iterations


def calc_pull_varying_MC(iterations, nsig, nMC_sig, nMC_bkg, nneigh,cdist_bool,sigwidths):
    
    pull_frac_list=[]
    average_best_frac = 0
    frac = []
    fit_frac = []
    fit_frac_uncert = []
    frac_org = nsig/float(nsig+nbkg)

    for num in range(iterations):
        
        nsig_iteration = np.random.poisson(nsig)
        nbkg_iteration = np.random.poisson(nbkg)
        data = gen_sig_and_bkg([nsig_iteration,nbkg_iteration],sigmeans,sigwidths,bkglos,bkghis)
        
        signal_points= signal_2D(nMC_sig,sigmeans,sigwidths)
        background_points = background_2D(nMC_bkg,bkglos,bkghis)
        frac_iteration = float(nsig_iteration)/(float(nbkg_iteration+nsig_iteration))
        frac.append(frac_iteration)
        
        if cdist_bool:
            signal_prob=nncdist(data,signal_points, nneighbors=nneigh)
            background_prob= nncdist(data,background_points, nneighbors=nneigh)
        else:
            signal_prob=nn(data,signal_points, nneighbors=nneigh)
            background_prob = nn(data,background_points, nneighbors=nneigh)

        def tot_prob(frac):
            tot_prob=[]
            #tot_prob.append(frac*signal_prob*(nMC_sig/(nMC_sig+nMC_bkg)) + ((1-frac)*background_prob)*(nMC_bkg)/(nMC_sig+nMC_bkg))
            tot_prob.append(frac*signal_prob/(nMC_sig) + ((1-frac)*background_prob)/(nMC_bkg))
            return np.array(tot_prob)
        
        def probability(frac):
            prob=tot_prob(frac)
            return -np.log(prob[prob>0]).sum()
        
        m1=Minuit(probability,frac= 0.2,limit_frac=(0.001,1),error_frac=0.001,errordef = 0.5,print_level=0)
        m1.migrad()

        if (m1.get_fmin().is_valid):
            param=m1.values
            err=m1.errors
            fit_frac.append(param["frac"])
            fit_frac_uncert.append(err["frac"])
            pull_frac=(frac_org-param["frac"])/err["frac"]
            pull_frac_list.append(pull_frac)
            
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,iterations


#Calculates the number of nearest neighbors or radius depending on input.
def nn(data0,data1,r=None,nneighbors=None):
    ret = -1
    ret_list=[]
    #if input has neither radius or nneighbors it exits.
    if r is not None and nneighbors is not None:
        exit(-1)
        return ret
    
    #if input has radius as the input. Calculates number of neighbors and returns list of neighbors for each data point.
    elif r is not None and nneighbors is None:
        rsq = r*r
        for d in data0.transpose():
            count=0
            diffx=d[0]-data1[0]
            diffy=d[1]-data1[1]
            diff= diffx*diffx + diffy*diffy
            count = len(diff[diff<rsq])
            ret_list.append(float(count)/(float(len(data1[0]))*r))
        ret_list = np.array(ret_list)
        return ret_list
    
    #if input is number of nearest neighbors, it calculates the radius it has to go out for that particular number and returns a list of radii for each data point.
    elif r is None and nneighbors is not None:
        for d in data0.transpose():
            diffx=d[0]-data1[0]
            diffy=d[1]-data1[1]
            diff= diffx*diffx + diffy*diffy
            diff.sort()
            radius2 = diff[nneighbors-1]
            ret_list.append(float(nneighbors)/(np.pi*radius2)) # Let's do the inverse of the radius squared, since this is a 2D problem.

        ret_list = np.array(ret_list)
        return ret_list
    return ret





def nncdist(data0,data1,r=None,nneighbors=None):   
    ret = -1
    ret_list=[]
    if r is not None and nneighbors is not None:
        exit(-1)
        return ret
    elif r is not None and nneighbors is None:
        combined = data0.transpose()
        combined1 = data1.transpose()
        dist=distance.cdist(combined,combined1,'euclidean')
        for num in dist:
            count=len(num[num<r])
            ret_list.append(float(count)/(float(len(data1[0]))*r))
        ret_list = np.array(ret_list)
        return ret_list
    elif r is None and nneighbors is not None:
        for num0 in data0:
            diff = np.abs(num0 - data1)
            diff.sort()
            radius2 = diff[nneighbors-1]
            ret_list.append(1./np.sqrt(radius2)) # radius 
        ret_list = np.array(ret_list)
        return ret_list
    return ret





#Calulates the pulls for the mean and std.
def calc_pull_w_bootstrapping(pull_iterations, nsig, nMC, nneigh,cdist_bool,sigwidths):
    
    pull_frac_list=[]
    average_best_frac = 0
    frac = []
    fit_frac = []
    fit_frac_uncert = []
    frac_org = nsig/float(nsig+nbkg)

    for num in range(pull_iterations):
        
        # Generate the data for this pull iteration
        nsig_iteration = np.random.poisson(nsig)
        nbkg_iteration = np.random.poisson(nbkg)
        data = gen_sig_and_bkg([nsig_iteration,nbkg_iteration],sigmeans,sigwidths,bkglos,bkghis)

        # Record the original amount of signal and background data
        frac_iteration = float(nsig_iteration)/(float(nbkg_iteration+nsig_iteration))
        frac.append(frac_iteration)
                
        # Generate the MC we will use to try to fit the data we just generated!
        signal_points= signal_2D(nMC,sigmeans,sigwidths)
        background_points = background_2D(nMC,bkglos,bkghis)

        # Calculate the signal and background prob with original MC samples
        signal_prob=nn(data,signal_points, nneighbors=nneigh)
        background_prob = nn(data,background_points, nneighbors=nneigh)        

        # Generate MC bootstrap samples and calculate the probs for each
        signal_MC_bs = []
        background_MC_bs = []

        signal_probs_bs = []
        background_probs_bs = []
        for i in range(0,num_bootstrapping_samples):
            signal_MC_bs.append(bootstrapping(signal_points))
            background_MC_bs.append(bootstrapping(background_points))

            signal_probs_bs.append(nn(data,signal_MC_bs[i], nneighbors=nneigh))
            background_probs_bs.append(nn(data,background_MC_bs[i], nneighbors=nneigh))        

        
        def tot_prob(frac,sig,bkg):
            tot_prob = frac*sig + ((1-frac)*bkg)            
            return tot_prob
        
        def negative_log_likelihood(frac):
            
            # First, use the original MC/probs to calculate the NLL
            prob=tot_prob(frac,signal_prob,background_prob)
            nll =  -np.log(prob[prob>0]).sum()
            
            # Then add in the prob/NLLs for the bootstrap samples
            for i in range(0,num_bootstrapping_samples):
                prob = tot_prob(frac,signal_probs_bs[i],background_probs_bs[i])
                nll +=  -np.log(prob[prob>0]).sum()
            return nll
        
        m1=Minuit(negative_log_likelihood,frac= 0.2,limit_frac=(0.001,1),error_frac=0.001,errordef =(num_bootstrapping_samples+1)*0.5,print_level=0)
        m1.migrad()

        if (m1.get_fmin().is_valid):
            param=m1.values
            err=m1.errors
            fit_frac.append(param["frac"])
            fit_frac_uncert.append(err["frac"])
            pull_frac=(frac_org-param["frac"])/err["frac"]
            pull_frac_list.append(pull_frac)
            
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,pull_iterations






def bootstrapping(data):
    npts = len(data[0])
    indices = np.random.randint(0,npts,npts)

    bs_data= np.array([data[0][indices].copy(),data[1][indices].copy()])
        
    return bs_data






def normal(x,mean,width):
    return (1.0/(width*np.sqrt(2*np.pi)))*(np.exp(-(x-mean)**2/(2*(width**2))))





# A product of two Gaussians
def signal_2D(npts,means,sigmas):
    pts = []
    for m,s in zip(means,sigmas):
        pts.append(np.random.normal(m,s,npts))
    pts = np.array(pts)
    return pts
    
    
    
    
    

# Flat in 2D
def background_2D(npts,lovals,hivals):
    pts = []
    for lo,hi in zip(lovals,hivals):
        width = hi-lo
        pts.append(lo + width*np.random.random(npts))
    pts = np.array(pts)
    return pts






# Helper function to generate signal and background at the same time
def gen_sig_and_bkg(npts,means,sigmas,lovals,hivals):
    sigpts = signal_2D(npts[0],means,sigmas)
    bkgpts = background_2D(npts[1],lovals,hivals)
    data = [sigpts[0].copy(),sigpts[1].copy()]
    data[0] = np.append(data[0],bkgpts[0])
    data[1] = np.append(data[1],bkgpts[1])
    data = np.array(data)
    return data
