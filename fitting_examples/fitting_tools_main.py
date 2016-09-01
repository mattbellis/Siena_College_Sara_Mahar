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

# For GPU stuff
from numba import cuda
import numba
import accelerate.cuda.sorting as csort

sigmeans = [5.0,7.0]
bkglos = [3.5,5]
bkghis = [6,9]
probability_background=.1 #This is found by using the volume of the sample. (2.5*4*h=1 and solve for h)

    
def bootstrapping(data):
    #since 1D data comes over with the amount of points, the length will be greater than 2
    if len(data) > 2:
        npts = len(data)
        indices = np.random.randint(0,npts,npts)
        bs_data= data[indices].copy()
        
    #2D data comes over in an array with length 2 since it has 2 dimensions
    else:    
        npts = len(data[0])
        indices = np.random.randint(0,npts,npts)
        bs_data= np.array([data[0][indices].copy(),data[1][indices].copy()])

        
    return bs_data



def normal(x,mean,width):
    return (1.0/(width*np.sqrt(2*np.pi)))*(np.exp(-(x-mean)**2/(2*(width**2))))

##################################################################
# Log normal stuff
##################################################################


# A product of two lognormals
def function_2D_lognormal(npts,means,sigmas,los,his):
    pts = []
    for m,s in zip(means,sigmas):
        #print m,s
        pts.append(np.random.lognormal(m,s,npts))
    pts = np.array(pts)
    index = pts[0]>los[0]
    index *= pts[0]<his[0]
    index *= pts[1]>los[1]
    index *= pts[1]<his[1]
    
    skimmed_pts = [None,None]
    skimmed_pts[0] = pts[0][index]
    skimmed_pts[1] = pts[1][index]
    skimmed_pts = np.array(skimmed_pts)

    return skimmed_pts
    
    


# Helper function to generate signal and background at the same time
def gen_sig_and_bkg_lognormal(npts,sigmeans,sigsigmas,bkgmeans,bkgsigmas,los,his):
    sigpts = function_2D_lognormal(npts[0],sigmeans,sigsigmas,los,his)
    bkgpts = function_2D_lognormal(npts[1],bkgmeans,bkgsigmas,los,his)
    data = [sigpts[0].copy(),sigpts[1].copy()]
    data[0] = np.append(data[0],bkgpts[0])
    data[1] = np.append(data[1],bkgpts[1])
    data = np.array(data)
    return data



##################################################################


# A product of two Gaussians
def signal_2D(npts,means,sigmas):
    pts = []
    for m,s in zip(means,sigmas):
        pts.append(np.random.normal(m,s,npts))
    pts = np.array(pts)
    return pts
    
    
    
    
#Flat background
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



##################################################
# GPU testing
##################################################
@numba.cuda.jit("void(float32[:],float32[:],uint32,float32[:],float32[:],uint32,float32[:])") #,float32)")
def distances_GPU(arr_ax, arr_ay, narr_a, arr_bx, arr_by, narr_b, arr_out):  #, nneigh):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw #The speific index of the thread 

    if i<narr_a:
        a0x = arr_ax[i] #The x value for each thread and data set 1
        a0y = arr_ay[i] #The y value for each thread and data set 1

        diff=0.0
        for d in xrange(narr_b): #looping through the second data set
            diffx=a0x-arr_bx[d] #finding the difference of the x value in data set 1 and the whole data set 2
            diffy=a0y-arr_by[d]
            diff=diffx*diffx + diffy*diffy

            idx_out = narr_b*i + d
            arr_out[idx_out] = diff
            
     
    
##################################################
# GPU testing for 1D arrays
##################################################
@numba.cuda.jit("void(float32[:],uint32,float32[:],uint32,float32[:])") #,float32)")
def distances_GPU_1D(arr_ax, narr_a, arr_bx, narr_b, arr_out):  #, nneigh):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw #The specific index of the thread 
    if i<narr_a:
        a0x = arr_ax[i] #The x value for each thread and data set 1

        diff=0.0
        for d in xrange(narr_b): #looping through the second data set
            diffx=a0x-arr_bx[d] #finding the difference of the x value in data set 1 and the whole data set 2
            idx_out = narr_b*i + d
            arr_out[idx_out] = diffx   
            
            
            
            
def sort_GPU(ndata,nMC,in_arr,nneigh):
    #array=in_arr.flatten()
    array=in_arr
    index = np.arange(0,ndata*nMC).astype(np.uint32)
    #x = np.arange(nMC,ndata*nMC,nMC).astype(np.uint32)
    x = np.arange(0,(ndata*nMC),nMC).astype(np.uint32)
    
    
    segsort = csort.segmented_sort(array,index,x)
    #csort.segmented_sort(array,index,x)
    radius2=float(nneigh)/(np.pi*(array[nneigh+x-1]))
    radius2 = np.float64(radius2)
    return radius2




def sort_GPU_1D(ndata,nMC,in_arr,nneigh):
    array=in_arr
    index = np.arange(0,ndata*nMC).astype(np.uint32)
    
    x = np.arange(0,(ndata*nMC),nMC).astype(np.uint32)
    segsort = csort.segmented_sort(array,index,x)
    radius=array[nneigh+x-1]
    radius=1.0/radius
    radius = np.float64(radius)
    #print radius
    return radius





def calc_pull_main(iterations, nsig, nbkg, nMC_sig, nMC_bkg, num_bootstrapping_samples, nneigh, sigwidths, sigmeans, dim, tag="default"):
    outfile_name='frac_values_%s_sig%d_bkg%d_MCsig%d_MCbkg%d_bs%d_nn%d_%ddimension.dat'%(tag,nsig,nbkg,nMC_sig,nMC_bkg,num_bootstrapping_samples,nneigh,dim)
    outfile=open(outfile_name,'w')
    print 'writing out to file %s' %outfile_name
    
    sigmeans = [0.0, 0.0]
    sigwidths = [0.20, 0.20]

    bkgmeans = [0.0, 0.0]
    bkgwidths = [0.7, 0.7]

    los = [0,0]
    his = [100,100]
    
    pull_frac_list=[]
    average_best_frac = 0
    frac = []
    fit_frac = []
    fit_frac_uncert = []
    frac_org = nsig/float(nsig+nbkg)
    my_gpu = numba.cuda.get_current_device()
    thread_ct = my_gpu.WARP_SIZE

    for num in range(iterations):
        
        nsig_iteration = np.random.poisson(nsig)
        nbkg_iteration = np.random.poisson(nbkg)
        
        #generating 1D data
        if dim==1:
            sigmeans=10.4
            sigwidths=.06
            signal = np.random.normal(sigmeans,sigwidths,nsig_iteration)
            background = 9.0+(2*np.random.random(nbkg_iteration))
            data = signal.copy()
            data = np.append(data,background.copy())
            #generate MC data
            signal_MC = np.random.normal(sigmeans,sigwidths,nMC_sig)
            background_MC= 9.0+(2*np.random.random(nMC_bkg))
            
        #generating 2D data
        else:
            #2 Gaussian data
            #data = gen_sig_and_bkg([nsig_iteration,nbkg_iteration],sigmeans,sigwidths,bkglos,bkghis)
            #signal_MC= signal_2D(nMC_sig,sigmeans,sigwidths)
            #background_MC = background_2D(nMC_bkg,bkglos,bkghis)
            
            #2D lognormal data 
            data = gen_sig_and_bkg_lognormal([nsig_iteration,nbkg_iteration], sigmeans,sigwidths, bkgmeans,bkgwidths,los,his)
            signal_MC= function_2D_lognormal(nMC_sig,sigmeans,sigwidths,los,his)
            background_MC = function_2D_lognormal(nMC_bkg,bkgmeans,bkgwidths,los,his)
        
        ndata_iteration = len(data[0])
        nMC_sig = len(signal_MC[0])
        nMC_bkg = len(background_MC[0])
        #print ndata_iteration,nMC_sig,nMC_bkg
        
        #Block count
        block_ct_sig = int(math.ceil(float(nMC_sig*(ndata_iteration)) / thread_ct))
        block_ct_bkg = int(math.ceil(float(nMC_bkg*(ndata_iteration)) / thread_ct))
        #setting up distances between data points and background MC and data points and signal MC
        signal_distances_GPU = np.zeros((ndata_iteration)*nMC_sig, dtype = np.float32)
        background_distances_GPU = np.zeros((ndata_iteration)*nMC_bkg, dtype = np.float32)
        
        #finding distances and sorting them through the GPU for 1D  
        if dim==1:
            signal_distances=distances_GPU_1D[block_ct_sig, thread_ct](np.float32(data), len(data), np.float32(signal_MC),len(signal_MC), signal_distances_GPU)
            background_distances=distances_GPU_1D[block_ct_bkg, thread_ct](np.float32(data), len(data), np.float32(background_MC), len(background_MC), background_distances_GPU)
            signal_prob=sort_GPU_1D((nsig_iteration+nbkg_iteration),nMC_sig,np.abs(signal_distances_GPU), nneigh)
            background_prob=sort_GPU_1D((nsig_iteration+nbkg_iteration),nMC_bkg, np.abs(background_distances_GPU), nneigh)
            
        #finding distances and sorting them through the GPU for 2D    
        else:  
            signal_distances=distances_GPU[block_ct_sig, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(signal_MC[0]), np.float32(signal_MC[1]),len(signal_MC[1]), signal_distances_GPU)
            background_distances=distances_GPU[block_ct_bkg, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(background_MC[0]), np.float32(background_MC[1]),len(background_MC[1]), background_distances_GPU)
            signal_prob=sort_GPU((ndata_iteration),nMC_sig, signal_distances_GPU,nneigh)
            background_prob=sort_GPU((ndata_iteration),nMC_bkg, background_distances_GPU,nneigh)
            
        signal_MC_bs = []
        background_MC_bs = []
        signal_prob_bs = []
        background_prob_bs = []
        
        #If there are bootstrapping samples to be done
        for i in range(0,num_bootstrapping_samples):
            
            #Generating bootstrapping samples
            signal_MC_bs.append(bootstrapping(signal_MC))
            background_MC_bs.append(bootstrapping(background_MC))
            
            signal_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_sig, dtype = np.float32)
            background_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_bkg, dtype = np.float32)

            if dim==1:
            #calculating distances for each point
                signal_distances=distances_GPU_1D[block_ct_sig, thread_ct](np.float32(data), len(data), np.float32(signal_MC_bs[i]),len(signal_MC_bs[i]), signal_distances_GPU)
                background_distances=distances_GPU_1D[block_ct_bkg, thread_ct](np.float32(data), len(data), np.float32(background_MC_bs[i]), len(background_MC_bs[i]), background_distances_GPU)
                signal_prob_bs.append(sort_GPU_1D((nsig_iteration+nbkg_iteration),nMC_sig, np.abs(signal_distances_GPU), nneigh))
                background_prob_bs.append(sort_GPU_1D((nsig_iteration+nbkg_iteration),nMC_bkg, np.abs(background_distances_GPU), nneigh))
                 #print signal_prob_bs
            else: 

                signal_distances=distances_GPU[block_ct_sig, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(signal_MC_bs[i][0]), np.float32(signal_MC_bs[i][1]),len(signal_MC_bs[i][1]), signal_distances_GPU)
            
                background_distances=distances_GPU[block_ct_bkg, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(background_MC_bs[i][0]), np.float32(background_MC_bs[i][1]),len(background_MC_bs[i][1]), background_distances_GPU)
           
            
                signal_prob_bs.append(sort_GPU((nsig_iteration+nbkg_iteration),nMC_sig, signal_distances_GPU,nneigh))
                background_prob_bs.append(sort_GPU((nsig_iteration+nbkg_iteration),nMC_bkg, background_distances_GPU,nneigh))
        
        
        def tot_prob(frac,sig,bkg):
            tot_prob = frac*sig/nMC_sig + ((1-frac)*bkg/nMC_bkg)
            #print frac,tot_prob
            #print sig
            #print nMC_sig
            #print bkg
            #print nMC_bkg
            return tot_prob
        
        def probability(frac):
            prob= tot_prob(frac,signal_prob,background_prob)
            nll= -np.log(prob[prob>0]).sum()
            for i in range(0,num_bootstrapping_samples):
                prob = tot_prob(frac,signal_prob_bs[i],background_prob_bs[i])
                nll +=  -np.log(prob[prob>0]).sum()
            return nll
        
        m1=Minuit(probability,frac= 0.2,limit_frac=(0.001,1),error_frac=0.01,errordef = 0.5*(num_bootstrapping_samples+1), print_level=0)
        
        m1.migrad()

        if (m1.get_fmin().is_valid):
            param=m1.values
            err=m1.errors
            fit_frac.append(param["frac"])
            fit_frac_uncert.append(err["frac"])
            pull_frac=(frac_org-param["frac"])/err["frac"]
            pull_frac_list.append(pull_frac)

        output="%f %f %f %d %d %d %d %d %d %d %d %d\n" % (frac_org,param["frac"],err["frac"], nsig,nsig_iteration,nbkg,nbkg_iteration,dim,nMC_sig,nMC_bkg,num_bootstrapping_samples,nneigh)
        outfile.write(output)
    outfile.close()
            
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,iterations,outfile