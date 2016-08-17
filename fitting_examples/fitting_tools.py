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




def calc_pull(iterations, nsig, nbkg, nMC_sig, nMC_bkg, sigwidths, nneigh=None, rad=None,tag="default"):
    outfile_name='frac_values_%s_sig%d_bkg%d_MCsig%d_MCbkg%d_bs%d_nn%d.dat'%(tag,nsig,nbkg,nMC_sig,nMC_bkg,0,nneigh)
    outfile=open(outfile_name,'w')
    print 'writing out to file %s' %outfile_name
    
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
        
        if nneigh is not None:
            signal_prob=nn(data,signal_points, nneighbors=nneigh)
            background_prob = nn(data,background_points, nneighbors=nneigh)
        else:
            signal_prob=nn(data,signal_points, r=rad)
            background_prob = nn(data,background_points, r=rad)
            print "calculating nn with radius"

        def tot_prob(frac):
            tot_prob=[]
            tot_prob.append((frac*signal_prob)/nMC_sig + ((1-frac)*background_prob/nMC_bkg))
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
        output="%f %f %f %d %d %d %d %d %d %d %d\n" % (frac_org,param["frac"],err["frac"], nsig,nsig_iteration,nbkg,nbkg_iteration,nMC_sig,nMC_bkg,0,nneigh)
        outfile.write(output)
    outfile.close()
            
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,iterations,outfile







#Calulates the pulls for the mean and std.
def calc_pull_w_bootstrapping(pull_iterations, nsig, nbkg,nMC_sig, nMC_bkg, num_bootstrapping_samples,  nneigh,sigwidths, tag="default"):
    outfile_name='frac_values_%s_sig%d_bkg%d_MCsig%d_MCbkg%d_bs%d_nn%d.dat'%(tag,nsig,nbkg,nMC_sig,nMC_bkg,num_bootstrapping_samples,nneigh)
    outfile=open(outfile_name,'w')
    print 'writing out to file %s' %outfile_name
    
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
        signal_points= signal_2D(nMC_sig,sigmeans,sigwidths)
        background_points = background_2D(nMC_bkg,bkglos,bkghis)

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
            tot_prob = frac*sig/nMC_sig + ((1-frac)*bkg/nMC_bkg)            
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
            
        output="%f %f %f %d %d %d %d %d %d %d %d\n" % (frac_org,param["frac"],err["frac"], nsig,nsig_iteration,nbkg,nbkg_iteration,nMC_sig,nMC_bkg,num_bootstrapping_samples,nneigh)
        outfile.write(output)
    outfile.close()
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,pull_iterations,outfile




def calc_pull_GPU(iterations, nsig, nbkg, nMC_sig, nMC_bkg, sigwidths, nneigh=None, rad=None,tag="default"):
    outfile_name='frac_values_%s_sig%d_bkg%d_MCsig%d_MCbkg%d_bs%d_nn%d.dat'%(tag,nsig,nbkg,nMC_sig,nMC_bkg,0,nneigh)
    outfile=open(outfile_name,'w')
    print 'writing out to file %s' %outfile_name
    
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
        data = gen_sig_and_bkg([nsig_iteration,nbkg_iteration],sigmeans,sigwidths,bkglos,bkghis)
        signal_points= signal_2D(nMC_sig,sigmeans,sigwidths)
        background_points = background_2D(nMC_bkg,bkglos,bkghis)
        frac_iteration = float(nsig_iteration)/(float(nbkg_iteration+nsig_iteration))
        frac.append(frac_iteration)
        
        #Block count and setting up arrays for distances between data points and background MC and 
        #data points and signal MC
        block_ct_sig = int(math.ceil(float(nMC_sig*(nsig_iteration+nbkg_iteration)) / thread_ct))
        block_ct_bkg = int(math.ceil(float(nMC_bkg*(nsig_iteration+nbkg_iteration)) / thread_ct))
        signal_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_sig, dtype = np.float32)
        background_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_bkg, dtype = np.float32)
        
        
        
        if nneigh is not None:
            signal_distances=distances_GPU[block_ct_sig, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(signal_points[0]), np.float32(signal_points[1]),len(signal_points[1]), signal_distances_GPU)
            
            background_distances=distances_GPU[block_ct_bkg, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(background_points[0]), np.float32(background_points[1]),len(background_points[1]), background_distances_GPU)
           
            
            signal_prob=sort_GPU((nsig_iteration+nbkg_iteration),nMC_sig,signal_distances_GPU,nneigh)
            background_prob=sort_GPU((nsig_iteration+nbkg_iteration),nMC_bkg,background_distances_GPU,nneigh)
            
        else:
            signal_prob=nn(data,signal_points, r=rad)
            background_prob = nn(data,background_points, r=rad)
            print "calculating nn with radius"

        def tot_prob(frac):
            tot_prob=[]
            tot_prob.append((frac*signal_prob)/nMC_sig + ((1-frac)*background_prob/nMC_bkg))
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
        output="%f %f %f %d %d %d %d %d %d %d %d\n" % (frac_org,param["frac"],err["frac"], nsig,nsig_iteration,nbkg,nbkg_iteration,nMC_sig,nMC_bkg,0,nneigh)
        outfile.write(output)
    outfile.close()
            
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,iterations,outfile







#Calulates the pulls for the mean and std.
def calc_pull_w_bootstrapping_GPU(pull_iterations, nsig, nbkg,nMC_sig, nMC_bkg, num_bootstrapping_samples,  nneigh,sigwidths, tag="default"):
    outfile_name='frac_values_%s_sig%d_bkg%d_MCsig%d_MCbkg%d_bs%d_nn%d.dat'%(tag,nsig,nbkg,nMC_sig,nMC_bkg,num_bootstrapping_samples,nneigh)
    outfile=open(outfile_name,'w')
    print 'writing out to file %s' %outfile_name
    
    pull_frac_list=[]
    average_best_frac = 0
    frac = []
    fit_frac = []
    fit_frac_uncert = []
    frac_org = nsig/float(nsig+nbkg)
    my_gpu = numba.cuda.get_current_device()
    thread_ct = my_gpu.WARP_SIZE

    for num in range(pull_iterations):        
        # Generate the data for this pull iteration
        nsig_iteration = np.random.poisson(nsig)
        nbkg_iteration = np.random.poisson(nbkg)
        data = gen_sig_and_bkg([nsig_iteration,nbkg_iteration],sigmeans,sigwidths,bkglos,bkghis)

        # Record the original amount of signal and background data
        frac_iteration = float(nsig_iteration)/(float(nbkg_iteration+nsig_iteration))
        frac.append(frac_iteration)
                
        # Generate the MC we will use to try to fit the data we just generated!
        signal_points= signal_2D(nMC_sig,sigmeans,sigwidths)
        background_points = background_2D(nMC_bkg,bkglos,bkghis)

        # Calculate the signal and background prob with original MC samples
        #Block count and setting up arrays for distances between data points and background MC and 
        #data points and signal MC
        block_ct_sig = int(math.ceil(float(nMC_sig*(nsig_iteration+nbkg_iteration)) / thread_ct))
        block_ct_bkg = int(math.ceil(float(nMC_bkg*(nsig_iteration+nbkg_iteration)) / thread_ct))
        
        signal_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_sig, dtype = np.float32)
        background_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_bkg, dtype = np.float32)
        
        signal_distances=distances_GPU[block_ct_sig, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(signal_points[0]), np.float32(signal_points[1]),len(signal_points[1]), signal_distances_GPU)
            
        background_distances=distances_GPU[block_ct_bkg, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(background_points[0]), np.float32(background_points[1]),len(background_points[1]), background_distances_GPU)
           
            
        signal_prob=sort_GPU((nsig_iteration+nbkg_iteration),nMC_sig,signal_distances_GPU,nneigh)
        background_prob=sort_GPU((nsig_iteration+nbkg_iteration),nMC_bkg,background_distances_GPU,nneigh)
     

        # Generate MC bootstrap samples and calculate the probs for each
        signal_MC_bs = []
        background_MC_bs = []
      
        

        signal_prob_bs = []
        background_prob_bs = []
        for i in range(0,num_bootstrapping_samples):
            
            #signal_probs_bs.append(nn(data,signal_MC_bs[i], nneighbors=nneigh))
            #background_probs_bs.append(nn(data,background_MC_bs[i], nneighbors=nneigh))   
            
            signal_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_sig, dtype = np.float32)
            background_distances_GPU = np.zeros((nsig_iteration+nbkg_iteration)*nMC_bkg, dtype = np.float32)
            signal_MC_bs=np.append(bootstrapping(signal_points),i)
            background_MC_bs=np.append(bootstrapping(background_points),i)
            signal_MC_bs=signal_MC_bs[i]
            background_MC_bs=background_MC_bs[i]
            
            #print signal_MC_bs
            #print data[0]
            #calculating distances for each point
            signal_distances=distances_GPU[block_ct_sig, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(signal_points[0]), np.float32(signal_points[1]),len(signal_points[1]), signal_distances_GPU)
            
            background_distances=distances_GPU[block_ct_bkg, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(background_points[0]), np.float32(background_points[1]),len(background_points[1]), background_distances_GPU)
            
            signal_prob_bs.append(sort_GPU((nsig_iteration+nbkg_iteration),nMC_sig,signal_distances_GPU,nneigh))
            background_prob_bs.append(sort_GPU((nsig_iteration+nbkg_iteration),nMC_bkg,background_distances_GPU,nneigh))
            
            
            '''
            def bootstrapping(data):
    npts = len(data[0])
    indices = np.random.randint(0,npts,npts)

    bs_data= np.array([data[0][indices].copy(),data[1][indices].copy()])
        
    return bs_data
    
    
    for i in range(0,num_bootstrapping_samples):
            signal_MC_bs.append(bootstrapping(signal_points))
            background_MC_bs.append(bootstrapping(background_points))

            signal_probs_bs.append(nn(data,signal_MC_bs[i], nneighbors=nneigh))
            background_probs_bs.append(nn(data,background_MC_bs[i], nneighbors=nneigh)) 
            '''
            
            
            background_distances=distances_GPU[block_ct_bkg, thread_ct](np.float32(data[0]), np.float32(data[1]), len(data[1]), np.float32(background_points[0]), np.float32(background_points[1]),len(background_points[1]), background_distances_GPU)
           
            
            signal_prob=sort_GPU((nsig_iteration+nbkg_iteration),nMC_sig,signal_distances_GPU,nneigh)
            background_prob=sort_GPU((nsig_iteration+nbkg_iteration),nMC_bkg,background_distances_GPU,nneigh)

        
        def tot_prob(frac,sig,bkg):
            tot_prob = frac*sig/nMC_sig + ((1-frac)*bkg/nMC_bkg)            
            return tot_prob
        
        def negative_log_likelihood(frac):
            
            # First, use the original MC/probs to calculate the NLL
            prob=tot_prob(frac,signal_prob,background_prob)
            nll =  -np.log(prob[prob>0]).sum()
            
            # Then add in the prob/NLLs for the bootstrap samples
            for i in range(0,num_bootstrapping_samples):
                prob = tot_prob(frac,signal_prob_bs[i],background_prob_bs[i])
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
            
        output="%f %f %f %d %d %d %d %d %d %d %d\n" % (frac_org,param["frac"],err["frac"], nsig,nsig_iteration,nbkg,nbkg_iteration,nMC_sig,nMC_bkg,num_bootstrapping_samples,nneigh)
        outfile.write(output)
    outfile.close()
    return pull_frac_list, frac, fit_frac, fit_frac_uncert,pull_iterations,outfile








#Calculate the mean and std of frac pulls
def calc_mean_std_of_pulls(values):
    pulls=(values[0]-values[1])/values[2]
    return (np.mean(pulls),np.std(pulls))






#Calculates the pulls for the analytic function
def calc_pull_analytic(iterations, nsig, nMC_sig, nMC_bkg, nneigh,sigwidths):
    
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

        def probability(frac):
            x=data[0]
            y=data[1]
            mean_x=sigmeans[0] #hard-coded for the means of x-direction
            mean_y=sigmeans[1] #hard-coded the mean of the y-direction
            width=.1
            signal_prob_x=(1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(x-mean_x)**2/(2*(width**2))) #must find probability in both x and y direction
            signal_prob_y=(1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(y-mean_y)**2/(2*(width**2))) 
            signal_prob=signal_prob_x*signal_prob_y
            tot_prob= -np.log(frac*signal_prob/nMC_sig+ ((1-frac)*probability_background)/nMC_bkg).sum()
            return tot_prob
        
        m1=Minuit(probability,frac= .15, limit_frac=(0.001,1), error_frac=0.001, print_level=0,errordef = 0.5)
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
        #print ret_list
        return ret_list
    
    return ret





#This calculates the number of nearest neighbors using cdist function
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


def nn_GPU(data_set0,data_set1, nneigh):
    my_gpu= numba.cuda.get_current_device()
    thread_ct = my_gpu.WARP_SIZE
    block_ct = int(math.ceil(float(data_size) / thread_ct))
    frac_nn_GPU = -999*np.ones(len(data))
    nn_kernel_GPU[block_ct, thread_ct](np.float32(data_set0[0]), np.float32(data_set0[1]), np.float32(data_set1[0]), np.float32(data_set1[1]), frac_nn_GPU, nneigh)
    return frac_nn_GPU
    



##################################################
# GPU testing
##################################################
@numba.cuda.jit("void(float32[:],float32[:],uint32,float32[:],float32[:],uint32,float32[:])") #,float32)")
def distances_GPU(arr_ax, arr_ay, narr_a, arr_bx, arr_by, narr_b, arr_out):  #, nneigh):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw #The spcific index of the thread 

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
            
            
            
            
            
            
            
            
def sort_GPU(ndata,nMC,in_arr,nneigh):
    #array=in_arr.flatten()
    array=in_arr
    index = np.arange(0,ndata*nMC).astype(np.uint32)
    #x = np.arange(nMC,ndata*nMC,nMC).astype(np.uint32)
    x = np.arange(0,(ndata*nMC),nMC).astype(np.uint32)
    
    
    segsort = csort.segmented_sort(array,index,x)
    #csort.segmented_sort(array,index,x)
    radius2=float(nneigh)/(np.pi*(array[nneigh+x-1]))
    return radius2





'''
##################################################
# GPU testing
##################################################
@numba.cuda.jit("void(float32[:],float32[:],float32[:],float32[:],float32[:],float32)")
def number_of_nearest_neighbors_GPU(arr_ax, arr_ay, arr_bx, arr_by, arr_out, radius):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw

    # Make sure we don't calculate the thread count that is bigger than the array.
    if i>= arr_out.size:
        return

    a0x = arr_ax[i]
    a0y = arr_ay[i]

    narr_b = len(arr_bx)
    
    for j in xrange(narr_b):
        diffx = a0x - arr_bx[j]
        diffy = a0y - arr_by[j]

        distance = math.sqrt(diffx * diffx + diffy * diffy)

        # Keep track of how many points are within our radius.
        if distance<=radius:
            arr_out[i] += 1

    # Normalize to the number of points.
    arr_out[i] /= narr_b
    # Alternate normalization.
    #arr_out[i] /= (k*narr_b)
   '''







def plot_data():
    # Test the tools to generate the datasets.
    
    sigpts = signal_2D(2000,[5.0,7.0],[0.1,0.1])
    sns.jointplot(sigpts[0],sigpts[1],kind='hex')
    bkgpts = background_2D(10000,[3.5,5],[6,9])
    sns.jointplot(bkgpts[0],bkgpts[1],kind='hex')
    data1 = [sigpts[0].copy(),sigpts[1].copy()]
    data1[0] = np.append(data1[0],bkgpts[0])
    data1[1] = np.append(data1[1],bkgpts[1])
    data1 = np.array(data1)
    sns.jointplot(data1[0],data1[1],kind='hex')


    
'''
#NEED TO FIXXXXXX
#input whether you want to plot the mean or the standard deviation
def plot_means_and_stds(which_plot, mean_or_std_list, width_list, MC_list):
    plt.figure()
    colors=['b','g','y','r']
    markers=['o','^','+']
    marker_sizes=[5,10,60]
    labels=[]

    if which_plot=='mean':
        
        for item in means1:
            if item['width']==width_list[0]:
                label='MC=' + str(item['MC_points']) + ' signal=' + str(item['signal'])
        
    
                if item['MC_points']== MC_list[0]:
                    color=colors[0]
                #if item['MC_points']==MC_list[1]:
                #    color=colors[1]

                if item['signal']==sig_list[0]:
                    marker=markers[0]
                #if item['signal']==sig_list[1]:
                #    marker=markers[1]
                for x in labels:
                    if label==x:
                        label=""       
                plt.plot(item['nearest neighbors'],item['mean pulls'], color=color, marker=marker,label=label)
                labels.append(label)

        plt.legend(loc='lower right')
        plt.title("Means with widths of "+str(width_list[0]))
           
        plt.figure()
        labels=[]
           
        for item in means1:
            if item['width']==width_list[1]:
                label='MC=' + str(item['MC_points']) +  ' signal=' + str(item['signal'])
    
                if item['MC_points']== MC_list[0]:
                    color=colors[0]
                #if item['MC_points']==MC_list[1]:
                #    color=colors[1]

            if item['signal']==sig_list[0]:
                marker=markers[0]
            #if item['signal']==sig_list[1]:
            #    marker=markers[1]

            for x in labels:
                if label==x:
                    label=""

            plt.plot(item['nearest neighbors'],item['mean pulls'], color=color, marker=marker,label=label)
            labels.append(label)

        plt.legend(loc='lower right')    

        plt.title("Means with widths of "+str(width_list[1]))
        
    if which_plot=='std'

        plt.figure()
        labels=[]

        for item in stds1:
            if item['width']==width_list[0]:
                label='MC=' + str(item['MC_points']) +  ' signal=' + str(item['signal'])

                if item['MC_points']== MC_list[0]:
                    color=colors[0]
                #if item['MC_points']==MC_list[1]:
                #    color=colors[1]

                if item['signal']==sig_list[0]:
                    marker=markers[0]
                #if item['signal']==sig_list[1]:
                #    marker=markers[1]

                for x in labels:
                    if label==x:
                        label=""

                plt.plot(item['nearest neighbors'],item['mean stds'], color=color, marker=marker,label=label)
                labels.append(label)

        plt.legend(loc='lower right')   

        plt.title("Standard Deviations with widths of "+str(width_list[0]))




        plt.figure()
        labels=[]

        for item in stds1:
            if item['width']==width_list[1]:
                label='MC=' + str(item['MC_points']) +  ' signal=' + str(item['signal'])

                if item['MC_points']== MC_list[0]:
                    color=colors[0]
                #if item['MC_points']==MC_list[1]:
                #    color=colors[1]

                if item['signal']==sig_list[0]:
                    marker=markers[0]
                #if item['signal']==sig_list[1]:
                #    marker=markers[1]

                for x in labels:
                    if label==x:
                        label=""

                plt.plot(item['nearest neighbors'],item['mean stds'], color=color, marker=marker,label=label)
                labels.append(label)

        plt.legend(loc='lower right')

        plt.title("Standard Deviations with widths of "+str(width_list[1]))


'''
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




def assign_names(trial):
    mean_name="mean_pulls" + str(trial)
    std_name = "std_pulls" + str(trial)
    uncertainty_name = "avg_uncertainty" + str(trial)
    return mean_name, std_name, uncertainty_name


#without cdist (pythagorean method) seems to be about 2/3 the time it takes to use cdist
def calc_pull_compare_cdist(iterations, nsig, nMC_sig, nMC_bkg, rad,cdist_bool,sigwidths):
    
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
            signal_prob=nncdist(data,signal_points, r=rad)
            background_prob= nncdist(data,background_points, r=rad)
        else:
            signal_prob=nn(data,signal_points, r=rad)
            background_prob = nn(data,background_points, r=rad)

        def tot_prob(frac):
            tot_prob=[]
            tot_prob.append(frac*signal_prob/nMC_sig + ((1-frac)*background_prob)/nMC_bkg)
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



