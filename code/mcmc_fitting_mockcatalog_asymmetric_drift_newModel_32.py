import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import datetime

#MCMC
import scipy.optimize as op
import emcee
import corner
from emcee.utils import MPIPool

#galpy
from astropy.table import Table

mpl.rcParams['agg.path.chunksize'] = 100000

hshR_prior = 2. # hs/hR from Marsha, Bland-Hawthorn et al. (2014)
hshR_prior_sigma = 0.5 # hs/hR from Marsha, Bland-Hawthorn et al. (2014)
#R0 = 8.2 +/- 0.1 kpc
R0_prior = 8.2 #kpc
R0_prior_sigma = 0.1
#Omega_sun = 30.24 +/- 0.12 km/s
Omega_prior = 30.24 # km/s/kpc
Omega_prior_sigma = 0.12 # km/s/kpc
_k = 4.74047  # km/s/kpc/(mas/yr)

distance_list = ['100pc','200pc','300pc','400pc','500pc',
                 '600pc','700pc','800pc','900pc','1kpc']
age_list = ['0to1','1to2','2to3','3to4','4to5',
            '5to6','6to7','7to8']
p0_list = [[17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #0to1
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #1to2
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #2to3
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #3to4
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #4to5
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #5to6
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5], #6to7
           [17.0,-11.0,-3.0,-1.0,9.6,11.5,7.2,8.2,2.0,3.0,3.0,2.5]] #7to8
            # A     B    C    K   u0  v0  w0  R0 hshR sml smb srv

#mcmc
ndim = 12
N_WALKERS = 60
Nrun= 3000
Nburn= 2000
#N_posterior= 200000
_list_labels = [r'$A$',
                r'$B$',
                r'$C$',
                r'$K$',
                r'$U_{\odot}$',
                r'$V_{\odot}$',
                r'$W_{\odot}$',
                r'$R_0$',
                r'$h_R$',
                r'$h_{\sigma}/h_R$',
                r'$s_{\mu_l}$',
                r'$s_{\mu_b}$',
                r'$s_{v_{\mathrm{los}}}$']

def lnprior(p):
    A = p[0]
    B = p[1]
    C = p[2]
    K = p[3]
    u0 = p[4]
    v0 = p[5]
    w0 = p[6]
    R0 = p[7]
    hshR = p[8]
    scatter_pml = np.exp(p[9])
    scatter_pmb = np.exp(p[10])
    scatter_vlos = np.exp(p[11])
    
    if -100 < A < 100 and -100 < B < 100 and -100 < C < 100 and -100 < K < 100 \
        and -100 < u0 < 100 and -100 < v0 < 100 and -100 < w0 < 100\
        and -1000 < scatter_pml < 1000 and -1000 < scatter_pmb < 1000\
        and -1000 < scatter_vlos < 1000:
        return -1
    return -np.inf

def lnlike(p,cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number):
    #p= [A,B,C,K,u0,v0,w0] 
    lnL = 0.

    A = p[0]
    B = p[1]
    C = p[2]
    K = p[3]
    u0 = p[4]
    v0 = p[5]
    w0 = p[6]
    R0 = p[7]
    hshR = p[8]
    scatter_pml = np.exp(p[9])
    scatter_pmb = np.exp(p[10])
    scatter_vlos = np.exp(p[11])
    
    D = 1./data_plx
    R = (D*D + R0*R0 - 2.*D*R0*cos_l)**0.5
    sin_t = D/R*sin_l
    cos_t = (1.-sin_t**2.)**0.5

    ###### convert from astrometric to plar coordinates
    vcirc = R0*(A-B)
    vl,vb = data_pml/data_plx*_k,data_pmb/data_plx*_k

    vx = -sin_l*vl - cos_l*sin_b*vb + cos_l*cos_b*data_vlos + u0
    vy =  cos_l*vl - sin_l*sin_b*vb + sin_l*cos_b*data_vlos + (v0+vcirc)
    #vz =                   cos_b*vb       + sin_b*vlos + w0

    vR =   cos_t*vx - sin_t*vy
    vphi = sin_t*vx + cos_t*vy
    #vz =                        vz 

    hR = -data_age*0.7 + 8.

    va = np.var(vR)/(2.*vcirc) * (np.var(vphi)/np.var(vR) - 1. + R0*(1./hR+2./(hshR*hR)))
    
    convert = np.array([[-sin_l*sin_t + cos_l*cos_t],
                        [-cos_l*sin_b*sin_t - sin_l*sin_b*cos_t],
                        [cos_l*cos_b*sin_t + sin_l*cos_b*cos_t]])

    Omega = (vcirc + v0)/R0

    pred_pml_l= (A*cos_twol-C*sin_twol+B)*cos_b+u0*sin_l*data_plx\
                 -v0*cos_l*data_plx - va*convert[0]*data_plx
    pred_pmb_l= -(A*sin_twol+C*cos_twol+K)*sin_b*cos_b\
                +sin_b*(u0*cos_l*data_plx+v0*sin_l*data_plx)\
                -w0*data_plx*cos_b - va*convert[1]*data_plx
    pred_vlos_l = (K+C*cos_twol+A*sin_twol)*cos_b**2./data_plx\
                  -((u0*cos_l+v0*sin_l)*cos_b+w0*sin_b) -va*convert[2]
    lnL = -0.5*np.log((2.*np.pi)**3.*((data_err_pml*_k)**2.+(scatter_pml*data_plx)**2.)\
                                    *((data_err_pmb*_k)**2.+(scatter_pmb*data_plx)**2.)\
                                    *(data_err_vlos**2.+scatter_vlos**2.))\
          -(pred_pml_l-data_pml*_k)**2./((data_err_pml*_k)**2.+(scatter_pml*data_plx)**2.)\
          -(pred_pmb_l-data_pmb*_k)**2./((data_err_pmb*_k)**2.+(scatter_pmb*data_plx)**2.)\
          -(pred_vlos_l-data_vlos)**2./(data_err_vlos**2.+scatter_vlos**2.)\
          -0.5*np.log(2.*np.pi*R0_prior_sigma**2.)-(R0-R0_prior)**2./(R0_prior_sigma**2.)\
          -0.5*np.log(2.*np.pi*Omega_prior_sigma**2.)-(Omega-Omega_prior)**2./(Omega_prior_sigma**2.)\
          -0.5*np.log(2.*np.pi*hshR_prior_sigma**2.)-(hshR-hshR_prior)**2./(hshR_prior_sigma**2.)
    return np.sum(lnL),va,np.mean(vR**2.),np.var(vphi),np.mean(vR),np.var(vR)

def lnprob(p, cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p, cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number)[0]

def run_mcmc(cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,\
             data_pmb,data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,\
             age_number,distance_number,Number):

    distance = distance_list[distance_number]
    age = age_list[age_number]
    
    #define the objective function
    negativelnLikelihood = lambda *args: -lnlike(*args)[0]
    
    #initial guess for p
    p_0 = p0_list[age_number]

    #generate random values. np.random.randn provides Gaussian with mean 0 and standard deviation 1
    #thus here is adding random values obeying the Gaussian like above to each values.
    pos = [p_0 + 1.*np.random.randn(ndim) for i in range(N_WALKERS)]

    #for multiprocessing
    pool = MPIPool(loadbalance=True)
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    obsevlosables= cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
                   data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number
    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, lnprob, args=obsevlosables)
          
    #sampler = emcee.EnsembleSampler(N_WALKERS, ndim, lnprob, pool=pool, \
    #          args=(cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
    #                data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos))
    sampler.run_mcmc(pos, Nrun)
    pool.close()

    print('Done.')
               
    #---
    # store the results
    burnin = Nburn
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    plt.clf()
    hight_fig_inch = np.int((ndim+1)*3.0)
    fig,axes = plt.subplots(ndim+1, 1, sharex=True, figsize=(8, hight_fig_inch))
    for i in range(ndim):
        axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.5)
        axes[i].set_ylabel(_list_labels[i])
    # last panel shows the evolution of ln-likelihood for the ensemble of walkers
    axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.5)
    axes[-1].set_ylabel('ln(L)')
    maxlnlike = np.max(sampler.lnprobability)
    axes[-1].set_ylim(maxlnlike-3*ndim, maxlnlike)
    fig.tight_layout(h_pad=0.)

    filename_pre = 'newModel_1/'+distance+'/line-time_walker%dNrun%dNburn%d_withscatter_'\
                   +age+'Gyr_'+distance+'_%dstars_newModel_1'
    filename = filename_pre % (N_WALKERS,Nrun,Nburn,Number)
    fig.savefig(filename+'.png')


    # Make a triangle plot
    burnin = Nburn
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    #convert scatters to exp(scatters)
    #samples[:,-3] = np.exp(samples[:,-3])
    #samples[:,-2] = np.exp(samples[:,-2])
    #samples[:,-1] = np.exp(samples[:,-1])

    fig = corner.corner(samples[:,:-3],
                        labels=_list_labels,
                        label_kwargs={'fontsize': 20},
                        # truths=_list_answer,
                        quantiles=[0.16, 0.5, 0.84],
                        plot_datapoints=True,
                        show_titles=True,
                        title_args={'fontsize': 20},
                        title_fmt='.3f',
                        )

    filename_pre = 'newModel_1/'+distance+'/trinagle_walker%dNrun%dNburn%d_withscatter_'\
                   +age+'Gyr_'+distance+'_%dstars_newModel_1'
    filename = filename_pre % (N_WALKERS,Nrun,Nburn,Number)
    fig.savefig(filename+'.png')
    #fig.savefig(filename+'.pdf')

    p = np.mean(samples, axis=0)
    e = np.var(samples, axis=0)**0.5
    filename = 'newModel_1/result_'+age+'Gyr_'+distance+'_'+str(Number)+'stars_newModel_1'+'.txt'
    np.savetxt(filename,(p,e),fmt="%.3f",delimiter=',')

    va,vR2,sigmaphi,meanvR,sigmaR =\
    lnlike(p,cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number)[1],\
    lnlike(p,cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number)[2],\
    lnlike(p,cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number)[3],\
    lnlike(p,cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number)[4],\
    lnlike(p,cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,data_pmb,\
           data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,age_number)[5]

    f = open('va_newModel_1.txt','a')
    printline = '%s, %f, %f, %f, %f, %f, %f\n' % (filename,np.mean(va),vR2,sigmaphi,np.mean(sigmaR)/80.,meanvR,sigmaR)
    f.write(printline)
    f.close()

    va_list = []
    sigmaR = []

    print(filename)
          
    return None

def read_data(filename):
    # read data, calculate velocity errors
    data = Table.read(filename)

    log10_age = np.array(data['log10_age'])
    l = np.array(data['l'])
    b = np.array(data['b'])
    plx = np.array(data['par'])
    err_plx = np.array(data['par_err'])
    pml = np.array(data['mu_l'])
    pmb = np.array(data['mu_b'])
    err_pml = np.array(data['mu_l_err'])
    err_pmb = np.array(data['mu_b_err'])
    vlos = np.array(data['vlos'])
    err_vlos = np.array(data['vlos_err'])
    duplicated = np.array(data['duplicated'])
    flag = np.array(data['flag'])
    z = np.array(data['z'])
    Z = np.array(data['Z'])

    idx = (plx/err_plx > 5.)\
          &(duplicated == 0)\
          &(flag == 0)\
          &np.isfinite(log10_age)\
          &np.isfinite(pml)\
          &np.isfinite(vlos)\
          &np.isfinite(l)\
          &np.isfinite(err_vlos)\
          &np.isfinite(err_pml)\
          &np.isfinite(err_pmb)\
          &(Z > -0.2)\
          &(np.absolute(z) < 0.1)

    log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos =\
        log10_age[idx],l[idx],b[idx],pml[idx],err_pml[idx],\
        pmb[idx],err_pmb[idx],plx[idx],err_plx[idx],vlos[idx],err_vlos[idx]

    vl = pml/plx*_k
    vb = pmb/plx*_k
    v = (vlos**2. + vl**2. + vb**2.)**0.5
    err_vl = ((plx*err_pml)**2. + (pml*err_plx)**2.)**0.5
    err_vb = ((plx*err_pmb)**2. + (pmb*err_plx)**2.)**0.5
    err_v = ((vl*err_vl)**2. + (vb*err_vb)**2. + (vlos*err_vlos)**2.)**0.5 /v

    idx = (err_v < 3.)
    log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos =\
        log10_age[idx],l[idx],b[idx],pml[idx],err_pml[idx],\
        pmb[idx],err_pmb[idx],plx[idx],err_plx[idx],vlos[idx],err_vlos[idx]

    Number = len(l)

    return log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,Number

def select_data(distance_number,log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,age_low,age_high):
    #select samples using stellar age
    idx = (age_low <= 10.**log10_age)\
          &(10.**log10_age < age_high)\
          &(1./plx <= (distance_number+1)/10.)
    
    age = 10.**log10_age
    l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,age =\
      l[idx],b[idx],pml[idx],err_pml[idx],pmb[idx],err_pmb[idx],plx[idx],err_plx[idx],vlos[idx],err_vlos[idx],age[idx]

    return l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,age

def time_record(time, Nrun, Nburn, Number):
    f = open('time.txt','a')
    printline = '%s %.2fmin %d %d %s %d\n' % (datetime.datetime.now(), time, Nrun, Nburn, 'mpi', Number)
    f.write(printline)
    f.close()

def MCMC(filename,log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,\
         age_number,distance_number,age_low,age_high):
    l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,data_age =\
        select_data(distance_number,log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,age_low,age_high)

    Number = len(l)
    print('N = ',Number)

    cos_twol = np.cos(2.*l)
    sin_twol = np.sin(2.*l)
    sin_l = np.sin(l)
    cos_l = np.cos(l)
    sin_b = np.sin(b)
    cos_b = np.cos(b)
    data_pml = pml
    data_err_pml = err_pml
    data_pmb = pmb
    data_err_pmb = err_pmb
    data_plx = plx
    data_err_plx = err_plx
    data_vlos = vlos
    data_err_vlos = err_vlos

    run_mcmc(cos_twol,sin_twol,sin_l,cos_l,sin_b,cos_b,data_pml,data_err_pml,\
             data_pmb,data_err_pmb,data_plx,data_err_plx,data_vlos,data_err_vlos,data_age,\
             age_number,distance_number,Number)

    return Number

def main():
    start = time.time()
    filename = '../../../gaia_spectro.hdf5'

    #9 means D<1kpc, 8 means D<900pc,,
    work_list = [9,8,7,6,5]

    #loop for distances
    for j in range(len(work_list)):
        distance_number = work_list[j]

        log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,Number =\
           read_data(filename)

        #loop for stellar ages
        for i in range(len(age_list)):
            Number = MCMC(filename,log10_age,l,b,pml,err_pml,pmb,err_pmb,plx,err_plx,vlos,err_vlos,\
                          i,distance_number,i,i+1)
        elapsed_time = (time.time() - start)/60
        print ("elapsed_time:{0}".format(elapsed_time) + "[min]")
        time_record(elapsed_time, Nrun, Nburn, Number)

if __name__ == '__main__':
    main()
