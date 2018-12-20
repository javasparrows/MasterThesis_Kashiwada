import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm

k = 4.74047

def MockGenerate(N):
    r = np.random.rand(N)
    phi = np.random.rand(N)*2.*np.pi
    z = np.random.rand(N)*2. - 1.

    x = r**(1./3.) * (1-z**2.)**0.5 * np.cos(phi)
    y = r**(1./3.) * (1-z**2.)**0.5 * np.sin(phi)
    z = r**(1./3.) * z 

    D = (x**2. + y**2. + z**2.)**0.5
    b = np.arcsin(z/D) 
    l = []
    for i in range(N):
        if   x[i]>0 and y[i]>0:
            l.append(np.arctan(y[i]/x[i]))
        elif x[i]<0 and y[i]>0:
            l.append(np.arctan(y[i]/x[i]) + np.pi)
        elif x[i]<0 and y[i]<0:
            l.append(np.arctan(y[i]/x[i]) + np.pi)
        else:
            l.append(np.arctan(y[i]/x[i]) + 2.*np.pi)
    l = np.array(l)

    #b = np.full(N, 0.)
    #plx = np.random.rand(N)*100
    #D = 1./plx
    sin_l = np.sin(l)
    cos_l = np.cos(l)
    sin_twol = np.sin(2.*l)
    cos_twol = np.cos(2.*l)
    sin_b = np.sin(b)
    cos_b = np.cos(b)
    D_2d = D #* cos_b
    plx = 1./D

    plt.figure(figsize=(8,4))
    plt.scatter(l,D, s=0.1)
    plt.savefig('l-D.png')

    R = np.random.rand(N)
    RR = np.random.rand(N)
    plt.figure(figsize=(6,4))
    plt.hist(R,bins=100)
    plt.hist(RR,bins=100)
    plt.savefig('CheckRandom.png')

    A, B, C, K, u0, v0, w0 = 15.300,  -11.900, -3.200, -5.300, 11.100, 12.240, 7.250
    #A, B, C, K, u0, v0, w0 = 15.300,  -11.900, -3.200, -5.300, 11.100, 0., 7.250

    pml= (A*cos_twol-C*sin_twol+B)*cos_b+plx*(u0*sin_l-v0*cos_l)
    pmb= -(A*sin_twol+C*cos_twol+K)*sin_b*cos_b+sin_b*plx*(u0*cos_l+v0*sin_l)-w0*plx*cos_b
    vlos= (K + C*cos_twol + A*sin_twol)*cos_b**2/plx - ((u0*cos_l + v0*sin_l)*cos_b + w0*sin_b)

    ###### correlation
    plt.figure(figsize=(6,4))
    df = pd.DataFrame({'l': l,
                       'b': b,
                       'pml': pml,
                       'pmb': pmb,
                       'plx': plx,                            
                       'vlos': vlos})
    df_corr = df.corr()
    sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
    plt.savefig('correlation.png')


    vl = pml/plx
    vb = pmb/plx

    R0 = 8.5 # kpc
    R = (D_2d*D_2d + R0*R0 - 2.*D_2d*R0*cos_l)**0.5
    sin_a = D_2d/R*sin_l
    cos_a = (1.-sin_a**2.)**0.5

    ###### convert from astrometric to plar coordinates
    vx = -sin_l*vl - cos_l*sin_b*vb + cos_l*cos_b*vlos #+ u0
    vy =  cos_l*vl - sin_l*sin_b*vb + sin_l*cos_b*vlos #+ (v0+220)
    vz =                   cos_b*vb       + sin_b*vlos #+ w0

    vR =   cos_a*vx - sin_a*vy
    vphi = sin_a*vx + cos_a*vy
    vz =                        vz

    ###### rotation-curve.png
    plt.figure(figsize=(8,4))
    plt.scatter(R,vphi, s=0.1)
    plt.savefig('rotation-curve.png')

    ###### vRvphi.png
    plt.figure(figsize=(8,4))
    plt.scatter(vR,vphi, s=0.1)
    plt.savefig('vRvphi.png')

    ###### add velocity dispersion
    sigma_vR = 35. # km/s
    sigma_vphi = 15. # km/s
    sigma_vz = 5. # km/s
    vasym = 20. # km/s

    #sigma_vR = 0. # km/s
    #sigma_vphi = 0. # km/s
    #sigma_vz = 0. # km/s
    #vasym = 0. # km/s

    vR = vR + np.random.normal(0, sigma_vR, N)
    vphi = vphi - np.random.normal(vasym, sigma_vphi, N)
    vz = vz + np.random.normal(0, sigma_vz, N)

    plt.scatter(vR,vphi, s=0.1)
    plt.savefig('vRvphi_withAD.png')

    ###### histogram ######
    plt.figure(figsize=(6,8))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    plt.subplot(3,1,1)
    #plt.title('1e6 stars')
    plt.hist(vR, bins=200)
    plt.xlabel('vR (km/s)')

    plt.subplot(3,1,2)
    plt.hist(vphi, bins=200)
    plt.xlabel('vphi (km/s)')

    plt.subplot(3,1,3)
    plt.hist(vz, bins=200)
    plt.xlabel('vz (km/s)')

    plt.savefig('hist_v.png')

    ###### change coordinate
    vx =  cos_a*vR + sin_a*vphi #- u0
    vy = -sin_a*vR + cos_a*vphi #- (v0+220)
    vz =                     vz #- w0

    vl =        -sin_l*vx +       cos_l*vy
    vb =  -cos_l*sin_b*vx - sin_l*sin_b*vy + cos_b*vz
    vlos = cos_l*cos_b*vx + sin_l*cos_b*vy + sin_b*vz

    ###### hist_gal_v.png
    plt.figure(figsize=(6,8))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    plt.subplot(3,1,1)
    #plt.title('1e6 stars')
    plt.hist(vl, bins=200)
    plt.xlabel('vl (km/s)')

    plt.subplot(3,1,2)
    plt.hist(vb, bins=200)
    plt.xlabel('vb (km/s)')

    plt.subplot(3,1,3)
    plt.hist(vlos, bins=200)
    plt.xlabel('vlos (km/s)')

    plt.savefig('hist_gal_v.png')

    ######
    pml = vl/D_2d/k
    pmb = vb/D_2d/k

    l = l*180./np.pi
    b = b*180./np.pi

    pr_vl = norm.fit(vl)
    pr_vb = norm.fit(vb)
    pr_rv = norm.fit(vlos)

    print(pr_vl)
    print(pr_vb)
    print(pr_rv)

    ###### l-v.png
    plt.figure(figsize=(6,8))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    plt.subplot(3,1,1)
    #plt.title('1e6 stars')
    plt.scatter(l,vl, s=0.1)
    plt.xlabel('l (deg)')
    plt.ylabel('vl (km/s)')

    plt.subplot(3,1,2)
    plt.scatter(l,vb, s=0.1)
    plt.xlabel('l (deg)')
    plt.ylabel('vb (km/s)')

    plt.subplot(3,1,3)
    plt.scatter(l,vlos, s=0.1)
    plt.xlabel('l (deg)')
    plt.ylabel('vlos (km/s)')
    plt.savefig('l-v.png')

    ###### l-pm.png
    plt.figure(figsize=(12,6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(2,2,1)
    plt.scatter(l,pml, s=0.1)
    plt.xlabel('l (deg)')
    plt.ylabel('pml (mas/yr)')
    plt.ylim(-100, 100)

    plt.subplot(2,2,2)
    plt.scatter(b,pml, s=0.1)
    plt.xlabel('b (deg)')
    plt.ylabel('pml (mas/yr)')
    plt.ylim(-100, 100)

    plt.subplot(2,2,3)
    plt.scatter(l,pmb, s=0.1)
    plt.xlabel('l (deg)')
    plt.ylabel('pmb (mas/yr)')
    plt.ylim(-100, 100)

    plt.subplot(2,2,4)
    plt.scatter(b,pmb, s=0.1)
    plt.xlabel('b (deg)')
    plt.ylabel('pmb (mas/yr)')
    plt.ylim(-100, 100)

    plt.savefig('l-pm.png')

    error_plx = 0.1#0.3
    error_pml =  0.1#0.1
    error_pmb =  0.1#0.1
    error_vlos = 0.1#1.5
    #plx = plx + np.random.normal(0, error_plx, N)    
    #pml = pml + np.random.normal(0, error_pml, N)    
    #pmb = pmb + np.random.normal(0, error_pmb, N)    
    #vlos = vlos + np.random.normal(0, error_vlos, N)
    error_plx = np.full(N,error_plx)
    error_pml = np.full(N,error_pml)
    error_pmb = np.full(N,error_pmb)
    error_vlos = np.full(N,error_vlos)

    data = np.array([plx,l,b,pml,pmb,vlos,error_plx,error_pml,error_pmb,error_vlos]).T
    np.savetxt('Mock_Kashiwada_AsymmetricDrift.dat',data)

    return plx,l,b,pml,pmb,vlos,error_plx,error_pml,error_pmb,error_vlos


if __name__ == '__main__':
    MockGenerate(10000) 
