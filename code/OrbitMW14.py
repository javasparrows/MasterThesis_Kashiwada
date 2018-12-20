import numpy as np
from galpy.potential import MWPotential2014
from galpy.potential import evaluatePotentials
from galpy.potential import evaluateRforces
from galpy.potential import evaluatezforces
from galpy.util import bovy_conversion
import time
from numba import jit

pc_in_km = 3.085e+13 #km
vr0=8.67
vphi0=220.+11.4
vz0=7.5  #km/s
v0=(vr0**2 + vphi0**2 + vz0**2)**0.5 #km/s
year_in_sec = 31536000
rc=3e3*pc_in_km  #km
qz=0.2

def RK():
    r=8e3*pc_in_km        #位置の初期値 km
    phi=0.0
    z = 0.0*pc_in_km  #km
    vr=vr0                #km/s    #速度の初期値
    vphi=vphi0 
    vz=vz0 
    tmax=46e+8*year_in_sec  #繰り返し最大回数
    dt=1e+3*year_in_sec                #刻み幅

    t_list,r_list,phi_list,z_list,vr_list,vphi_list,vz_list,E_list,Lz_list =\
    [],[],[],[],[],[],[],[],[]

    for i in range(0,int(tmax/dt)):
        t = dt*i

        k10=dt*fr(r,phi,z,vr,vphi,vz)
        k11=dt*fphi(r,phi,z,vr,vphi,vz)
        k12=dt*fz(r,phi,z,vr,vphi,vz)
        k13=dt*fvr(r,phi,z,vr,vphi,vz)
        k14=dt*fvphi(r,phi,z,vr,vphi,vz)
        k15=dt*fvz(r,phi,z,vr,vphi,vz)

        k20=dt/2.*fr(r+k10/2.,phi+k11/2.,
                       z+k12/2.,vr+k13/2.,
                       vphi+k14/2.,vz+k15/2.)
        k21=dt/2.*fphi(r+k10/2.,phi+k11/2.,
                         z+k12/2.,vr+k13/2.,
                         vphi+k14/2.,vz+k15/2.)
        k22=dt/2.*fz(r+k10/2.,phi+k11/2.,
                         z+k12/2.,vr+k13/2.,
                         vphi+k14/2.,vz+k15/2.)
        k23=dt/2.*fvr(r+k10/2.,phi+k11/2.,
                        z+k12/2.,vr+k13/2.,
                        vphi+k14/2.,vz+k15/2.)
        k24=dt/2.*fvphi(r+k10/2.,phi+k11/2.,
                          z+k12/2.,vr+k13/2.,
                           vphi+k14/2.,vz+k15/2.)
        k25=dt/2.*fvz(r+k10/2.,phi+k11/2.,
                        z+k12/2.,vr+k13/2.,
                        vphi+k14/2.,vz+k15/2.)

        k30=dt/2.*fr(r+k20/2.,phi+k21/2.,
                       z+k22/2.,vr+k23/2.,
                       vphi+k24/2.,vz+k25/2.)
        k31=dt/2.*fphi(r+k20/2.,phi+k21/2.,
                         z+k22/2.,vr+k23/2.,
                         vphi+k24/2.,vz+k25/2.)
        k32=dt/2.*fz(r+k20/2.,phi+k21/2.,
                       z+k22/2.,vr+k23/2.,
                       vphi+k24/2.,vz+k25/2.)
        k33=dt/2.*fvr(r+k20/2.,phi+k21/2.,
                        z+k22/2.,vr+k23/2.,
                        vphi+k24/2.,vz+k25/2.)
        k34=dt/2.*fvphi(r+k20/2.,phi+k21/2.,
                          z+k22/2.,vr+k23/2.,
                          vphi+k24/2.,vz+k25/2.)
        k35=dt/2.*fvz(r+k20/2.,phi+k21/2.,
                        z+k22/2.,vr+k23/2.,
                        vphi+k24/2.,vz+k25/2.)

        k40=dt*fr(r+k30,phi+k31,
                  z+k32,vr+k33,
                  vphi+k34,vz+k35)
        k41=dt*fphi(r+k30,phi+k31,
                    z+k32,vr+k33,
                    vphi+k34,vz+k35)
        k42=dt*fz(r+k30,phi+k31,
                  z+k32,vr+k33,
                  vphi+k34,vz+k35)
        k43=dt*fvr(r+k30,phi+k31,
                   z+k32,vr+k33,
                   vphi+k34,vz+k35)
        k44=dt*fvphi(r+k30,phi+k31,
                     z+k32,vr+k33,
                     vphi+k34,vz+k35)
        k45=dt*fvz(r+k30,phi+k31,
                   z+k32,vr+k33,
                   vphi+k34,vz+k35)

        r+=(k10+2.*k20+2.*k30+k40)/6.0
        phi+=(k11+2.*k21+2.*k31+k41)/6.0
        z+=(k12+2.*k22+2.*k32+k42)/6.0
        vr+=(k13+2.*k23+2.*k33+k43)/6.0
        vphi+=(k14+2.*k24+2.*k34+k44)/6.0
        vz+=(k15+2.*k25+2.*k35+k45)/6.0

        E = 0.5*(vr**2. + vphi**2. + vz**2.) + evaluatePotentials(MWPotential2014,R=r,z=z)

        Lz = r*vphi

        t_list.append(t)
        r_list.append(r)
        phi_list.append(phi)
        z_list.append(z)
        vr_list.append(vr)
        vphi_list.append(vphi)
        vz_list.append(vz)
        E_list.append(E)
        Lz_list.append(Lz)

    t_array = np.array([t_list])[0,:]
    r_array = np.array([r_list])[0,:]
    phi_array = np.array([phi_list])[0,:]
    z_array = np.array([z_list])[0,:]
    vr_array = np.array([vr_list])[0,:]
    vphi_array = np.array([vphi_list])[0,:]
    vz_array = np.array([vz_list])[0,:]
    E_array = np.array([E_list])[0,:]
    Lz_array = np.array([Lz_list])[0,:]

    data = np.array([t_array/(1e6*year_in_sec),r_array/1e3/pc_in_km,phi_array,\
                    z_array/pc_in_km,vr_array,vphi_array,vz_array,E_array,Lz_array]).T
    np.savetxt('output_MW14.dat', data)

@jit('float32(float32,float32,float32,float32,float32,float32)')
def fr(r,phi,z,vr,vphi,vz):
    return vr

@jit('float32(float32,float32,float32,float32,float32,float32)')
def fphi(r,phi,z,vr,vphi,vz):
    return vphi/r

@jit('float32(float32,float32,float32,float32,float32,float32)')
def fz(r,phi,z,vr,vphi,vz):
    return vz

@jit('float32(float32,float32,float32,float32,float32,float32)')
def fvr(r,phi,z,vr,vphi,vz):
    force_r = vphi**2./r + evaluateRforces(MWPotential2014,R=r/1e3/pc_in_km/8.,z=z/1e3/pc_in_km)*bovy_conversion.force_in_kmsMyr(220.,8.)/(1e6*year_in_sec)
    return force_r

@jit('float32(float32,float32,float32,float32,float32,float32)')
def fvphi(r,phi,z,vr,vphi,vz):
    #return 0
    return -vr*vphi/r

@jit('float32(float32,float32,float32,float32,float32,float32)')
def fvz(r,phi,z,vr,vphi,vz):
    force_z =  evaluatezforces(MWPotential2014,R=r/1e3/pc_in_km/8.,z=z/1e3/pc_in_km)*bovy_conversion.force_in_kmsMyr(220.,8.)/(1e6*year_in_sec)
    return force_z

if __name__ == '__main__':
    start = time.time()

    RK()

    elapsed_time = (time.time() - start)/60
    print ("elapsed_time:{0}".format(elapsed_time) + "[min]")