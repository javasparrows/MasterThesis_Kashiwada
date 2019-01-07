import numpy as np

k = 4.74047

Rd = 3    #kpc
zd = 0.3  #kpc
R0 = 8.2 # kpc

p_list = [18.,-11,-2.,-1.,9.,11.,7.5]
generate_number_list = [100,1000*3,10000,100000]
stars_number_list = ['1e2','1e3','1e4','1e5']
sigma_list = ['sigma1','sigma2','sigma3','sigma4','sigma5']
VE_list = [0,10,20,30,40]

#velocity dispersion for stellar ages from Yu&Liu(2018)
sigma_array = np.array([
[21.7,12.0,8.6],
[21.4,16.7,10.1],
[27.8,18.9,10.5],
[32.7,18.4,11.0],
[31.3,16.8,11.9],
[30.1,16.9,11.7],
[34.7,17.8,12.6],
[36.8,21.2,16.8],
[39.3,22.1,17.7],
[42.5,23.0,18.3],
[43.8,24.2,23.3],
[51.8,25.8,23.3]])

def MockGenerate(N,sigma_number,stars_number,data_number,lv_deg):
    '''
    p_list = [[17.0,0,0,0,0,0,0], 
              [0,-12.0,0,0,0,0,0], 
              [0,0,-3.0,0,0,0,0], 
              [0,0,0,-2.0,0,0,0], 
              [0,0,0,0,9.0,0,0], 
              [0,0,0,0,0,14.0,0], 
              [0,0,0,0,0,0,7.0]] 
    '''
    A,B,C,K,u0,v0,w0 = p_list
    sigma_vR,sigma_vphi,sigma_vz = sigma_array[sigma_number]
    #A,B,C,K,u0,v0,w0,sigma_pml,sigma_pmb,sigma_rv = p_list[sigma_number]

    ######## generate random in the sphere
    R = np.random.rand(N)
    phi = np.random.rand(N)*2.*np.pi
    z = np.random.rand(N)*2. - 1.

    x = R**(1./3.) * (1.-z**2.)**0.5 * np.cos(phi)
    y = R**(1./3.) * (1.-z**2.)**0.5 * np.sin(phi)
    z = R**(1./3.) * z 

    za = np.absolute(z)
    rho = np.exp(-R/Rd)*np.exp(-za/zd)
    rho_Standardization = np.exp(-R/Rd)*np.exp(-za/zd)

    AdoptionRate = rho/np.amax(rho_Standardization)

    index = (np.random.rand(N) < AdoptionRate)

    x = x[index]
    y = y[index]
    z = z[index]
    N = np.sum(index)

    D = (x**2. + y**2. + z**2.)**0.5
    b = np.arcsin(z/D) 
    l = []
    for i in range(N):
        if   x[i]>0. and y[i]>0.:
            l.append(np.arctan(y[i]/x[i]))
        elif x[i]<0. and y[i]>0.:
            l.append(np.arctan(y[i]/x[i]) + np.pi)
        elif x[i]<0. and y[i]<0.:
            l.append(np.arctan(y[i]/x[i]) + np.pi)
        else:
            l.append(np.arctan(y[i]/x[i]) + 2.*np.pi)
    l = np.array(l)

    index = (0.1 < D)

    D = D[index]
    l = l[index]
    b = b[index]
    N = np.sum(index)

    #b = np.full(N, 0.)
    #plx = np.random.rand(N)*100
    #D = 1./plx
    sin_l = np.sin(l)
    cos_l = np.cos(l)
    sin_twol = np.sin(2.*l)
    cos_twol = np.cos(2.*l)
    sin_b = np.sin(b)
    cos_b = np.cos(b)
    plx = 1./D

    pml_analytic = (A*cos_twol-C*sin_twol+B)*cos_b+plx*(u0*sin_l-v0*cos_l)
    pmb_analytic = -(A*sin_twol+C*cos_twol+K)*sin_b*cos_b+sin_b*plx*(u0*cos_l+v0*sin_l)-w0*plx*cos_b
    rv_analytic = (K + C*cos_twol + A*sin_twol)*cos_b**2/plx - ((u0*cos_l + v0*sin_l)*cos_b + w0*sin_b)

    pml,pmb,rv = pml_analytic,pmb_analytic,rv_analytic

    vl = pml/plx
    vb = pmb/plx

    R = (D*D + R0*R0 - 2.*D*R0*cos_l)**0.5
    sin_a = D/R*sin_l
    cos_a = (1.-sin_a**2.)**0.5

    vcirc = R0*(A-B)

    ###### convert from astrometric to plar coordinates
    vx = -sin_l*vl - cos_l*sin_b*vb + cos_l*cos_b*rv + u0
    vy =  cos_l*vl - sin_l*sin_b*vb + sin_l*cos_b*rv + (v0+vcirc)
    vz =                   cos_b*vb       + sin_b*rv + w0
    
    vR =   cos_a*vx - sin_a*vy
    vphi = sin_a*vx + cos_a*vy
    vz =                        vz

    ###### add velocity dispersion
    sigma_vR_value = np.random.normal(0., sigma_vR, N)
    vR = vR + sigma_vR_value - np.mean(sigma_vR_value)
    #vasym = np.mean(vR**2.)/80. # km/s
    #vasym = np.var(vR)/80. # km/s
    #print('vasym = ',np.round(np.mean(vasym),3))

    sigma_vphi_value = np.random.normal(0., sigma_vphi, N)
    vphi = vphi + sigma_vphi_value - np.mean(sigma_vphi_value)# - vasym
    sigma_vz_value = np.random.normal(0., sigma_vz, N)
    vz = vz + sigma_vz_value - np.mean(sigma_vz_value)

    ######### Velocity Ellipsoid #########
    lv = lv_deg*np.pi/180.
    sigma_vR_value_VE = np.cos(lv)*sigma_vR_value - np.sin(lv)*sigma_vphi_value
    sigma_vphi_value_VE = np.sin(lv)*sigma_vR_value + np.cos(lv)*sigma_vphi_value

    vR = vR + sigma_vR_value_VE
    vphi = vphi + sigma_vphi_value_VE 
    vz = vz + sigma_vz_value

    #vasym = np.mean(vR**2.)/80. # km/s
    #vphi = vphi - vasym

    ###### change coordinate
    vx =  cos_a*vR + sin_a*vphi - u0
    vy = -sin_a*vR + cos_a*vphi - (v0+vcirc)
    vz =                     vz - w0

    vl =       -sin_l*vx +       cos_l*vy
    vb = -cos_l*sin_b*vx - sin_l*sin_b*vy + cos_b*vz
    rv =  cos_l*cos_b*vx + sin_l*cos_b*vy + sin_b*vz

    ######
    pml = vl/D#/k
    pmb = vb/D#/k
    l = l*180./np.pi
    b = b*180./np.pi

    l_a = np.linspace(0.,2.*np.pi,2000)
    b_a = np.random.rand(2000)*np.pi-np.pi/2.
    sin_l_a = np.sin(l_a)
    cos_l_a = np.cos(l_a)
    sin_twol_a = np.sin(2.*l_a)
    cos_twol_a = np.cos(2.*l_a)
    sin_b_a = np.sin(b_a)
    cos_b_a = np.cos(b_a)
    D_a = np.random.rand(2000)*0.9 + 0.1
    plx_a = 1./D_a

    pml_a = (A*cos_twol_a-C*sin_twol_a+B)*cos_b_a+plx_a*(u0*sin_l_a-v0*cos_l_a)
    pmb_a = -(A*sin_twol_a+C*cos_twol_a+K)*sin_b_a*cos_b_a+sin_b_a*plx_a*(u0*cos_l_a+v0*sin_l_a)-w0*plx_a*cos_b_a
    rv_a = (K + C*cos_twol_a + A*sin_twol_a)*cos_b_a**2/plx_a - ((u0*cos_l_a + v0*sin_l_a)*cos_b_a + w0*sin_b_a)

    l_a = l_a*180./np.pi
    b_a = b_a*180./np.pi

    idx_b = np.argsort(b_a)

    pml = pml/k
    pmb = pmb/k
    error_plx = 0.01#0.3
    error_pml =  0.01#0.1
    error_pmb =  0.01#0.1
    error_rv = 0.01#1.5
    plx = plx + np.random.normal(0, error_plx, N)    
    pml = pml + np.random.normal(0, error_pml, N)    
    pmb = pmb + np.random.normal(0, error_pmb, N)    
    rv = rv + np.random.normal(0, error_rv, N)
    error_plx = np.full(N,error_plx)
    error_pml = np.full(N,error_pml)
    error_pmb = np.full(N,error_pmb)
    error_rv = np.full(N,error_rv)

    data = np.array([plx,l,b,pml,pmb,rv,error_plx,error_pml,error_pmb,error_rv]).T
    filename = 'data/MockData_'+'sigma'+str(sigma_number+1)+'_'+stars_number_list[stars_number]\
              +'stars_'+'VE'+str(lv_deg)+'_'+str(data_number)+'.dat'
    np.savetxt(filename,data)

    vRvphi_sigma = np.mean((vR-np.mean(vR))*(vphi-np.mean(vphi)))
    vRvphi_tilt = np.arctan(2.*vRvphi_sigma/(np.var(vR)-np.var(vphi)))/2. *180./np.pi
    print(vRvphi_tilt)
    print(len(pml))

    return plx,l,b,pml,pmb,rv,error_plx,error_pml,error_pmb,error_rv

def main():
    for m in range(10):
        for j in range(4):
            for ii in range(len(VE_list)):
                for i in range(len(sigma_array)):
                    N = generate_number_list[j]
                    MockGenerate(N,i,j,m,VE_list[ii])

if __name__ == '__main__':
    main()
