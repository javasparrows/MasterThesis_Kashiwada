#include <stdio.h>
#include <iostream>
#include <math.h>

double fr(double r,double phi,double z,double vr,double vphi,double vz);
double fvr(double r,double phi,double z,double vr,double vphi,double vz);
double fphi(double r,double phi,double z,double vr,double vphi,double vz);
double fvphi(double r,double phi,double z,double vr,double vphi,double vz);
double fz(double r,double phi,double z,double vr,double vphi,double vz);
double fvz(double r,double phi,double z,double vr,double vphi,double vz);

double pc_in_km = 3.085e+13; //km
double vr0=8.67,vphi0=220.+11.4,vz0=7.5;  //km/s
double v0=pow(vr0*vr0 + vphi0*vphi0 + vz0*vz0, 0.5); //km/s
double rc=3e3*pc_in_km; // km
double qz=0.2;

int main()
{
        double t,r,phi,z,vr,vphi,vz,dt,tmax;
        double k1[6],k2[6],k3[6],k4[6];
        r=8e3*pc_in_km;        //位置の初期値 km
        phi=0.0;
        z = 0.0*pc_in_km;  //km
        vr=vr0;   //km/s        //速度の初期値
        vphi=vphi0; 
        vz=vz0; 
        double year_in_sec = 31536000;
        tmax=46e+8*year_in_sec;  //繰り返し最大回数
        dt=1e+3*year_in_sec;                //刻み幅
        double E, Lz;

        FILE *output;
        output=fopen("output.dat","w");

        for(t=0;t<tmax;t+=dt) {
                k1[0]=dt*fr(r,phi,z,vr,vphi,vz);
                k1[1]=dt*fphi(r,phi,z,vr,vphi,vz);
                k1[2]=dt*fz(r,phi,z,vr,vphi,vz);
                k1[3]=dt*fvr(r,phi,z,vr,vphi,vz);
                k1[4]=dt*fvphi(r,phi,z,vr,vphi,vz);
                k1[5]=dt*fvz(r,phi,z,vr,vphi,vz);

                k2[0]=dt/2.*fr(r+k1[0]/2.0,phi+k1[1]/2.0,
                               z+k1[2]/2.0,vr+k1[3]/2.0,
                               vphi+k1[4]/2.0,vz+k1[5]/2.0);
                k2[1]=dt/2.*fphi(r+k1[0]/2.0,phi+k1[1]/2.0,
                                 z+k1[2]/2.0,vr+k1[3]/2.0,
                                 vphi+k1[4]/2.0,vz+k1[5]/2.0);
                k2[2]=dt/2.*fz(r+k1[0]/2.0,phi+k1[1]/2.0,
                                 z+k1[2]/2.0,vr+k1[3]/2.0,
                                 vphi+k1[4]/2.0,vz+k1[5]/2.0);
                k2[3]=dt/2.*fvr(r+k1[0]/2.0,phi+k1[1]/2.0,
                                z+k1[2]/2.0,vr+k1[3]/2.0,
                                vphi+k1[4]/2.0,vz+k1[5]/2.0);
                k2[4]=dt/2.*fvphi(r+k1[0]/2.0,phi+k1[1]/2.0,
                                  z+k1[2]/2.0,vr+k1[3]/2.0,
                                   vphi+k1[4]/2.0,vz+k1[5]/2.0);
                k2[5]=dt/2.*fvz(r+k1[0]/2.0,phi+k1[1]/2.0,
                                z+k1[2]/2.0,vr+k1[3]/2.0,
                                vphi+k1[4]/2.0,vz+k1[5]/2.0);

                k3[0]=dt/2.*fr(r+k2[0]/2.0,phi+k2[1]/2.0,
                               z+k2[2]/2.0,vr+k2[3]/2.0,
                               vphi+k2[4]/2.0,vz+k2[5]/2.0);
                k3[1]=dt/2.*fphi(r+k2[0]/2.0,phi+k2[1]/2.0,
                                 z+k2[2]/2.0,vr+k2[3]/2.0,
                                 vphi+k2[4]/2.0,vz+k2[5]/2.0);
                k3[2]=dt/2.*fz(r+k2[0]/2.0,phi+k2[1]/2.0,
                               z+k2[2]/2.0,vr+k2[3]/2.0,
                               vphi+k2[4]/2.0,vz+k2[5]/2.0);
                k3[3]=dt/2.*fvr(r+k2[0]/2.0,phi+k2[1]/2.0,
                                z+k2[2]/2.0,vr+k2[3]/2.0,
                                vphi+k2[4]/2.0,vz+k2[5]/2.0);
                k3[4]=dt/2.*fvphi(r+k2[0]/2.0,phi+k2[1]/2.0,
                                  z+k2[2]/2.0,vr+k2[3]/2.0,
                                  vphi+k2[4]/2.0,vz+k2[5]/2.0);
                k3[5]=dt/2.*fvz(r+k2[0]/2.0,phi+k2[1]/2.0,
                                z+k2[2]/2.0,vr+k2[3]/2.0,
                                vphi+k2[4]/2.0,vz+k2[5]/2.0);

                k4[0]=dt*fr(r+k3[0],phi+k3[1],
                            z+k3[2],vr+k3[3],
                            vphi+k3[4],vz+k3[5]);
                k4[1]=dt*fphi(r+k3[0],phi+k3[1],
                              z+k3[2],vr+k3[3],
                              vphi+k3[4],vz+k3[5]);
                k4[2]=dt*fz(r+k3[0],phi+k3[1],
                            z+k3[2],vr+k3[3],
                            vphi+k3[4],vz+k3[5]);
                k4[3]=dt*fvr(r+k3[0],phi+k3[1],
                             z+k3[2],vr+k3[3],
                             vphi+k3[4],vz+k3[5]);
                k4[4]=dt*fvphi(r+k3[0],phi+k3[1],
                               z+k3[2],vr+k3[3],
                               vphi+k3[4],vz+k3[5]);
                k4[5]=dt*fvz(r+k3[0],phi+k3[1],
                             z+k3[2],vr+k3[3],
                             vphi+k3[4],vz+k3[5]);

                r+=(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])/6.0;
                phi+=(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0;
                z+=(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6.0;
                vr+=(k1[3]+2.0*k2[3]+2.0*k3[3]+k4[3])/6.0;
                vphi+=(k1[4]+2.0*k2[4]+2.0*k3[4]+k4[4])/6.0;
                vz+=(k1[5]+2.0*k2[5]+2.0*k3[5]+k4[5])/6.0;

                E = 0.5*(vr*vr + vphi*vphi + vz*vz) 
                  - 0.5*v0*v0*log(pow(rc,2) + pow(r,2) + pow(z/qz,2));

                Lz = r/1e3/pc_in_km*vphi;

                fprintf(output,"%f %f %f %f %f %f %f %f %f\n",
                   t/(1e+6*year_in_sec),r/1e3/pc_in_km,phi,z/pc_in_km,vr,vphi,vz,E,Lz);

        }

        fclose(output);

        return 0;
}

double fr(double r,double phi,double z,double vr,double vphi,double vz)
{
        return vr;
}

double fphi(double r,double phi,double z,double vr,double vphi,double vz)
{
        return vphi/r;
}

double fz(double r,double phi,double z,double vr,double vphi,double vz)
{
        return vz;
}

double fvr(double r,double phi,double z,double vr,double vphi,double vz)
{
        return vphi*vphi/r -v0*v0*r/(rc*rc + r*r + z*z/qz/qz);
}

double fvphi(double r,double phi,double z,double vr,double vphi,double vz)
{
        return -vr*vphi/r;
}

double fvz(double r,double phi,double z,double vr,double vphi,double vz)
{
        return -v0*v0*z/qz/qz/(rc*rc + r*r + z*z/qz/qz);
}