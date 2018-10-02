from numpy import *
from pylab import *
from numpy.random import *

## ESTIMATION BOUNDS

class Bounds(object):

    def __init__(self,signal_model,name="CRB"):
        self.signal_model = signal_model
        self.name=name

    def __str__(self):
        return self.name

class Exact_Bounds(Bounds):

    def compute(self,method=0):
    
        """ Compute the Cramer Rao bound """
        CRB=zeros(10)
        
        # extract parameter
        N=self.signal_model.N
        w0=self.signal_model.w0
        c=self.signal_model.c
        sigma2=self.signal_model.sigma2

        if method==0:
            #Closed form inversion of the Fisher Inversion Matrix
            
            #precompute values
            n_vect=arange(N)
            n_vect2=n_vect**2
            q=sum(exp(2j*w0*n_vect))
            beta=1/((N**2)-(abs(q)**2))
            
            den=0;
            for num_phase in range(0,3):
                ck=c[num_phase]
                ak=abs(ck)
                phik=angle(ck)
                
                f1k=sum(n_vect*(1-exp(2j*(n_vect*w0+phik))))
                f2k=sum(n_vect2*(1-exp(2j*(n_vect*w0+phik))))
                
                den=den+(ak**2)*real(f2k-N*beta*(abs(f1k)**2)-beta*conj(q)*(f1k**2)*exp(-2j*phik))  # angular frequency
                CRB[2*num_phase]=(beta**2)*imag((N-conj(q))*f1k*conj(ck))**2                        # real part of phasor
                CRB[2*num_phase+1]=(beta**2)*real((N+conj(q))*f1k*conj(ck))**2                      # imag part of phasor
            
            CRB[6]=2*sigma2/den  # Need to be computed first
            CRB[:6:2]=2*beta*sigma2*(N-real(q))+CRB[6]*CRB[:6:2]
            CRB[1:6:2]=2*beta*sigma2*(N+real(q))+CRB[6]*CRB[1:6:2]
        
        
        else:
            #Numerical inversion of the Fisher Inversion Matrix using numpy
            C=vstack((real(c),imag(c)))
            c_vect=matrix(ravel(C.T)).T
            n_vect=arange(0,N)
            n_vect2=n_vect**2
            
            #construct matrix A and Q
            a=matrix(exp(1j*w0*n_vect)).T
            b=matrix(n_vect*exp(1j*w0*n_vect)).T
            A=hstack((real(a),-imag(a)))
            Q=hstack((-imag(b),-real(b)))
            
            #construct Fisher Information Matrix
            F1=kron(eye(3),A)
            f2=kron(eye(3),Q)*c_vect
            F=(1/sigma2)*bmat([[F1.T*F1, F1.T*f2], [f2.T*F1, f2.T*f2]])
            CRB[:7]=diag(linalg.inv(F))     #Computation of the CRB


        # compute TVE2
        a_vect=abs(c)
        CRB[7:]=(CRB[:6:2]+CRB[1:6:2])/(a_vect**2)

        return CRB


class Approximated_Bounds(Bounds):

    def compute(self):
    
        """ Compute the CRB approximation when N>>1"""
        CRB=zeros(10)
        
        # extract parameter
        N=self.signal_model.N
        c=self.signal_model.c
        sigma2=self.signal_model.sigma2
        
        a_vect=abs(c)
        eta=sum(a_vect**2)/(6*sigma2)
        
        #Compute CRB
        for num_phase in range(0,3):
            ck=c[num_phase]
            CRB[2*num_phase]=2*(sigma2/N)+(imag(ck)**2)/(eta*N)
            CRB[2*num_phase+1]=2*(sigma2/N)+(real(ck)**2)/(eta*N)

        CRB[6]=4/(eta*(N**3))
        
        # Compute TVE2
        etak=(a_vect**2)/(2*sigma2)
        CRB[7:]=(1/N)*((2/etak)+(1/eta))
    
        return CRB



