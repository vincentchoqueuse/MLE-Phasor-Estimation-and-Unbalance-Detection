from numpy import *
from pylab import *
from numpy.random import *

class Phasor_Estimator(object):

    def __init__(self,w_init=0.5,nb_iter=10,verbose=0,name="ML estimator"):
        self.w_init=w_init
        self.nb_iter=nb_iter
        self.verbose=verbose
        self.name=name

    def __str__(self):
        return self.name

class ML_estimator_approx(Phasor_Estimator):
    
    def estimate(self,signal):

        #Frequency estimation
        w=self.w_init
        N=signal.shape[1]
        
        g=zeros((3*(N-2),1))
        p=zeros((3*(N-2),1))
        alpha=cos(w)
        """
        for num_phase in range(3):
            s=signal[num_phase,:]
            g_temp=s[2:]+s[:-2]
            p_temp=2*s[1:-1]

            p[num_phase*(N-2):(num_phase+1)*(N-2),0]=p_temp
            g[num_phase*(N-2):(num_phase+1)*(N-2),0]=g_temp
        
        p=mat(p)
        g=mat(g)
       
        for iter in range(self.nb_iter):
            # construct B
            B=zeros((3*(N-2),3*N))
            for i in range(1,N-1):
                B[(i-1)*3:i*3,:]=bmat([zeros((3,3*(i-1))),eye(3),-2*alpha*eye(3),eye(3),zeros((3,3*(N-i-2)))])

            B=mat(B)
            
            R=inv(B*B.T)
            alpha=ravel(inv(p.T*R*p)*p.T*R*g)[0]
            alpha=max(alpha,-1)
            alpha=min(alpha,1)
            w=arccos(alpha)
        """

        num=0
        den=0
        for num_phase in range(3):
            for n in range(2,N-1):
                num=num+signal[num_phase,n-1]*(signal[num_phase,n]+signal[num_phase,n-2])
                den=den+(signal[num_phase,n-1]**2)

        alpha=num/(2*den)
        w=arccos(alpha)
        # Amplitude estimation
        c=zeros(3)+1j*zeros(3)
       
        c_vect=ravel((real(c),imag(c)),order='F')
        theta_vect=append(c_vect,w)
        return theta_vect

