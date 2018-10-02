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

class ML_estimator(Phasor_Estimator):
    
    def estimate(self,signal):

        N=signal.shape[1]
        n_vect=arange(N)

        #Frequency estimation
        w_vect=zeros(self.nb_iter+1)
        w=self.w_init
        w_vect[0]=w

        for num_iter in range(1,self.nb_iter+1):
            
            #compute q, qp, qpp
            q=sum(exp(2j*w*n_vect))
            qp=2j*sum(n_vect*exp(2j*w*n_vect))
            qpp=-4*sum((n_vect**2)*exp(2j*w*n_vect))
            
            #compute beta, betap, betapp
            beta=1/(N**2-abs(q)**2)
            betap=2*(beta**2)*real(conj(q)*qp)
            betapp=2*(beta**2)*(abs(qp)**2)+2*(beta**2)*real(conj(q)*qpp)+8*(beta**3)*(real(conj(q)*qp)**2)
            
            Cp=0
            Cpp=0
            for indice in range(3):

                #compute Xk,Xkp, Xkpp
                Xk=sum(signal[indice,:]*exp(-1j*w*n_vect))
                Xkp=-1j*sum(n_vect*signal[indice,:]*exp(-1j*w*n_vect))
                Xkpp=-sum((n_vect**2)*signal[indice,:]*exp(-1j*w*n_vect))
                
                #compute gamma,gammap,gammapp
                gamma=N*(abs(Xk)**2)-real(q*Xk*Xk)
                gammap=2*N*real(conj(Xk)*Xkp)-real(qp*Xk*Xk+2*q*Xk*Xkp)
                gammapp=2*N*((abs(Xkp)**2)+real(conj(Xk)*Xkpp))-real(qpp*Xk*Xk+4*qp*Xk*Xkp)-2*real(q*Xkp*Xkp+q*Xk*Xkpp)

                Cp=Cp+2*betap*gamma+2*beta*gammap
                Cpp=Cpp+2*betapp*gamma+4*betap*gammap+2*beta*gammapp

            w=w-Cp/Cpp
            w_vect[num_iter]=w

        # Amplitude estimation
        c=zeros(3)+1j*zeros(3)
        q=sum(exp(2j*w*n_vect))
        for indice in range(3):
            Xk=sum(signal[indice,:]*exp(-1j*w*n_vect))
            c[indice]=(2/(N**2-(abs(q)**2)))*(N*Xk-conj(q)*conj(Xk))

        c_vect=ravel((real(c),imag(c)),order='F')
        theta_vect=append(c_vect,w)
        return theta_vect


class ML_estimator_approx(Phasor_Estimator):

    def estimate(self,signal):

        N=signal.shape[1]
        n_vect=arange(N)
        
        #Frequency estimation
        w_vect=zeros(self.nb_iter+1)
        w=self.w_init
        w_vect[0]=w
        
        for num_iter in range(1,self.nb_iter+1):
            
            Cp=0
            Cpp=0
            beta=1/(N**2)
            for indice in range(3):
                Xk=sum(signal[indice,:]*exp(-1j*w*n_vect))
                Xkp=-1j*sum(n_vect*signal[indice,:]*exp(-1j*w*n_vect))
                Xkpp=-sum((n_vect**2)*signal[indice,:]*exp(-1j*w*n_vect))

                gamma=N*(abs(Xk)**2)
                gammap=2*N*real(conj(Xk)*Xkp)
                gammapp=2*N*((abs(Xkp)**2)+real(conj(Xk)*Xkpp))
                
                Cp=Cp+2*beta*gammap
                Cpp=Cpp+2*beta*gammapp
            
            w=w-Cp/Cpp
            w_vect[num_iter]=w

        # Amplitude estimation
        c=zeros(3)+1j*zeros(3)
        q=sum(exp(2j*w*n_vect))
        for indice in range(3):
            Xk=sum(signal[indice,:]*exp(-1j*w*n_vect))
            c[indice]=(2/(N**2))*N*Xk

        c_vect=ravel((real(c),imag(c)),order='F')
        theta_vect=append(c_vect,w)
        return theta_vect


