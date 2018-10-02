from numpy import *
from numpy.random import *
from numpy.linalg import *
from scipy.stats import f as f_law


class Unbalanced_detector(object):
    def __str__(self):
        return self.name
    
    def __init__(self,estimator,threshold,name="Unbalanced detector",verbose=0):
        self.estimator=estimator                                #estimator of the frequency
        self.threshold=threshold                                #probability of false alarm
        self.name=name
    
    def get_threshold(self,N):
        return self.threshold
    
    def compute_criterion(self,signal):
        N=signal.shape[1]
        n_vect=arange(N)
        
        #Estimate the signal parameter
        theta_est=self.estimator.estimate(signal)
        w=theta_est[-1]
        
        #precomputation
        q=sum(exp(2j*w*n_vect))
        beta=1/(N**2-abs(q)**2)
        signal_mat=mat(signal)
        R=(signal_mat.T*signal_mat)/3
        
        # compute symmetrical components
        c_est=mat(theta_est[:6:2]+1j*theta_est[1:6:2]).T        #estimator of the phasor
        F=(1/sqrt(3))*mat([[1,1,1],[1,exp(-2j*pi/3),exp(2j*pi/3)]])   # Fortescue transform
        u=F*c_est
        
        #estimate the noise variance
        C=0
        for indice in range(3):
            Xk=sum(signal[indice,:]*exp(-1j*w*n_vect))
            C=C+2*beta*(N*(abs(Xk)**2)-real(q*Xk*Xk))
        
        sigma2_est=(trace(R)-(1/3)*C)/(N-2)
        
        #compute test statistic
        criterion=(1/(8*sigma2_est))*(N*abs(u[0])**2+real(q*u[0]**2)+((N**2-abs(q)**2)/N)*abs(u[1])**2)
        
        return criterion

    def detect(self,signal,method=0):
        
        N=signal.shape[1]
        criterion=self.compute_criterion(signal)                #compute criterion
        threshold=self.get_threshold(N)                         #get threshold
            
        if criterion> threshold:
                output=1
        else:
            output=0
        
        return output

class CFAR_Unbalanced_detector(Unbalanced_detector):

    
    def __init__(self,estimator,pfa=0.01,name="CFAR detector",verbose=0):
        self.estimator=estimator                                #estimator of the frequency
        self.pfa=pfa                                            #probability of false alarm
        self.name=name
    
    def get_threshold(self,N):
        return f_law.ppf(1-self.pfa,4,3*N-6)

