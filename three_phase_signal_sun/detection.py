from numpy import *
from numpy.random import *
from numpy.linalg import *
from scipy.stats import chi2


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
        signal_mat=mat(signal)
        theta_est=self.estimator.estimate(signal)
        w=theta_est[-1]

        # construct G
        G=zeros((3*N,6))
        for n in range(N):
            Gn=zeros((3,6))
            Gn[0,:2]=array([sqrt(2)*cos(n*w),-sqrt(2)*sin(n*w)])
            Gn[1:3,2:]=array([[cos(n*w),-sin(n*w),cos(n*w),-sin(n*w)],
                               [-sin(n*w),-cos(n*w),sin(n*w),cos(n*w)]])
            G[n*3:(n+1)*3,:]=Gn
        
        G=mat(G)
        
        # create matrix A
        A=zeros((4,6))
        A[:,:4]=eye(4)
        A=mat(A)

        # create T
        T=(2/3)*mat([[sqrt(2)/2,sqrt(2)/2,sqrt(2)/2],[1,-1/2,-1/2],[0,sqrt(3)/2,-sqrt(3)/2]])
        vtn=T*signal_mat
        x=mat(ravel(vtn,order='F')).T

        # create vt
        theta=pinv(G)*x        #estimate theta
        criterion=(theta.T*A.T*inv(A*inv(G.T*G)*A.T)*A*theta)/(2*self.sigma2/3)
        
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

    
    def __init__(self,estimator,pfa=0.01,name="Sun CFAR detector",verbose=0):
        self.estimator=estimator                                #estimator of the frequency
        self.pfa=pfa                                            #probability of false alarm
        self.name=name
    
    def get_threshold(self,N):
        return chi2.ppf(1-self.pfa,4)