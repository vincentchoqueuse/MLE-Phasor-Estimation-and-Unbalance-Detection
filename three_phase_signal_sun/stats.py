from numpy import *
from pylab import *
from numpy.random import *
from scipy.stats import f, ncf


# CRITERION PDF

class PDF_criterion(object):
    
    def __str__(self):
        return self.name

class criterion_H0(PDF_criterion):
    
    def __init__(self,signal_model,name="Criterion under assumption H0"):
        self.signal_model = signal_model
        self.name=name

    def pdf(self,x_vect=0):
        """ Probability density function of the given RVS """
        N=self.signal_model.N

        pdf=f.pdf(x_vect,4,3*N-6)
        return pdf

    def cdf(self,x=0):
        """ Cumulative distribution function of the given RVS """
        N=self.signal_model.N
        cdf=f.cdf(x,4,3*N-6)
        return cdf

class criterion_H1(PDF_criterion):
    
    def __init__(self,signal_model,name="Criterion under assumption H1"):
        self.signal_model = signal_model
        self.name=name
    
    def compute_nc(self):
        """ compute decentralized parameter """
        
        N=self.signal_model.N
        w=self.signal_model.w0
        c=self.signal_model.c
        sigma2=self.signal_model.sigma2
        
        # Compute decentralized parameter
        n_vect=arange(N)
        c_vect=mat(c).T
        F=(1/3)*mat([[1,1,1],[1,exp(-2j*pi/3),exp(2j*pi/3)],[1,exp(2j*pi/3),exp(-2j*pi/3)]])   # Fortescue transform
        u=F*c_vect                                              # Symmetrical component
        
        # reverse order (
        theta=mat(ravel([real(u),imag(u)],order='F')).T
        
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
        
        Lambda=(3/(2*sigma2))*(theta.T*A.T*inv(A*inv(G.T*G)*A.T)*A*theta)
        nc=ravel(Lambda)[0]
    
        return nc    
    
    def pdf(self,x_vect=0):
        """ Probability density function at x of the given RVS """
        N=self.signal_model.N
        pdf=ncf.pdf(x_vect,4,3*N-6,self.compute_nc())
        
        return pdf
    
    def cdf(self,x=0):
        """ Cumulative distribution function of the given RVS """
        N=self.signal_model.N
        cdf=ncf.cdf(x,4,3*N-6,self.compute_nc())
        return cdf
        
