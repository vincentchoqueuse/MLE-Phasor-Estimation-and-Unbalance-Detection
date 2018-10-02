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
        w0=self.signal_model.w0
        c=self.signal_model.c
        sigma2=self.signal_model.sigma2
        
        # Compute decentralized parameter
        n_vect=arange(N)
        q=sum(exp(2j*w0*n_vect))
        c_vect=mat(c).T
        F=(1/3)*mat([[1,1,1],[1,exp(-2j*pi/3),exp(2j*pi/3)]])   # Fortescue transform
        u=F*c_vect                                              # Symmetrical component
        Lambda=(3/(2*sigma2))*(N*abs(u[0])**2+real(q*u[0]**2)+((N**2-abs(q)**2)/N)*abs(u[1])**2)
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


