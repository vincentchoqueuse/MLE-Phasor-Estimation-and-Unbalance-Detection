from numpy import *
from pylab import *
from numpy.random import *


class Signal_Model(object):
    
    def __init__(self,c,w0,Fe,N,sigma2):

        """ Create Object """
        self.c = c
        self.w0 = w0
        self.N = N
        self.Fe=Fe
        self.sigma2=sigma2
    
    def set_parameter(self,name,value):
        """ Set signal parameter """
    
        if name=="SNR":
            SNR=10**(value/10)
            self.sigma2=sum(abs(self.c)**2)/(6*SNR)
        if name=="N":
            self.N=value

    def waveform(self):
        """ waveform: generate the signal waveform """
        n_vect=arange(self.N)
        signal=zeros((3,self.N))
        for num_phase in range(3):
            ck=self.c[num_phase]
            ak=abs(ck)
            phik=angle(ck)
            signal[num_phase,:]=ak*cos(self.w0*n_vect+phik)+sqrt(self.sigma2)*randn(self.N)

        t=n_vect/self.Fe
        return t,signal

    def show(self):
        """ Show waveform """
        
        t,signal=self.waveform()
        for num_phase in range(3):
            plot(t,signal[num_phase,:])
                
    def compute_squared_error(self,theta_est):
        """ compute square error for the unknown parameters """

        squared_error=zeros(10)
    
        #Compute error for theta
        c_vect=ravel((real(self.c),imag(self.c)),order='F')
        theta=append(c_vect,self.w0)
        squared_error[:7]=(theta-theta_est)**2                      # compute square error
        
        #compute TVE
        c_est=theta_est[:6:2]+1j*theta_est[1:6:2]                   # extract complex phasor
        squared_error[7:]=(abs(self.c-c_est)/abs(self.c))**2        # compute TVE
        
        return squared_error


    def get_error_attribute_name(self):
        return ["MSE Re c0", "MSE Im c0","MSE Re c1","MSE Im c1","MSE Re c2","MSE Im c2","w0","MTVE2 0","MTVE2 1", "MTVE2 2"]


class Signal_Model_with_SNRdB(Signal_Model):

    def __init__(self,c,w0,Fe,N,SNRdB):
        """ Create Object """
        self.c = c
        self.w0 = w0
        self.N = N
        self.Fe=Fe
        SNR=10**(SNRdB/10)                          #convert SNR in dB to SNR natural value
        self.sigma2=sum(abs(self.c)**2)/(6*SNR)     #convert SNR to sigma2



class Signal_Model_NS(object):
    """ Non stationary signal model """

    def __init__(self,signal_model_list):
        """ Create Object """
        self.signal_model_list=signal_model_list

    def waveform(self):
  
        #get sampling frequency
        Fe=self.signal_model_list[0].Fe
        
        signal=np.zeros((3,0))
        for signal_model in self.signal_model_list:
            t,signal_temp=signal_model.waveform();
            signal=hstack((signal,signal_temp))
        
        N=shape(signal)[1]
        t=arange(N)/Fe

        return t,signal

    def show(self):
        """ Show waveform """
        
        t,signal=self.waveform()
        for num_phase in range(3):
            plot(t,signal[num_phase,:])

