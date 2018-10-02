from numpy import *
from .general import *

class Monte_Carlo_Detection(Monte_Carlo):
    
    def __init__(self,signal_model,simulation_config,detectors_list,PDF_criterion_list):
        """ Create Object """
        self.signal_model=signal_model
        self.simulation_config =simulation_config
        self.detectors_list = detectors_list           #list of detectors
        self.PDF_criterion_list=PDF_criterion_list                          #list of theoretical PDF_criterion
        self.attribute_list=[0]                         #only one parameter
        self.plot_type=plot                             #plot type
    
    def simulate(self,verbose=1):
        
        #extract parameter (for code readibility)
        detectors_list=self.detectors_list
        PDF_criterion_list=self.PDF_criterion_list
        attribute_list=self.attribute_list
        simulation_config=self.simulation_config
        Ntest=self.simulation_config.Ntest
        parameter_values=self.simulation_config.parameter_values
        parameter_name=self.simulation_config.parameter_name
        
        #Create data matrix to save theoretical and experimental values
        data_matrix_exp=Data_matrix(simulation_config,detectors_list,self.get_attribute_name(),name="exp")
        data_matrix_theo=Data_matrix(simulation_config,PDF_criterion_list,self.get_attribute_name(),name="theo")
        
        for index_x, value in enumerate(parameter_values):
            print('----- %s=%f -----' % (parameter_name,value))
            self.signal_model.set_parameter(parameter_name,value)                               #set the new parameter value
            nb_detection=zeros((len(detectors_list),1))                                         #init nb_detection to 0 for all detectors
    
            for indice in range(Ntest):                                                         #estimation probability with MC simulations
                t,signal=self.signal_model.waveform()                                           #new signal realisation
                for index_detector,detector in enumerate(detectors_list):                       #simulation for each detector
                    detector.sigma2=self.signal_model.sigma2       #some detector require the knowledge of the noise variance
                    detected=detector.detect(signal)                                            #detect
                    nb_detection[index_detector,0]=nb_detection[index_detector,0]+detected      #update nb_detection
        
                data_matrix_exp.data[:,index_x,:]=nb_detection/Ntest                            #compute and store probability of detection
        
            threshold=detector.get_threshold(self.signal_model.N)                               #get criterion threshold from signal parameter

            for index_PDF_criterion,PDF_criterion in enumerate(PDF_criterion_list):
                data_matrix_theo.data[index_PDF_criterion,index_x,:]=1-PDF_criterion.cdf(threshold)

            if verbose==1:
                data_matrix_exp.show_values_at_parameter_index(index_x)
                data_matrix_theo.show_values_at_parameter_index(index_x)

        # save value in object attribute
        self.data_matrix_exp=data_matrix_exp
        self.data_matrix_theo=data_matrix_theo

    def get_attribute_name(self):
        return ["probability"]


class ROC_curve(Monte_Carlo):
    
    def __init__(self,signal_model_list,simulation_config,detectors_list,PDF_criterion_list):
        """ Create Object """
        self.signal_under_H0=signal_model_list[0]
        self.signal_under_H1=signal_model_list[1]
        self.simulation_config =simulation_config
        self.detectors_list = detectors_list                #list of detectors
        self.PDF_criterion_list=PDF_criterion_list
        self.attribute_list=[0,1]
    
    def simulate(self,verbose=1):
        
        #extract parameter (for code readibility)
        detectors_list=self.detectors_list
        PDF_criterion_list=self.PDF_criterion_list
        attribute_list=self.attribute_list
        simulation_config=self.simulation_config
        Ntest=self.simulation_config.Ntest
        parameter_values=self.simulation_config.parameter_values
        parameter_name=self.simulation_config.parameter_name
        
        #Create data matrix to save theoretical and experimental values
        data_matrix_exp=Data_matrix(simulation_config,detectors_list,self.get_attribute_name(),name="exp")
        data_matrix_theo=Data_matrix(simulation_config,["clairvoyant"],self.get_attribute_name(),name="theo")
    
        for index_threshold, threshold in enumerate(parameter_values):
            print('----- %s=%f -----' % (parameter_name,threshold))
            
            #update threshold for each detector
            for index_detector,detector in enumerate(detectors_list):
                detector.threshold=threshold

            # Experimental values
            nb_detection=zeros((len(self.detectors_list),2))
            for index_signal,signal_model in enumerate([self.signal_under_H0,self.signal_under_H1]):

                for indice in range(Ntest):
                    t,signal=signal_model.waveform()                                                #new signal realisation
                
                    for index_detector,detector in enumerate(detectors_list):
                        detector.sigma2=signal_model.sigma2    #some detector require the knowledge of the noise variance
                        detected=detector.detect(signal)                                            #estimate parameters
                        nb_detection[index_detector,index_signal]=nb_detection[index_detector,index_signal]+detected

            data_matrix_exp.data[:,index_threshold,:]=nb_detection/Ntest
            
            # Theoretical values
            for index_attribute,PDF_criterion in enumerate(PDF_criterion_list):
                data_matrix_theo.data[0,index_threshold,index_attribute]=1-PDF_criterion.cdf(threshold)

            if verbose==1:
                data_matrix_exp.show_values_at_parameter_index(index_threshold)
                data_matrix_theo.show_values_at_parameter_index(index_threshold)

        # save MSE and bound in the object attribute
        self.data_matrix_exp=data_matrix_exp
        self.data_matrix_theo=data_matrix_theo

    def get_attribute_name(self):
            return ["Pfa","Pd"]

    def show(self):
    
        # Display result
        figure()
        for index,technique_name in enumerate(self.data_matrix_exp.technique_names):
            plot(self.data_matrix_exp.data[index,:,0],self.data_matrix_exp.data[index,:,1],label=technique_name)
        plot(self.data_matrix_theo.data[0,:,0],self.data_matrix_theo.data[0,:,1],label="theoretical")
        xlabel("Pfa")
        ylabel("Pd")
        legend()


class BT_Histogram_vs_Pdf(object):
    
    """ Binary Test / criterion Histogram vs pdf """
    
    def __init__(self,signal_model_list,simulation_config,detector,PDF_criterion_list,Nbins):
        
        """ Create Object """
        self.signal_under_H0=signal_model_list[0]
        self.signal_under_H1=signal_model_list[1]
        self.detector=detector
        self.PDF_criterion_list=PDF_criterion_list
        self.Nbins=Nbins
        self.simulation_config=simulation_config
    
    def simulate(self,verbose=1):
        
        PDF_criterion_list=self.PDF_criterion_list
        simulation_config=self.simulation_config
        Ntest=self.simulation_config.Ntest
        parameter_values=self.simulation_config.parameter_values
        parameter_name=self.simulation_config.parameter_name

        Nb_histogram=2
        Nb_pdf=2
        
        experimental_values_mat=zeros((Nb_histogram,self.Nbins,2))
        theoretical_values_mat=zeros((Nb_pdf,len(parameter_values),2))
        
        #create signal model
        for index,signal in enumerate([self.signal_under_H0,self.signal_under_H1]):
            
            T_vect=zeros(Ntest)
            for n in range(Ntest):
                t,x=signal.waveform()
                T_vect[n]=self.detector.compute_criterion(x)
            
            #compute histogram
            histogram_n, histogram_bins= histogram(T_vect,self.Nbins,density=1)
            experimental_values_mat[index,:,0]=(histogram_bins[:-1]+histogram_bins[1:])/2    # store the center of each bin
            experimental_values_mat[index,:,1]=histogram_n
        
        for index,PDF_criterion in enumerate(self.PDF_criterion_list):
            # compute PDF
            theoretical_values_mat[index,:,0]=parameter_values
            theoretical_values_mat[index,:,1]=PDF_criterion.pdf(parameter_values)
        
        #save data
        self.experimental_values_mat=experimental_values_mat
        self.theoretical_values_mat=theoretical_values_mat
    
    def show(self):
        
        """ Show histogram and pdf """
        figure()
        for index in range(2):
            # show histogram
            width=(self.experimental_values_mat[index,1,0]-self.experimental_values_mat[index,0,0])
            bar(self.experimental_values_mat[index,:,0],self.experimental_values_mat[index,:,1], align='center', width=width,label="Histogram H%d" % index)
            plot(self.theoretical_values_mat[index,:,0],self.theoretical_values_mat[index,:,1], label="pdf H%d" % index) # show pdf
        
        legend()
    
    def save(self,path):
        
        for index in range(2):
            savetxt("%s_histogram_H%d.csv" % (path,index), self.experimental_values_mat[index,:,:], delimiter=",",header="Bins center, Histogram ")
            savetxt("%s_pdf_H%d.csv" % (path,index), self.theoretical_values_mat[index,:,:], delimiter=",",header="X, pdf") #export file



class criterion_NS(object):

    def __init__(self,signal,detector,N,Noverlap=0.5,Fe=1):
    
        """ Create Object """
        self.signal=signal
        self.Fe=Fe
        self.detector=detector
        self.N=N
        self.Noverlap=Noverlap

    def compute_criterion(self):

        Ntot=shape(self.signal)[1]
        criterion=[]
        N_vect=arange(0,Ntot,self.N*self.Noverlap )
        
        for N_slice in N_vect:
            signal_temp=self.signal[:,N_slice:N_slice+self.N]
            criterion_temp=self.detector.compute_criterion(signal_temp)
            criterion=append(criterion,criterion_temp)

        t=(N_vect+self.N)/self.Fe
        
        self.criterion=criterion
        self.t=t

    def show(self):

        figure()
        plot(self.t,self.criterion,label="criterion")
        legend()
    
    def save(self,path):
        
        Ntot=shape(self.signal)[1]
        signal_mat=vstack(( arange(Ntot)/self.Fe ,self.signal))
        savetxt("%s_signal.csv" % path, signal_mat.T, delimiter=",",header="t, signal ")
        criterion_mat=vstack((self.t,self.criterion))
        savetxt("%s_criterion.csv" % path, criterion_mat.T, delimiter=",",header="t,criterion") #export file


