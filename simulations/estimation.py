from numpy import *
from pylab import *
from .general import *

class Monte_Carlo_Estimation(Monte_Carlo):

    def __init__(self,signal_model,simulation_config,estimators_list,bounds_list,attribute_list):
        """ Create Object """
        self.signal_model=signal_model
        self.simulation_config =simulation_config
        self.estimators_list = estimators_list        #list of estimators
        self.bounds_list=bounds_list               #list of bounds
        self.attribute_list=attribute_list
        self.plot_type=semilogy
    
    def simulate(self,verbose=1):
        
        #extract parameter (for code readibility)
        estimators_list=self.estimators_list
        bounds_list=self.bounds_list
        attribute_list=self.attribute_list
        simulation_config=self.simulation_config
        Ntest=self.simulation_config.Ntest
        parameter_values=self.simulation_config.parameter_values
        parameter_name=self.simulation_config.parameter_name
        
        #Create data matrix to save theoretical and experimental values
        data_matrix_exp=Data_matrix(simulation_config,estimators_list,self.get_attribute_name(),name="exp")
        data_matrix_theo=Data_matrix(simulation_config,bounds_list,self.get_attribute_name(),name="theo")
        
        for index_x, value in enumerate(parameter_values):
            print('----- %s=%f -----' % (parameter_name,value))
    
            self.signal_model.set_parameter(parameter_name,value)                               #set the new parameter value
    
            experimental_values=zeros((len(estimators_list),len(attribute_list)))
            for indice in range(Ntest):
                t,signal=self.signal_model.waveform()                                           #new signal realisation
                for index_estimator,estimator in enumerate(estimators_list):
                    
                    theta_est=estimator.estimate(signal)                                        #estimate parameters
                    squared_error=self.signal_model.compute_squared_error(theta_est)
                    experimental_values[index_estimator,:]=experimental_values[index_estimator,:]+squared_error[self.attribute_list]
        
            data_matrix_exp.data[:,index_x,:]=experimental_values/Ntest
            for index_bound,bound in enumerate(self.bounds_list):
                data_matrix_theo.data[index_bound,index_x,:]=bound.compute()[attribute_list]    #compute CRB
               
            if verbose==1:
                data_matrix_exp.show_values_at_parameter_index(index_x)
                data_matrix_theo.show_values_at_parameter_index(index_x)

        # save MSE and bound in the object attribute
        self.data_matrix_exp=data_matrix_exp
        self.data_matrix_theo=data_matrix_theo

    def get_attribute_name(self):
        attribute_name_tot=self.signal_model.get_error_attribute_name()
        attribute_name=["%s" % attribute_name_tot[index_attribute] for index_attribute in self.attribute_list ]
        return attribute_name


