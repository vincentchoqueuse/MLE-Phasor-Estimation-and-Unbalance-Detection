from numpy import *
from pylab import *

class Simulation_Config(object):
    
    def __init__(self,parameter_name,parameter_values,Ntest=50):
        """ Create Object """
        self.parameter_name=parameter_name
        self.parameter_values =parameter_values
        self.Ntest=Ntest

class Data_matrix(object):

    def __init__(self,simulation_config,technique_names,attribute_names,name="data_matrix"):

        self.name=name
        self.simulation_config=simulation_config
        self.technique_names=technique_names
        self.attribute_names=attribute_names
        
        self.Nb_techniques=len(technique_names)
        self.Nb_attributes=len(attribute_names)
        self.Nb_parameter_values=len(self.simulation_config.parameter_values)
        
        self.empty_data()                                                                           # create data matrix
        
    def empty_data(self):
        self.data=zeros((self.Nb_techniques,self.Nb_parameter_values,self.Nb_attributes))
    
    def convert_to_2D_array(self):
        #header
        header=[self.simulation_config.parameter_name]
        for technique in self.technique_names:
            header = header + ["%s %s" % (technique,attribute) for attribute in self.attribute_names]
        
        # data
        col=self.simulation_config.parameter_values.reshape((self.Nb_parameter_values,1))           #first dimension -> column vector
        data=self.data.swapaxes(0,1).reshape((self.Nb_parameter_values,self.Nb_techniques*self.Nb_attributes))      # convert 3D to 2D array
        full_data=hstack((col,data))                                                                # horizontal concatenation

        return header,full_data

    def show_values_at_parameter_index(self,index_parameter):
        for index_technique,technique in enumerate(self.technique_names):
            print("%s %s: %s" %(self.name,technique,array_str(self.data[index_technique,index_parameter,:],precision=2)))

class Monte_Carlo(object):

    def show(self):

        # Display result
        for index_attribute,attribute in enumerate(self.attribute_list):
            figure()
            for index,technique_name in enumerate(self.data_matrix_exp.technique_names):
                self.plot_type(self.simulation_config.parameter_values,self.data_matrix_exp.data[index,:,index_attribute],label=technique_name)
            for index,technique_name in enumerate(self.data_matrix_theo.technique_names):
                self.plot_type(self.simulation_config.parameter_values,self.data_matrix_theo.data[index,:,index_attribute],label=technique_name)

            xlabel(self.simulation_config.parameter_name)
            title("parameter %s" % self.get_attribute_name()[index_attribute])
            legend()

    def save(self,path):
        
        # construct header
        header,data=self.data_matrix_exp.convert_to_2D_array()
        savetxt("%s_%s.csv" % (path,self.data_matrix_exp.name), data, delimiter=",",header=",".join(header))   #export file

        header,data=self.data_matrix_theo.convert_to_2D_array()
        savetxt("%s_%s.csv" % (path,self.data_matrix_theo.name), data, delimiter=",",header=",".join(header))   #export file
