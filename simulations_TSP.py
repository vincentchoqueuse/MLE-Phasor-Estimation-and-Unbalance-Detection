from three_phase_signal.model import *
from three_phase_signal.estimation import *
import three_phase_signal_sun.estimation as sun_est
from three_phase_signal.detection import *
import three_phase_signal_sun.detection as sun
from three_phase_signal.bounds import *
from three_phase_signal.stats import *
from simulations.estimation import *
from simulations.detection import *
from simulations.general import *
from numpy import *
from pylab import *

############################
## SIMULATION PARAMETERS  ##
############################

estimation_mc=1
detection_mc=1
detection_hist=1
detection_roc=1
detection_ns=0
Ntest=5000


#Signal parameter (See Phadke book, exemple 3?4 on page 76)
Fe=1440                 #24*60
f0=60.5                 # frequency
w0=2*pi*f0/Fe           # normalized angular frequency
SNR=20
N=48


# list of phasors configuration)
c_balanced=array([100*exp(0),100*exp(-2j*pi/3),100*exp(2j*pi/3)])
c_unbalanced_1=array([130*exp(0.7854j),85.44*exp(-1.4105j),85.44*exp(2.9813j)])
c_unbalanced_2=array([100.05*exp(0.01j),99.7*exp(-2.00j*pi/3),99.9*exp(2.01j*pi/3)])

#Algorithm parameters
fnorm=60
winit=2*pi*fnorm/Fe
Nb_iter=2

# Monte carlo Simulation
if estimation_mc==1:
    
    simulation_config_list=[Simulation_Config('N',arange(10,101,1),Ntest),Simulation_Config('SNR',arange(0,51,1),Ntest)]
    parameters_list=[6,7,8,9]
    
    for simulation_config in simulation_config_list:
        
        signal_unbalanced=Signal_Model_with_SNRdB(c_unbalanced_1,w0,Fe,N,SNR)

        estimation_list=[sun_est.ML_estimator_approx(winit,Nb_iter,name="Sun Approx. MLE"),
                         ML_estimator(winit,Nb_iter,name="Exact MLE"),
                         ML_estimator_approx(winit,Nb_iter,name="Approx. MLE"),
                         ]

        bounds_list=[Exact_Bounds(signal_unbalanced,name="Exact CRB"),
                     Approximated_Bounds(signal_unbalanced,name="Approx. CRB")]#List of statistical bounds

        mc=Monte_Carlo_Estimation(signal_unbalanced,simulation_config,estimation_list,bounds_list,parameters_list)
        mc.simulate()
        mc.save("./csv/Monte_Carlo_%s" % simulation_config.parameter_name)
        mc.show()

if detection_hist==1:
    
    N=48
    SNR=5
    
    simulation_config_hist=Simulation_Config('Criterion',arange(0.01,15,0.2),Ntest)
    
    
    detector=CFAR_Unbalanced_detector(ML_estimator(w0,0) )                                              # create detector
    signal_balanced=Signal_Model_with_SNRdB(c_balanced,w0,Fe,N,SNR)                                     # create signal
    signal_unbalanced=Signal_Model_with_SNRdB(c_unbalanced_1,w0,Fe,N,SNR)                               # create signal
    signal_list=[signal_balanced,signal_unbalanced]
    law_list=[criterion_H0(signal_balanced),criterion_H1(signal_unbalanced)]                            # create pdf
    
    BT_hist=BT_Histogram_vs_Pdf(signal_list,simulation_config_hist,detector,law_list,50)
    BT_hist.simulate()
    BT_hist.save("./csv/Criterion")
    
    BT_hist.show()

if detection_mc==1:
    N=24
    SNR=5
    pfa=0.05
    
    simulation_config_list=[Simulation_Config('N',arange(10,101,1),Ntest),Simulation_Config('SNR',arange(0,26,1),Ntest)]
    
    clairvoyant_detector=CFAR_Unbalanced_detector(ML_estimator(w0,0),pfa,name="Clairvoyant CFAR detector")
    blind_detector=CFAR_Unbalanced_detector(ML_estimator(winit,Nb_iter),pfa,name="Blind CFAR detector")
    clairvoyant_detector_sun=sun.CFAR_Unbalanced_detector(ML_estimator(w0,0),pfa,name="Clairvoyant Sun detector")

    
    detector_list=[clairvoyant_detector_sun,clairvoyant_detector,blind_detector]
    
    for simulation_config in simulation_config_list:
        
        #compute probability of detection
        signal_unbalanced=Signal_Model_with_SNRdB(c_unbalanced_1,w0,Fe,N,SNR)
        mc=Monte_Carlo_Detection(signal_unbalanced,simulation_config,detector_list,[criterion_H1(signal_unbalanced)])
        mc.simulate()
        mc.save("./csv/pd_%s" % simulation_config.parameter_name)
        mc.show()

        # compute probability of false alarm
        signal_balanced=Signal_Model_with_SNRdB(c_balanced,w0,Fe,N,SNR)
        mc=Monte_Carlo_Detection(signal_balanced,simulation_config,detector_list,[criterion_H0(signal_balanced)])
        mc.simulate()
        mc.save("./csv/pfa_%s" % simulation_config.parameter_name)
        mc.show()

if detection_roc==1:
    
    N=24
    SNR=40
    
    simulation_config_roc=Simulation_Config('threshold',arange(0,110,1),Ntest)

    clairvoyant_detector=Unbalanced_detector(ML_estimator(w0,0),threshold=0.0,name="Clairvoyant detector")
    blind_detector=Unbalanced_detector(ML_estimator(winit,Nb_iter),threshold=0.0,name="Blind detector")
    clairvoyant_detector_sun=sun.Unbalanced_detector(ML_estimator(w0,0),threshold=0.0,name="Clairvoyant Sun detector")
    blind_detector_sun=sun.Unbalanced_detector(sun_est.ML_estimator_approx(winit,Nb_iter),threshold=0.0,name="Blind Sun detector")
    
    
    detector_list=[clairvoyant_detector_sun,clairvoyant_detector,blind_detector_sun,blind_detector]

    signal_balanced=Signal_Model_with_SNRdB(c_balanced,w0,Fe,N,SNR)
    signal_unbalanced=Signal_Model_with_SNRdB(c_unbalanced_2,w0,Fe,N,SNR)
    signal_list=[signal_balanced,signal_unbalanced]

    law_list=[criterion_H0(signal_balanced),criterion_H1(signal_unbalanced)]
    mc=ROC_curve(signal_list,simulation_config_roc,detector_list,law_list)
    mc.simulate()
    mc.save("./csv/ROC")
    mc.show()

if detection_ns==1:

    Nns= 150
    q=exp(1j*w0*(Nns))

    var_noise=10
    signal_balanced=Signal_Model(c_balanced,w0,Fe,Nns,var_noise)
    signal_unbalanced=Signal_Model(q*c_unbalanced_1*exp(-0.7854j),w0,Fe,Nns,var_noise)
    signal_balanced2=Signal_Model((q**2)*c_balanced,w0,Fe,Nns,var_noise)

    signal_model_NS=Signal_Model_NS([signal_balanced,signal_unbalanced,signal_balanced2])
    detector=Unbalanced_detector(ML_estimator(winit,Nb_iter),threshold=0.0,name="Blind detector")
    t,signal=signal_model_NS.waveform()
    detection_NS=criterion_NS(signal,detector,24,Noverlap=0.5,Fe=Fe)
    detection_NS.compute_criterion()
    detection_NS.save("./csv/NS")
    detection_NS.show()

show()


