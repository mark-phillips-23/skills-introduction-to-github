import antares.devkit as dk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import pickle 

class Phillips_DeSoto_AnomalyDetection(dk.Filter):
    NAME = "Phillips_DeSoto_Anomaly Detection"
    ERROR_SLACK_CHANNEL = "U04196B2ABA" #Put your Slack User ID here
    INPUT_LOCUS_PROPERTIES = [
        "ztf_object_id",
        "desoto_dynesty_param_2",
        "desoto_dynesty_param_3",
        "desoto_dynesty_param_5",
        "desoto_dynesty_param_6",
        "desoto_dynesty_param_7",
        "desoto_dynesty_param_8",
        "desoto_dynesty_param_9",
        "desoto_dynesty_param_10",
        "desoto_dynesty_param_11",
        "desoto_dynesty_param_12",
        "desoto_dynesty_param_13",
        "desoto_dynesty_param_14"
    ]
    INPUT_ALERT_PROPERTIES = [
        'ant_mjd',
        'ztf_magpsf',
        'ztf_fid', # 1=g , 2=r
        'ztf_sigmapsf',
        'ztf_magzpsci',
        'ant_ra',
        'ant_dec'
    ]
    OUTPUT_LOCUS_PROPERTIES = [ 
        {
            'name': 'anomaly_score_gaussian',
            'type': 'float',
            'description': 'Anomaly score described in ___ using Gaussian Mixture Models' # ___ is research note/paper name
        },
        {
            'name': 'anomaly_score_isolation_forest',
            'type': 'float',
            'description': 'Anomaly score described in __ using Isolation Forests'
        }
    ]
    OUTPUT_ALERT_PROPERTIES = []
    OUTPUT_TAGS = [
        {
            'name': 'phillips_desoto_anomaly_gaussian',
            'description': 'Anomaly found from Gaussian Mixture Model described by Phillips et. al (in prep)'
        },
        {
            'name': 'phillips__desoto_anomaly_isolation_forest',
            'description': 'Anomaly found from Isolation Forest described by Phillips et. al (in prep)'
        }
    ]
    
    REQUIRES_FILES = ['Phillips_deSoto_gm_fit_covariances.npy', 
                      'Phillips_deSoto_gm_fit_means.npy',
                      'Phillips_deSoto_gm_fit_weights.npy',
                      'Phillips_IsolationForest.npy'
                     ]
    
    def setup(self):
        """
        ANTARES will call this function once each night 
        when the filters are loaded.
        """
        
         # Loads SFDQuery object once to lower overhead
        
        from dustmaps.config import config
        #config['data_dir'] = '/static_files/'  # production
        config['data_dir'] = '/tmp/'  # datalab
        import dustmaps.sfd
        dustmaps.sfd.fetch()
        from dustmaps.sfd import SFDQuery
        self.sfd = SFDQuery()
        
        # Gaussian Mixture Model
        
        means = np.load('Phillips_deSoto_gm_fit' + '_means.npy')
        weights = np.load('Phillips_deSoto_gm_fit' + '_weights.npy')
        covar = np.load('Phillips_deSoto_gm_fit' + '_covariances.npy')
        
        self.loaded_gm = GaussianMixture(n_components = len(means), random_state = 0, covariance_type = 'full')
        self.loaded_gm.means_ = means
        self.loaded_gm.covariances_ = covar
        self.loaded_gm.weights_ = weights
        self.loaded_gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
        
        # Isolation Forest
        self.loaded_IF = np.load('Phillips_IsolationForest.npy', allow_pickle = True)
        
                                               
    def run(self, locus):
        """
        Runs a filter that fits transient objects a Gaussian Mixture Model and Isolation
        Forest anomaly detection techniques described in ___. Tags the transient
        with an anomaly score from both techniques. Then, if the anomaly score is higher
        than a threshold, tags the object as an anomaly.
        
        Parameters
        ----------
        locus: Locus Object is the transient to be tagged with an anomaly score
        lightcurve parameters: Parameters of the transient's lightcurve described by deSoto et al, TBD
        """
        # Reading in lightcurve parameters from de Soto et al, TBD.
        parameters = []
        
        for i in range(14):
            if i == 0 or i == 3:
                continue
            parameters.append(locus.properties['desoto_dynesty_param_' + str(i+1)])
        params = np.array(parameters)
        print(params)
        params[1:5] = np.log10(params[1:5]) # Why is log not safe?
        
        # Gaussian Mixture Model anomaly score
        anom_score = -1 * self.loaded_gm.score_samples([params])[0]
        
        threshold = 14.77 # 3 sigma from the mean score from the training set
        mean_training_score = -33.43 # mean score from the training set
        
        threshold_score = mean_training_score + threshold
        
        locus.properties['anomaly_score_gaussian'] = anom_score
        
        if anom_score > threshold_score:
            print('Anomalous transient lightcurve')
            locus.tag('phillips_desoto_anomaly_gaussian')
            
        # Isolation Forest anomaly score 
        # We use newer version than ANTARES uses so need to wait till it's update
        anom_score_IF = -1 * self.loaded_IF.score_samples([params])[0]
        
        
        threshold_IF = 0.1692
        mean_training_score_IF = 0.4014
        
        threshold_score = mean_training_score_IF + threshold_IF
        
        locus.properties['anomaly_score_isolation_forest'] = anom_score_IF
        
        if anom_score_IF > threshold_score:
            print('Anomalous transient lightcurve')
            locus.tag('phillips_desoto_anomaly_isolation_forest')
        
        