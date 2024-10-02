# import lib
import numpy as np
from scipy import signal
import math


def feature_connectivity(filtered,p,filtered_data,freq_band,win2=0.5):
    """calculate connectivity features such as coherence and phase synchrony

    Args:
        filtered (dictionary): data
        p (dictionary): contains all parameters of the dataset
        class_label (string): Control / Tbi
        species (string): human / mice
        filtered_data (dictionary): data filtered in different frequency bands
        freq_band (list): list of frequency bands

    Returns:
        m (dictionary) : coherence
        phase_syn (dictionary): phase synchrony
    """
    
    win = int(2/win2) * p['fs']
    delta_low, delta_high = 0.5, 4
    theta_low, theta_high = 4, 8
    alpha_low, alpha_high = 8, 12
    sigma_low, sigma_high = 13, 16
    beta_low, beta_high = 16, 25
    gama_low, gama_high = 30, 40
    total_low, total_high = 0.5, 50

    #=========================================================================================================
    # coherence
    if 'coherence' in p['features']:
        m={}
        for ele in filtered:
            z=0
            m[ele]=np.zeros((6,int(np.shape(filtered[ele])[0]*(np.shape(filtered[ele])[0]-1)/2),np.shape(filtered[ele])[2]))
            for a in range (len(p['channel'])):
                for b in range (len(p['channel'])):
                    if a>b:
                        for i in range (np.shape(filtered[ele])[2]):
                            f, Cxy=signal.coherence(filtered[ele][a,:,i],filtered[ele][b,:,i],p['fs'],'hann',win)
                            idx_delta = np.logical_and(f >= delta_low, f <= delta_high)
                            m[ele][0,z,i]=np.mean(Cxy[idx_delta])

                            idx_theta = np.logical_and(f >= theta_low, f <= theta_high)
                            m[ele][1,z,i]=np.mean(Cxy[idx_theta])

                            idx_alpha = np.logical_and(f >= alpha_low, f <= alpha_high)
                            m[ele][2,z,i]=np.mean(Cxy[idx_alpha])

                            idx_sigma = np.logical_and(f >= sigma_low, f <= sigma_high)
                            m[ele][3,z,i]=np.mean(Cxy[idx_sigma])

                            idx_beta = np.logical_and(f >= beta_low, f <= beta_high)
                            m[ele][4,z,i]=np.mean(Cxy[idx_beta])

                            idx_gama = np.logical_and(f >= gama_low, f <= gama_high)
                            m[ele][5,z,i]=np.mean(Cxy[idx_gama])
                        z=z+1
    
    #========================================================================================================= 
    # phase synchrony
    if 'phase synchrony' in p['features']:
        def hilphase2(y1,y2,n_sample):
            sig1_hill=signal.hilbert(y1)
            sig2_hill=signal.hilbert(y2)
            phase_y1=np.unwrap(np.angle(sig1_hill))
            phase_y2=np.unwrap(np.angle(sig2_hill))
            Inst_phase_diff=phase_y1-phase_y2
            avg_phase=np.average(Inst_phase_diff)

            perc10w =  math.floor(n_sample/10)
            phase_y1 = phase_y1[perc10w:-perc10w]
            phase_y2 = phase_y2[perc10w:-perc10w]

            plv=np.abs(np.sum(np.exp(1j * (phase_y1 - phase_y2))))/len(phase_y1)

            return plv,avg_phase

        def cal_plv(filtered_data,p):
            r={}
            n_samples=p['epoch_len']*p['fs']

            for ele in filtered_data:
                z=0
                r[ele]=np.zeros((2,int(np.shape(filtered_data[ele])[0]*(np.shape(filtered_data[ele])[0]-1)/2),np.shape(filtered_data[ele])[2]))
                for a in range (len(p['channel'])):
                    for b in range (len(p['channel'])):
                        if a>b:
                            for i in range (np.shape(filtered_data[ele])[2]):
                                r[ele][0,z,i],r[ele][1,z,i]=hilphase2(filtered_data[ele][a,:,i],filtered_data[ele][b,:,i],n_samples)
                                #r[ele][3,z,i]=np.corrcoef(filtered[ele][a,:,i],filtered[ele][b,:,i])[0,1]
                            z=z+1
            return r

        # Phase locking value
        phase_syn = [[] for _ in range(6)]
        for i in range(len(freq_band)):
                p['freq_band']      = freq_band[i]
                phase_syn[i] = cal_plv(filtered_data[freq_band[i]],p)
        
    #========================================================================================================= 
    
    return m,phase_syn
    