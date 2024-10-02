# import lib
import numpy as np
from scipy import signal
from scipy.integrate import simps
from tensorpac import Pac
from Frequency_filtering import freq_filt

# function to calculate spectral features
def feature_spectral(filtered,p,win2=0.5):
    freq_band=['delta','theta','alpha','sigma','beta','gama']
    """calculate spectral features such as absolute power, relative power, spectral entropy, frequency amplitude asymmetry and phase amplitude coupling

    Args:
        filtered (dictionary): artifact removed filtered clean data
        p (dictionary): contains all parameters of the dataset

    Returns:
        filtered_data (dictionary): data filtered in different frequency bands
        band_power (dictionary): average power
        rel_power (dictionary): relative power
        slow_fast (dictionary): power ratios
        spec_ent (dictionary): spectral entropy
        n (dictionary): frequency amplitude asymmetry
        r (dictionary): phase amplitude coupling
        
    """
    
    #https://raphaelvallat.com/bandpower.html
    if 'absolute_power' in p['features'] or 'relative_power' in p['features'] or 'spectral_entropy' in p['features']:        

        delta_low, delta_high = 0.5, 4
        theta_low, theta_high = 4, 8
        alpha_low, alpha_high = 8, 12
        sigma_low, sigma_high = 13, 16
        beta_low, beta_high = 16, 25
        gama_low, gama_high = 30, 40
        total_low, total_high = 0.5, 50

        #q={}
        band_power={}
        rel_power={}
        spec_ent={}
        slow_fast={}
        
        for ele in filtered:
            
            win = int(2/win2) * p['fs']

            #q[ele]=[[] for _ in range(2)]
            band_power[ele]=[[] for _ in range(8)]
            rel_power[ele]=[[] for _ in range(6)]
            spec_ent[ele]=[[] for _ in range(6)]
            slow_fast[ele]=[[] for _ in range(3)]
            for i in range (np.shape(filtered[ele])[2]):
                a,b,c,d,e,f,g,h,l,m=np.zeros((3,np.shape(filtered[ele])[0])),np.zeros((4,np.shape(filtered[ele])[0])),np.zeros((4,np.shape(filtered[ele])[0])),np.zeros((4,np.shape(filtered[ele])[0])),np.zeros((1,np.shape(filtered[ele])[0])),np.zeros((1,np.shape(filtered[ele])[0])),np.zeros((4,np.shape(filtered[ele])[0])),np.zeros((4,np.shape(filtered[ele])[0])),np.zeros((4,np.shape(filtered[ele])[0])),np.zeros((3,np.shape(filtered[ele])[0]))
                for j in range(np.shape(filtered[ele])[0]):
                    freqs, psd = signal.welch(filtered[ele][j,:,i], p['fs'], nperseg=win)
                    norm_psd=psd/np.sum(psd)
                    freq_res = freqs[1] - freqs[0]

                    idx_total = np.logical_and(freqs >= total_low, freqs <= total_high)
                    a[0,j] = simps(psd[idx_total], dx=freq_res)
                    a[1,j] = simps(norm_psd[idx_total],dx=freq_res)
                    a[2,j] = - np.sum(a[1,j] * np.log2(a[1,j]))

                    idx_delta = np.logical_and(freqs >= delta_low, freqs <= delta_high)
                    b[0,j] = simps(psd[idx_delta], dx=freq_res)
                    b[1,j] = simps(norm_psd[idx_delta],dx=freq_res)
                    b[2,j] = - np.sum(b[1,j] * np.log2(b[1,j]))
                    b[3,j] = b[0,j] / a[0,j]

                    idx_theta = np.logical_and(freqs >= theta_low, freqs <= theta_high)
                    c[0,j] = simps(psd[idx_theta], dx=freq_res)
                    c[1,j] = simps(norm_psd[idx_theta],dx=freq_res)
                    c[2,j] = - np.sum(c[1,j] * np.log2(c[1,j]))
                    c[3,j] = c[0,j] / a[0,j]

                    idx_alpha = np.logical_and(freqs >= alpha_low, freqs <= alpha_high)
                    d[0,j] = simps(psd[idx_alpha], dx=freq_res)
                    d[1,j] = simps(norm_psd[idx_alpha],dx=freq_res)
                    d[2,j] = - np.sum(d[1,j] * np.log2(d[1,j]))
                    d[3,j] = d[0,j] / a[0,j]

                    idx_alpha1 = np.logical_and(freqs >= 8, freqs <= 10)
                    e[0,j] = simps(psd[idx_alpha1], dx=freq_res)
                    #alpha1_rel_power[ele][i] = alpha1_power[ele][i] / total_power[ele][i]

                    idx_alpha2 = np.logical_and(freqs >= 10, freqs <= 12)
                    f[0,j] = simps(psd[idx_alpha2], dx=freq_res)
                    #alpha2_rel_power[ele][i] = alpha2_power[ele][i] / total_power[ele][i]

                    idx_sigma = np.logical_and(freqs >= sigma_low, freqs <= sigma_high)
                    g[0,j] = simps(psd[idx_sigma], dx=freq_res)
                    g[1,j] = simps(norm_psd[idx_sigma],dx=freq_res)
                    g[2,j] = - np.sum(g[1,j] * np.log2(g[1,j]))
                    g[3,j] = g[0,j] / a[0,j]

                    idx_beta = np.logical_and(freqs >= beta_low, freqs <= beta_high)
                    h[0,j] = simps(psd[idx_beta], dx=freq_res)
                    h[1,j] = simps(norm_psd[idx_beta],dx=freq_res)
                    h[2,j] = - np.sum(h[1,j] * np.log2(h[1,j]))
                    h[3,j] = h[0,j] / a[0,j]

                    idx_gama = np.logical_and(freqs >= gama_low, freqs <= gama_high)
                    l[0,j] = simps(psd[idx_gama], dx=freq_res)
                    l[1,j] = simps(norm_psd[idx_gama],dx=freq_res)
                    l[2,j] = - np.sum(l[1,j] * np.log2(l[1,j]))
                    l[3,j] = l[0,j] / a[0,j]

                    m[0,j] = c[0,j] / e[0,j]
                    m[1,j] = c[0,j] / f[0,j]
                    m[2,j] = e[0,j] / f[0,j]

                #q[ele][0].append(a[0,:])
                #q[ele][1].append(a[2,:])
                
                band_power[ele][0].append(b[0,:])
                band_power[ele][1].append(c[0,:])
                band_power[ele][2].append(d[0,:])
                band_power[ele][3].append(g[0,:])
                band_power[ele][4].append(h[0,:])
                band_power[ele][5].append(l[0,:])
                band_power[ele][6].append(e[0,:])
                band_power[ele][7].append(f[0,:])
                
                rel_power[ele][0].append(b[3,:])
                rel_power[ele][1].append(c[3,:])
                rel_power[ele][2].append(d[3,:])
                rel_power[ele][3].append(g[3,:])
                rel_power[ele][4].append(h[3,:])
                rel_power[ele][5].append(l[3,:])
                
                spec_ent[ele][0].append(b[2,:])
                spec_ent[ele][1].append(c[2,:])
                spec_ent[ele][2].append(d[2,:])
                spec_ent[ele][3].append(g[2,:])
                spec_ent[ele][4].append(h[2,:])
                spec_ent[ele][5].append(l[2,:])
                
                slow_fast[ele][0].append(m[0,:])
                slow_fast[ele][1].append(m[1,:])
                slow_fast[ele][2].append(m[2,:])
                
            band_power[ele] = np.transpose(band_power[ele],(0,2,1))
            rel_power[ele] = np.transpose(rel_power[ele],(0,2,1))
            spec_ent[ele] = np.transpose(spec_ent[ele],(0,2,1))
            slow_fast[ele] = np.transpose(slow_fast[ele],(0,2,1))
            
    #=========================================================================================================                    
    # frequency amplitude asymmetry
    if 'frequency amplitude asymmetry' in p['features']:
        n={}

        for ele in filtered:
            z=0
            n[ele]=np.zeros((6,int(np.shape(band_power[ele][2])[0]*(np.shape(band_power[ele][2])[0]-1)/2),np.shape(band_power[ele][2])[1]))
            for i in range(np.shape(band_power[ele][2])[0]):
                for j in range(np.shape(band_power[ele][2])[0]):
                    if j > i:
                        # major correction: changed len(p['channels']) to len(freq_band)===============================
                        for k in range(len(freq_band)):
                            n[ele][k,z,:] = (band_power[ele][k][i]-band_power[ele][k][j])/(band_power[ele][k][i]+band_power[ele][k][j])
                            
                        z=z+1

    #=========================================================================================================                    
    # phase amplitude coupling
    if 'phase amplitude coupling' in p['features']:
        r={}

        for ele in filtered:
            # theta-gama
            r[ele]=np.zeros((2,np.shape(filtered[ele])[0],np.shape(filtered[ele])[2]))
            q1 = Pac(idpac=(2, 0, 0), f_pha=[4,8], f_amp=[30,40],dcomplex='hilbert',n_bins=18,verbose=False)

            for i in range ((np.shape(filtered[ele])[2])):
                for j in range ((np.shape(filtered[ele])[0])):
                    r[ele][0,j,i] = q1.filterfit(p['fs'], filtered[ele][j,:,i])

            # theta-alpha
            q1 = Pac(idpac=(2, 0, 0), f_pha=[4,8], f_amp=[8,12],dcomplex='hilbert',n_bins=18,verbose=False)

            for i in range ((np.shape(filtered[ele])[2])):
                for j in range ((np.shape(filtered[ele])[0])):
                    r[ele][1,j,i] = q1.filterfit(p['fs'], filtered[ele][j,:,i])

    # filter data in different bands
    filtered_data = {}
    
    freq_band=['delta','theta','alpha','sigma','beta','gama']
    
    for i in range(len(freq_band)):
            p['freq_band']      = freq_band[i]
            filtered_data[freq_band[i]]  =    freq_filt(filtered,p)
    
    return filtered_data,band_power,rel_power,slow_fast,spec_ent,n,r
    
    #========================================================================================================= 
    #========================================================================================================= 