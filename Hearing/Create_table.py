# import lib
import numpy as np
import pandas as pd

def create_table(Control_BP_human,Control_RP_human,Control_SF_human,Control_spec_ent_human,Control_FAA_human,Control_PAC_human,Activity_Control_human,Mobility_Control_human,Complexity_Control_human,Control_human,freq_band,Control_sleep_label,p,*args):
    """consolidate all dictionaries into a dataframe for each subject

    Args:
        Control_BP_human (dictionary): average power
        Control_RP_human (dictionary): relative power
        Control_SF_human (dictionary): power ratios
        Control_spec_ent_human (dictionary): spectral entropy
        Control_FAA_human (dictionary): frequency amplitude asymmetry
        Control_PAC_human (dictionary): phase amplitude coupling
        Activity_Control_human (dictionary): Activity
        Mobility_Control_human (dictionary): Mobility
        Complexity_Control_human (dictionary): Complexity
        Control_human (list): list of subject ID
        freq_band (string): list of frequency bands
        Control_sleep_label (dictionary): corresponding labels
        p (dictionary): contains all parameters of the dataset

    Returns:
        dictionary: dataframe for each subject
    """

    Control_coh_human  = args[0]
    Control_phase_syn_human = args[1]
        
    datfrm = {}
    for i in Control_human:
        
        datfrm[i]=pd.DataFrame({})
        
        if 'absolute_power' in p['features']:
            for j in range(len(freq_band)):
                for k in range(len(p['channel'])):
                    datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                        freq_band[j]+'_power_'+p['channel'][k]:np.log10(Control_BP_human[i][j,k,:]),
                        'alpha1'+'_power_'+p['channel'][k]:np.log10(Control_BP_human[i][6,k,:]),
                        'alpha2'+'_power_'+p['channel'][k]:np.log10(Control_BP_human[i][7,k,:]),
                    })], axis=1)
        
        if 'relative_power' in p['features']:
            for j in range(len(freq_band)):
                for k in range(len(p['channel'])):
                    datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                        freq_band[j]+'_rel_power_'+p['channel'][k]:np.log10((Control_RP_human[i][j,k,:])/(1-(Control_RP_human[i][j,k,:])))
                    })], axis=1)

        if 'hjorth' in p['features']:
            for j in range(len(freq_band)):
                for k in range(len(p['channel'])):
                    datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                        freq_band[j]+'_activity_hjorth_'+p['channel'][k]:Activity_Control_human[freq_band[j]][i][k],
                        freq_band[j]+'_mobility_hjorth_'+p['channel'][k]:Mobility_Control_human[freq_band[j]][i][k],
                        freq_band[j]+'_complexity_hjorth_'+p['channel'][k]:Complexity_Control_human[freq_band[j]][i][k],
                    })], axis=1)

        if 'phase amplitude coupling' in p['features']:
            for k in range(len(p['channel'])):
                datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                    'theta_gama_PAC_'+p['channel'][k]:Control_PAC_human[i][0,k,:],
                    'theta_alpha_PAC_'+p['channel'][k]:Control_PAC_human[i][1,k,:],
                })], axis=1)

        if 'slow_fast' in p['features']:
            for k in range(len(p['channel'])):
                datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                    'alpha1_alpha2_'+p['channel'][k]:np.log10((Control_SF_human[i][0,k,:])/(100-(Control_SF_human[i][0,k,:]))),
                    'theta_alpha1_'+p['channel'][k]:np.log10((Control_SF_human[i][1,k,:])/(100-(Control_SF_human[i][1,k,:]))),
                    'theta_alpha2_'+p['channel'][k]:np.log10((Control_SF_human[i][2,k,:])/(100-(Control_SF_human[i][2,k,:]))),
                })], axis=1)

        if 'frequency amplitude asymmetry' in p['features']:
            for j in range(len(freq_band)):
                z=0
                for k in range(len(p['channel'])):
                    for l in range(len(p['channel'])):
                        if l>k:
                            datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                                p['channel'][k]+'_'+p['channel'][l]+'_'+freq_band[j]:np.log10((2+(Control_FAA_human[i][j,z,:]))/(2-(Control_FAA_human[i][j,z,:])))
                            })], axis=1)
                            z=z+1

        if 'coherence' in p['features']:
            for j in range(len(freq_band)):
                z=0
                for k in range(len(p['channel'])):
                    for l in range(len(p['channel'])):
                        if k>l:
                            datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                                'Coh_'+freq_band[j]+'_'+p['channel'][k]+'_'+p['channel'][l]:np.log10((Control_coh_human[i][j,z,:])/(1-(Control_coh_human[i][j,z,:]))),
                                'PLV_'+freq_band[j]+'_'+p['channel'][k]+'_'+p['channel'][l]:Control_phase_syn_human[j][i][0,z,:],
                                'Phase_'+freq_band[j]+'_'+p['channel'][k]+'_'+p['channel'][l]:Control_phase_syn_human[j][i][1,z,:],
                            })], axis=1)
                            z=z+1

        if 'spectral_entropy' in p['features']:
            for j in range(len(freq_band)):
                for k in range(len(p['channel'])):
                    datfrm[i] = pd.concat([datfrm[i], pd.DataFrame({
                        'spec_ent_'+freq_band[j]+'_'+p['channel'][k]:-np.log10(1-(Control_spec_ent_human[i][j,k,:])),
                    })], axis=1)
        
        
        #datfrm[i]['age']=np.ones(len(datfrm[i]))*p['age'][i][0]
        
        #datfrm[i]['sleep_label']=[x[0:-1] for x in Control_sleep_label[i]]
        
        #datfrm[i]['Tbi_label']=[x[-1] for x in Control_sleep_label[i]]
        
        datfrm[i] = datfrm[i].loc[:,~datfrm[i].columns.duplicated()]
    
    return datfrm