# import lib
import numpy as np
 
def feature_nonlin(filtered_data,p,freq_band):
    """calculate non-linear features such as Hjorth parameters

    Args:
        filtered_data (dictionary): data
        p (dictionary): contains all parameters of the dataset
        freq_band (list): list of frequency bands
        species (string): human / mice

    Returns:
        Activity_Hjorth (dictionary): Activity
        Mobility_Hjorth (dictionary): Mobility
        Complexity_Hjorth (dictionary): Complexity
    """

    # Hjorth parameters
    def Hjorth(extract):
        yt = extract

        diff_yt = np.diff(yt,axis=1)
        diff2_yt = np.diff(diff_yt,axis=1)
        Activity_Hjorth = np.var(yt,axis=1)
        var_D_yt = np.var(diff_yt,axis=1)
        var_2D_yt = np.var(diff2_yt,axis=1)
        Mobility_Hjorth = np.sqrt(var_D_yt/Activity_Hjorth)
        Mobility_Hjorth2 = np.sqrt(var_2D_yt/var_D_yt)        

        Complexity_Hjorth = Mobility_Hjorth2/Mobility_Hjorth

        return Activity_Hjorth, Mobility_Hjorth, Complexity_Hjorth

    if 'hjorth' in p['features']:
        Activity_Hjorth, Mobility_Hjorth, Complexity_Hjorth = {},{},{}
        
        for i in freq_band:
            Activity_Hjorth[i], Mobility_Hjorth[i], Complexity_Hjorth[i] = {},{},{}
            for j in  filtered_data[i]:
                Activity_Hjorth[i][j],Mobility_Hjorth[i][j],Complexity_Hjorth[i][j]=Hjorth(filtered_data[i][j])

    return Activity_Hjorth,Mobility_Hjorth,Complexity_Hjorth