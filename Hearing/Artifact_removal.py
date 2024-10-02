# import lib
import numpy as np

def artifact_removal(data,p):
    """remove epochs containing artifacts if its z score is greater than 2.5 for certain criterias

    Args:
        data (dictionary): data from create_epochs function to
        p (dictionary): contains all parameters of the dataset

    Returns:
        AR_rem (dictionary): data after artifact removal
        mini ():
        m2 (dictionary): list of epoch numbers which are removed
    """

    AR_rem={}
    m2={}
    
    for i in data:

        m2[i]=np.zeros(np.shape(data[i])[2], dtype=bool)
        
        for j in range(np.shape(data[i])[0]):
            
            # calculate epochs whose z value for (max-min) is greater than 2.5
            a=(np.max(data[i][j],axis=0))-(np.min(data[i][j],axis=0))
            z=((a-np.mean(a))/np.std(a))
            avglatlist = np.arange(1, a.shape[0] + 1)
            m=np.abs(z) > 2.5
            
            # create mask to remove epochs
            m2[i]=np.logical_or(m2[i],m)

            # calculate epochs whose z value for sum(abs) is greater than 2.5
            #a=(np.var(data[i][j],axis=0))
            a=np.sum(np.abs(data[i][j]),axis=0)
            z=((a-np.mean(a))/np.std(a))
            m=np.abs(z) > 2.5

            # create mask to remove epochs
            m2[i]=np.logical_or(m2[i],m)

        # delete epochs 
        AR_rem[i] = np.delete(data[i],avglatlist[m2[i]]-1,2)
        mini=10**10

        for ele in AR_rem:
            if np.shape(AR_rem[ele])[2]<mini:
                mini=np.shape(AR_rem[ele])[2]
    
    return AR_rem,mini,m2