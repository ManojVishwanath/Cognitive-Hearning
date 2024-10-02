# import lib
import numpy as np

def create_epoch(data,p,class_label,species):
    """create EEG epochs (divide data into epochs)

    Args:
        data (dictionary): data extracted from extract functions
        p (dictionary): contains all parameters of the dataset
        class_label (string): Control / Tbi
        species (string): human / mice

    Returns:
        dictionary: data in the form (no. of electrode, epoch length*sampling freq, no. of epochs)
    """
    
    epoch_data = {}
    for ele in data:

        # choose sampling frequency
        if species == 'human':
            if ele in p['dataset1_'+class_label+'_human']:
                p['fs']=p['dataset1_fsh']
            elif ele in p['dataset2_'+class_label+'_human']:
                p['fs']=p['dataset2_fsh']
            
        # calculate number of epochs based on epoch length and sampling frequency
        num_epoch = int(np.shape(data[ele])[1]/(p['epoch_len']*p['fs']))
        a=[]
        
        # divide data into epochs
        for i in range(num_epoch):
            a.append(data[ele][:,i*p['fs']*p['epoch_len']:i*p['fs']*p['epoch_len']+p['fs']*p['epoch_len']])

        # transpose data into shape (ele x samples x epochs)
        epoch_data[ele] = np.transpose(a, (1, 2, 0))
        
    return epoch_data