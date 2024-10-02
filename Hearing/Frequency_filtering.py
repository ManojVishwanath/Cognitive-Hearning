# import lib
import numpy as np


def filt(sig,p,lf1,lf2,hf2,hf1):
    """multiply data in frequency domain with a smooth window and take ifft

    Args:
        sig (1D array): single epoch 
        p (_type_): contains all parameters of the dataset
        lf1 (float): _description_
        lf2 (float): _description_
        hf2 (float): _description_
        hf1 (float): _description_

    Returns:
        1D array: epoch after filtering
    """
        
    x=sig-np.mean(sig)
    no_samp=len(x)
    mfft=round(np.log(len(x))/np.log(2)+0.5)+1
    nfft=int(2**mfft)

    
    xf=np.fft.fft(x, int(nfft))
    xf2=np.zeros((nfft),dtype = 'complex_')
    lp2=int(np.fix((hf1/p['fs'])*nfft))
    lp22=int(np.fix((hf2/p['fs'])*nfft))
    wl1=np.ones(lp2)
    wl1[lp22:lp2]=0.5+0.5*np.cos(np.pi*(np.arange(lp2-lp22)+1)/(lp2-lp22))
    
    wl2=wl1[::-1]
    xf2[0:lp2]=np.multiply(xf[0:lp2],np.transpose(wl1))
    xf2[nfft-lp2:nfft]=np.multiply(xf[nfft-lp2:nfft],np.transpose(wl2))
    
    qf1=lf1
    lq2=int(np.fix((qf1/p['fs'])*nfft+1))
    qf2=lf2
    lq22=int(np.fix((qf2/p['fs'])*nfft+1))# fix(0.5*lp2);
    ql1=np.zeros(lq22)
    ql1[lq2:lq22]=0.5-0.5*np.cos(np.pi*(np.arange(lq22-lq2)+1)/(lq22-lq2))
    ql2=ql1[::-1]
    xf2[0:lq22]=np.multiply(xf2[0:lq22],ql1)
    xf2[nfft-lq22:nfft]=np.multiply(xf2[nfft-lq22:nfft],ql2)
    
    xi=np.real(np.fft.ifft(xf2,nfft))
    xi=xi[0:no_samp]
    
    return xi


def freq_filt(unfiltered_data,p):
    """filter data in frequency domian

    Args:
        unfiltered_data (dictionary): data
        p (dictionary): contains all parameters of the dataset
   

    Returns:
        dictionary: filtered data
    """
    filtered_data={}
    
    # define different frequency band
    if p['freq_band']   ==  'delta':
        lf1     =   0.5
        lf2     =   1
        hf2     =   4
        hf1     =   4.5
    elif p['freq_band'] ==  'theta':
        lf1     =   3.5
        lf2     =   4
        hf2     =   8
        hf1     =   8.5
    elif p['freq_band'] ==  'alpha':
        lf1     =   7.5
        lf2     =   8
        hf2     =   12
        hf1     =   12.5
    elif p['freq_band'] ==  'sigma':
        lf1     =   12.5
        lf2     =   13
        hf2     =   16
        hf1     =   16.5
    elif p['freq_band'] ==  'beta':
        lf1     =   16.5
        lf2     =   17
        hf2     =   25
        hf1     =   25.5
    elif p['freq_band'] ==  'gama':
        lf1     =   29.5
        lf2     =   30
        hf2     =   40
        hf1     =   40.5
    elif p['freq_band'] ==  'slow':
        lf1     =   0.5
        lf2     =   1
        hf2     =   6
        hf1     =   6.5
    elif p['freq_band'] ==  'normal':
        lf1     =   0.5
        lf2     =   1
        hf2     =   50
        hf1     =   50.5
        
    
    for i in unfiltered_data:

        filtered_data[i]=np.zeros(np.shape(unfiltered_data[i]))
        

        for j in range(np.shape(unfiltered_data[i])[0]):
            for k in range(np.shape(unfiltered_data[i])[2]):
                filtered_data[i][j,:,k]=filt(unfiltered_data[i][j,:,k],p,lf1,lf2,hf2,hf1)       
    
    return filtered_data
