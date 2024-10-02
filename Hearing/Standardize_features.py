# import lib
import numpy as np

# Function to standardize features
# Standardize control dataset
def MR_control(data):
    col_mean=[0]*(len(data.iloc[0])-2)
    col_std=[0]*(len(data.iloc[0])-2)
    
    for i in range(len(data.iloc[0])-2):
        col_mean[i]=np.mean(data.iloc[:,i])
        data.iloc[:,i]=data.iloc[:,i]-col_mean[i]
        col_std[i]=np.std(data.iloc[:,i])
        data.iloc[:,i]=data.iloc[:,i]/col_std[i]
    
    return data, col_mean, col_std

# Standardize TBI dataset using parameters obtained for control data
def MR_tbi(new_big_df,col_mean_test,col_std_test):
    
    for i in range(len(new_big_df.iloc[0])-2):
        #col_mean[i]=np.mean(new_big_df.iloc[:,i])
        new_big_df.iloc[:,i]=new_big_df.iloc[:,i]-col_mean_test[i]
        #col_std[i]=np.max(new_big_df.iloc[:,i])
        new_big_df.iloc[:,i]=new_big_df.iloc[:,i]/col_std_test[i]
    
    return new_big_df