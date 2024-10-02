# import lib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# function to select best features based on Recursive feature elimination (RFE)
def fea_sel(data,Y,p):
    
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=1, verbose=False)
    rfe_selector.fit(data, Y)
    #rfe_support = rfe_selector.get_support()
    #rfe_feature = data.loc[:,rfe_support].columns.tolist()
    
    # rank all features
    X=np.array(data.columns)
    Y=rfe_selector.ranking_

    Z = [x for _,x in sorted(zip(Y,X))]
    
    df=pd.DataFrame()
    
    for i in (Z):
        df[Z]=data[Z]
        
    # return only top p['max_features'] features
    return df.iloc[:,0:p['max_features']],df.iloc[:,0:p['max_features']].columns
