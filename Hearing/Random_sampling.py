# import lib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier

from Feature_selection import fea_sel
from Standardize_features import MR_control

from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

# function to calculate Random Sampling accuracy
def RS(Total_dataframe_human,p):
    """calculate 10 fold cross validation metrics and print accuracy

    Args:
        Total_dataframe_human (dataframe): entire dataset in dataframe format
        p (dictionary): contains all parameters of the dataset

    Returns:
        Random_training (dataframe): data
        Random_Y (dataframe): corresponding labels
    """
    
    # choose number of features based on number of datapoint
    p['max_features']=int(np.sqrt(len(Total_dataframe_human)))
    print("No. of features = %d"%(p['max_features']))

    # shuffle dataset
    Random=shuffle(Total_dataframe_human)

    # standardize dataset
    Random, col_mean_test, col_std_test = MR_control(Random)

    Random_train_fea = Random.iloc[:,0:-2]
    Random_Y = Random.iloc[:,-1]
    Random_training,random_selected_fea=fea_sel(Random_train_fea,Random_Y,p)

    # apply ML
    dtree = DecisionTreeClassifier()
    random_dtree_scores = cross_val_score(dtree, Random_training, Random_Y, cv=10)

    classifier = KNeighborsClassifier(n_neighbors=5)
    random_k1_scores = cross_val_score(classifier, Random_training, Random_Y, cv=10)

    classifier = KNeighborsClassifier(n_neighbors=11)
    random_k2_scores = cross_val_score(classifier, Random_training, Random_Y, cv=10)

    classifier = KNeighborsClassifier(n_neighbors=19)  
    random_k3_scores = cross_val_score(classifier, Random_training, Random_Y, cv=10)

    #mlp = MLPClassifier(hidden_layer_sizes=(100,10),max_iter=10000)
    #random_mlp_scores = cross_val_score(mlp, Random_training, Random_Y, cv=10)

    clf = RandomForestClassifier()
    random_rf_scores = cross_val_score(clf, Random_training, Random_Y, cv=10)

    svclassifier = SVC(kernel='rbf')
    random_svc_scores = cross_val_score(svclassifier, Random_training, Random_Y, cv=10)

    model = XGBClassifier(verbosity=0, use_label_encoder=False)
    random_XGB_scores = cross_val_score(model, Random_training, Random_Y.astype(int), cv=10)

    # print result
    print('=======================================================================')
    print('Random Sampling 10 fold CV')
    print('RS dtree accuracy %0.2f (+/-%0.2f)' % (np.mean(random_dtree_scores)*100,np.std(random_dtree_scores)*100))
    print('RS k5 accuracy %0.2f (+/-%0.2f)' % (np.mean(random_k1_scores)*100,np.std(random_k1_scores)*100))
    print('RS k11 accuracy %0.2f (+/-%0.2f)' % (np.mean(random_k2_scores)*100,np.std(random_k2_scores)*100))
    print('RS k19 accuracy %0.2f (+/-%0.2f)' % (np.mean(random_k3_scores)*100,np.std(random_k3_scores)*100))
    #print('RS NN accuracy %0.2f (+/-%0.2f)' % (np.mean(random_mlp_scores)*100,np.std(random_mlp_scores)*100))
    print('RS RF accuracy %0.2f (+/-%0.2f)' % (np.mean(random_rf_scores)*100,np.std(random_rf_scores)*100))
    print('RS SVM accuracy %0.2f (+/-%0.2f)' % (np.mean(random_svc_scores)*100,np.std(random_svc_scores)*100))
    print('RS XGBoost accuracy %0.2f (+/-%0.2f)' % (np.mean(random_XGB_scores)*100,np.std(random_XGB_scores)*100))
    print('=======================================================================')
    
    return Random_training,Random_Y