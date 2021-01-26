import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix as cm
from sklearn import preprocessing
from Naive_Bayes import Multi_NB
from SMOTE import Smote

def main():
    
    """ Read Data and Create Numpy arrays"""
    data = pd.read_csv("haberman.data", sep=",", header=None )
    data = data.values
    X = data[:,:3]
    #X = preprocessing.normalize(X, axis=1)
    #print(X)
    Y = data[:, 3];

    """Create class labels"""
    yNFP = np.where(Y==2,1,0)

    """Minority class instances"""
    unique_minority, counts_minority = np.unique(yNFP, return_counts=True)
    minority_shape = dict(zip(unique_minority, counts_minority))[1]
    minority_x = np.ones((minority_shape, X.shape[1]))
    minority_y = np.ones((minority_shape))
    j=0
    for i in range(0,X.shape[0]):
        if yNFP[i] == 1.0:
            minority_x[j] = X[i]
            j += 1
        
    """Majority class instances"""       
    unique_majority, counts_majority = np.unique(yNFP, return_counts=True)
    majority_shape = dict(zip(unique_majority, counts_majority))[0]       
    majority_x = np.ones((majority_shape, X.shape[1]))
    majority_y = np.zeros((majority_shape))

    k = 0
    for i in range(0,X.shape[0]):
        if yNFP[i] == 0.0:
            majority_x[k] = X[i]
            k += 1


    """Split Majority an Minirity Instances to 3 partitions each"""
    xmaj_1,xmaj_2,xmaj_3 = np.array_split(majority_x,3)
    ymaj_1,ymaj_2,ymaj_3 = np.array_split(majority_y,3)

    xmin_1,xmin_2,xmin_3 = np.array_split(minority_x,3)
    ymin_1,ymin_2,ymin_3 = np.array_split(minority_y,3)


    """Create 3 Folds of Majority Instances """
    fold_of2x_maj1 = np.concatenate((xmaj_1,xmaj_2))
    fold_of2y_maj1 = np.concatenate((ymaj_1,ymaj_2))

    fold_of2x_maj2 = np.concatenate((xmaj_2,xmaj_3))
    fold_of2y_maj2 = np.concatenate((ymaj_2,ymaj_3))
  
    fold_of2x_maj3 = np.concatenate((xmaj_1,xmaj_3))
    fold_of2y_maj3 = np.concatenate((ymaj_1,ymaj_3))
    
    """--------------------------------------------------------------------------------------------"""
    
    """Create 3 Folds of Minority Instances """
    fold_of2x_min1 = np.concatenate((xmin_1,xmin_2))
    fold_of2y_min1 = np.concatenate((ymin_1,ymin_2))
 
    fold_of2x_min2 = np.concatenate((xmin_2,xmin_3))
    fold_of2y_min2 = np.concatenate((ymin_2,ymin_3))
  
    fold_of2x_min3 = np.concatenate((xmin_1,xmin_3))
    fold_of2y_min3 = np.concatenate((ymin_1,ymin_3))
    
    """--------------------------------------------------------------------------------------------"""
  
    for k in [3]:
        for smote_instances in [700]:
            ptest_res = np.array([])
            y_test_res = np.array([])
            
            sm = Smote(k,'dice',smote_instances)
            cl = Multi_NB()
            
            X_train = np.concatenate((fold_of2x_maj1,fold_of2x_min1))
            y_train = np.concatenate((fold_of2y_maj1,fold_of2y_min1))
            X_test = np.concatenate((xmaj_3,xmin_3))
            y_test = np.concatenate((ymaj_3,ymin_3))
            
            x_smote = sm.smote(fold_of2x_min1)
            X_train = np.concatenate((X_train,x_smote))
            y_smote = np.ones((x_smote.shape[0]))
            y_train = np.concatenate((y_train,y_smote))
            
            model = cl.fit(X_train,y_train)
            ptesting = model.predict(X_test)
            ptest_res = np.append(ptest_res,ptesting )
            y_test_res = np.append(y_test_res,y_test)
            ############################################
            
            X_train = np.concatenate((fold_of2x_maj2,fold_of2x_min2))
            y_train = np.concatenate((fold_of2y_maj2,fold_of2y_min2))
            X_test = np.concatenate((xmaj_1,xmin_1))
            y_test = np.concatenate((ymaj_1,ymin_1))
            
            x_smote = sm.smote(fold_of2x_min2)
            X_train = np.concatenate((X_train,x_smote))
            y_smote = np.ones((x_smote.shape[0]))
            y_train = np.concatenate((y_train,y_smote))
            
            model = cl.fit(X_train,y_train)
            ptesting = model.predict(X_test)
            ptest_res = np.append(ptest_res,ptesting )
            y_test_res = np.append(y_test_res,y_test)
            ############################################
            
            X_train = np.concatenate((fold_of2x_maj3,fold_of2x_min3))
            y_train = np.concatenate((fold_of2y_maj3,fold_of2y_min3))
            X_test = np.concatenate((xmaj_2,xmin_2))
            y_test = np.concatenate((ymaj_2,ymin_2))
            
            x_smote = sm.smote(fold_of2x_min3)
            X_train = np.concatenate((X_train,x_smote))
            y_smote = np.ones((x_smote.shape[0]))
            y_train = np.concatenate((y_train,y_smote))
            
            model = cl.fit(X_train,y_train)
            ptesting = model.predict(X_test)
            ptest_res = np.append(ptest_res,ptesting )
            y_test_res = np.append(y_test_res,y_test)
    
            """Confusion Matrix"""
            conf_mat = cm(y_test_res,ptest_res)
            tn,fp,fn,tp  = cm(y_test_res,ptest_res).ravel()
            print(conf_mat)
            print("CM for ",k,"K, and ",smote_instances," SMOTEs:" )
            print("Total size:",str(tp+fp+fn+tn))
            print("True Positives:",tp)
            print("False Positives:",fp)
            print("False Negatives:",fn)
            print("True Negatives:",tn)
            print("Results for ",k,"K, and ",smote_instances," SMOTEs:" )
            accuracy = (tp+tn)/(tp+fp+fn+tn)
            precision  = tp/(tp + fp)
            recall = tp/(tp+fn)
            f1_score = 2*(precision*recall)/(precision+recall)
            print("-----------------------------------")
            print("Accuracy:",accuracy)
            print("Precision:",precision)
            print("Recall:",recall)
            print("F1 Score:",f1_score)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
if __name__ == "__main__":
	main()