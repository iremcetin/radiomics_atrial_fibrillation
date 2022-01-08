# -*- coding: utf-8 -*-
"""
Created on Mar 8 2019
@author: Irem Cetin
email : irem.cetin@upf.edu
################################################################################
THIS SCRIPT IS FOR REPRODUCING CARS 2019 RESULTS WITH THE SAVED SAMPLES
Tested with Python 2.7 and Python 3.5 on Ubuntu Mate Release 16.04.5 LTS (Xenial Xerus) 64-bit
###############################################################################
Analysis of AFIB using radiomics in UK Biobank dataset (ENSEMBLE LEARNING)
This script :
* reads randomly selected samples from UK Biobank (equal size of AFIB and Normal cases)
        --> this script analyzes atrial fibrillation with id 7, 
        --> Creates 10 different training and testing sets consists of the same AFIB
            and different healthy subjects
        --> Split training and testing with the ratio of 4:1
          Training :  45 AFIB and randomly sampled different 45 healthy subjects
          Testing :   15 AFIB and randomly sampled different 15 healthy subjects
          
* Preprocess the training and testing dataset is done 
by scaling the datasets to a given range
        --> For this reason, the range (-1,1) is used
        
* each classifier is trained on different (10) datasets
        --> For this reason, each classifier consists of different feature subsets
        --> for this experiment, SVM with RBF kernel is selected
* Feature selection is done using Sequential forward floating feature selection
(SFFS) and training dataset is used to select the best feature subset in a range 
of [2,20] with 10 fold cross validation.

If needed, feature selection and classifier parameters can be changed 
forward = False -->  Sequential backward feature selection
floating = False --> Sequtian feature selection w/o using floating approach
cv= # of folds


################################################################################

"""

'''
IMPORT LIBRARIES
'''
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from collections import Counter
from scipy.stats import mode

### INPUT FILES #### ##########################################################
## Define the directories TO READ
path_to_read ='/home/irem/Desktop/ACDC_Test/AFIB_Classifier_Fusion/'
count_model =10 # this will build the model for 10 times = Number of sets
### indices for 

Afib = 7


      
#### List of CVDs in UK Biobank  ##############################################
cvds_classify=['angina',\
               'stroke', \
               'transient ischaemic attack (tia)',\
               'peripheral vascular disease',\
               'deep venous thrombosis (dvt)', \
               'heart valve problem/heart murmur',\
               'cardiomyopathy',\
               'atrial fibrillation',\
               'irregular heart beat', \
               'heart/cardiac problem', \
               'raynaud\'s phenomenon/disease',\
               'heart attack/myocardial infarction',\
               'high cholesterol', \
               'hypertension','essential hypertension',\
               'hypertrophic cardiomyopathy (hcm / hocm)',\
               'aortic regurgitation',\
               'aortic stenosis',\
               'aortic aneurysm',\
               'vasculitis',\
               'varicose veins',\
               'mitral valve prolapse',\
               'pericardial problem',\
               'pericardial effusion',\
               'pericarditis',\
               'svt / supraventricular tachycardia',\
               'wegners granulmatosis'
       ]
       
       
       
cvds_samples=[] 
cvd_classifier_acc=[]  
acc_all=[]
models=[]

'''
READ the radiomics features for Normals and AFIB (if needed)
'''            
nor_df = pd.read_csv(path_to_read+'normal_all.csv', low_memory = False)
setA_df = pd.read_csv(path_to_read+'setA_all.csv', low_memory = False)

#Number of samples for testing and training
### take 4:1 ratio for training and testing
sample_testing=int((setA_df.shape[0]*0.25)) # in AFIB take 15 for testing 
sample_training=(setA_df.shape[0]-sample_testing) #& 45 for training

'''
READ the sets 
AFIB = 60 samples (45 training and 15 testing)
Normal : Training = 45 * number of sets, Testing : 15 * number of sets
'''

nor_df_testing = pd.read_csv(path_to_read+'normal_df_testing.csv', low_memory = False)
setA_df_testing =pd.read_csv(path_to_read+'setA_df_testing.csv', low_memory = False)
######################################################################################
nor_df_training = pd.read_csv(path_to_read+'normal_df_training.csv', low_memory = False)
setA_df_training = pd.read_csv(path_to_read+'setA_df_training.csv', low_memory = False)
######################################################################################

df_all_training =pd.concat([nor_df_training, setA_df_training])

df_all_testing = pd.concat([nor_df_testing, setA_df_testing])
###############################################################################
Labels_training =df_all_training['label']
###############################################################################
Labels_testing =df_all_testing['label']
'''
PREPROCESS THE TRAINING AND TESTING SETS
'''

df_training = df_all_training.iloc[:,3:-1]
df_testing = df_all_testing.iloc[:,3:-1]
scaler=MinMaxScaler(feature_range=(-1,1))
Features_scl_train=scaler.fit_transform(df_training.values)
Features_scl_test= scaler.transform(df_testing.values)


'''
PREPARE TRAINING AND TESTING SETS
'''
#Split the array to train and test
######## TRAINING #############################################################
setA_scl_train =Features_scl_train[-sample_training:,:]
Labels_setA_scl_train =Labels_training[-sample_training:]
nor_scl_train =Features_scl_train[:sample_training*count_model,:]
Labels_nor_scl_train =Labels_training[:sample_training*count_model]
#split normal samples training into different groups of arrays
nor_scl_train_split=np.split(nor_scl_train, count_model)
Labels_nor_scl_train_split =np.split(Labels_nor_scl_train, count_model)
####### TESTING ############################################################### 
setA_scl_test =Features_scl_test[-sample_testing:,:]
Labels_setA_scl_test =Labels_testing[-sample_testing:]
nor_scl_test =Features_scl_test[:sample_testing*count_model,:]
Labels_nor_scl_test =Labels_testing[:sample_testing*count_model]
#split normal samples training into different groups of arrays
nor_scl_test_split =np.split(nor_scl_test, count_model)
Labels_nor_scl_test_split =np.split(Labels_nor_scl_test, count_model)

'''
BUILDING THE MODEL / FEATURE SELECTION / TRAINING 
'''
for i in range(count_model): 
##################### Get samples for each set#################################
##################### combine AFIB cases and different normal subjects ########
                X_train =np.concatenate((nor_scl_train_split[i], setA_scl_train))
                Y_train = np.concatenate((Labels_nor_scl_train_split[i],Labels_setA_scl_train ))
#               
################### Define the classifier #####################################
                clf = SVC(kernel='rbf', 
                          decision_function_shape='ovr',
                          C=10, gamma=0.1,
                          class_weight='balanced',
                          probability=True,
                          random_state=42)
################## Define the feature selector ################################
                sfs1 = SFS(clf, 
                           k_features=(2,20), # define the number of features or the range 
                           forward=True, 
                           floating=True, 
                           verbose=2,
                           scoring='accuracy', 
                           cv=10) ## Select the features using cv
### Select features on Training set    ######################################## 
                sfs1 = sfs1.fit(X_train, Y_train) 
        
#### Generate the new subsets based on the selected features ##################
## this line is equivalent to :
#X_train_sfs = Features_scl_train[:,sfs1.subsets_[k]["feature_idx"]]   
                X_train_sfs =sfs1.transform(X_train)

###############################################################################
##### Building the classifier using selected features #########################
#### train the classifier using training dataset with the selected features ###
#        
                clf.fit(X_train_sfs,Y_train)
### Save fitted/learned classifiers and feature selection results #############
                models.append((sfs1, clf, sfs1.subsets_)) 

'''
PREDICTION / TESTING
'''               
results=[]               
set_results = np.zeros((count_model, len(models)))
set_number_of_feats=[]
for j in range(count_model):
      X_test =np.concatenate((nor_scl_test_split[j], setA_scl_test))
      Y_test =np.concatenate((Labels_nor_scl_test_split[j],Labels_setA_scl_test ))
      for i in range(len(models)): #### number of fitted classifiers
                sfs=models[i][0] ### Get first classifier and corresponding feature selection
                clf = models[i][1]             
                X_test_sfs = sfs.transform(X_test) ### Select only feature subset
                y_pred =clf.predict(X_test_sfs) ## Predict the testing data
                acc = float((Y_test == y_pred).sum()) / y_pred.shape[0] ### Calculate the accuracy
                print('Set %d Classifier %d test set accuracy: %.2f %%' % (j, i, acc * 100))
                ### save the results
                results.append((i,j,acc,y_pred,sfs,clf)) ### save the results for each classifier
### results consists of (classifier_id, testing_set_id, predictions, sfs,classifier)
                set_results[j,i] =np.format_float_scientific(acc) # results of each classifier for each set = INDIVIDUAL TESTING RESULTS
                set_number_of_feats.append((len(sfs.k_feature_idx_), sfs.k_score_, sfs.k_feature_idx_))
'''
FUSION OF CLASSIFIERS / MAJORITY VOTING
'''
################### Majority voting ############################################
y_pred_mv=np.zeros((len(models),len(X_test)))

y_pred_all=[]
for i in range(count_model):
    for j in range(count_model):
        y_pred_all.append(results[i+j][3])
       
    y_pred_mv[i,:]=mode(y_pred_all,axis=0)[0]

'''
SCORES OF THE TESTING SETS
'''    
cvd_acc=np.zeros((count_model))
for i in range(count_model):
    Y_test =np.concatenate((Labels_nor_scl_test_split[j],Labels_setA_scl_test ))
    cvd_acc[i] = float((Y_test == y_pred_mv[i]).sum()) / y_pred_mv[i].shape[0]
#
'''
TAKE THE AVERAGE OF THE SCORES
'''    
final_average_acc=np.mean(cvd_acc)
print('Final accuracy : %f'%final_average_acc)


'''
Find the most common features in the feature subsets
'''

name =[]
ids=[]
for i in range(len(models)):
    sfs = models[i][0]
    feature_idx = sfs.k_feature_idx_
    ids.append(feature_idx)
    for j in feature_idx:
        name.append((df_all_training.columns[j+3]))

feature_ids = Counter([x for t in ids for x in set(t)])
feature_ids_sorted =feature_ids.most_common(len(name))
feature_ids_top10 = [item[0] for item in feature_ids_sorted][:90]

duplicates=[]

for i in range(len(ids)):
    duplicates.append( set(ids[i]).intersection(feature_ids_top10))

feature_ids_top10 = np.array(feature_ids_top10)
feature_votes = (Counter(name))
feature_votes_sorted =feature_votes.most_common(len(name))



count=0
feature_count =[5,10,15,20,25,30,35,40]
score_last=np.zeros((len(feature_count)))
for i in(feature_count):
    '''
    Build classifier with only frequently selected features
    '''
    features_last_training = df_training.iloc[:,feature_ids_top10[:i]]
    features_last_testing =df_testing.iloc[:,feature_ids_top10[:i]]
    '''
    Preprocess
    '''
    scaler = MinMaxScaler(feature_range=(-1,1))
    features_training = scaler.fit_transform(features_last_training)
    features_testing = scaler.transform(features_last_testing)

    clf = SVC(kernel='rbf', 
          decision_function_shape='ovr',
          C=10, gamma=0.1,
          class_weight='balanced',
          probability=True,
          random_state=42)

    clf.fit(features_training, Labels_training)
    y_pred = clf.predict(features_testing)
    score_last[count] = clf.score(features_testing, Labels_testing)
    print('Testing score using %d selected features: %f'%(i, score_last[count]))
    count=count+1
    
#import seaborn as sns
final_results =pd.DataFrame(columns =['Number_of_features', 'scores'])
final_results.iloc[:,0]=feature_count
final_results.iloc[:,1]=score_last
final_results.plot(x='Number_of_features', y='scores', kind ='line', title ='Atrial Fibrillation')


import dill                            #pip install dill --user
filename = '/home/irem/Desktop/ACDC_Test/Methods/globalsave_AFIB_ensemble_features.pkl'
dill.dump_session(filename)
#'''
#DATA VISUALIZATION
#'''
#'''
#1. DENSITY PLOTS
#'''
#path_plot_save ='/home/irem/Desktop/ACDC_Test/AF_CLF_FUSION_PLOTS/Most_common_feats_density_plots/'
#### plot of normal and minf subjects with the most frequent feature
#import matplotlib.pyplot as plt
#for i in range(len(feature_ids_top10)):
#    p1 =sns.kdeplot(setA_df_training.iloc[:,i+3],shade = True, color='r', label='AF')
#    p1 =sns.kdeplot(nor_df_training.iloc[:,i+3],shade = True, color='b', label='Control')
#    title ='%s'%feature_votes_sorted[i][0]
#    plt.title('%s'%title)
#    plt.savefig(path_plot_save+'%s.png'%title)
#    plt.close()
#    
#'''
#2. HEATMAP OF RADIOMICS --> NOT WORKING
#'''
##
##path_heatmap = '/home/irem/Desktop/ACDC_Test/AF_CLF_FUSION_PLOTS/Heatmap/'
##
##
##sns.heatmap(setA_df, annot=True)
#
#
#'''
#3. CORRELOGRAM
#'''
#
#sns.pairplot(setA_df_training)
#sns.plt.show()