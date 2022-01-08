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

import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import gaussian_kde, ks_2samp
import numpy as np
import pandas as pd
import difflib
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold,StratifiedKFold
from numpy import newaxis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
'''
Calculate Bhattacharyya Distance between two distributions
'''
def bhattacharyya(a, b):
    """ Bhattacharyya distance between distributions (lists of floats). """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))


def similarity(a,b):
    a=list(a)
    b=list(b)
    sm=difflib.SequenceMatcher(None,a,b)
    sm_ratio = sm.ratio()*100
    return sm_ratio

def p_value(a,b):
#    a=list(a)
#    b=list(b)
    t,p =ttest_ind(a,b)
    return t,p
'''
Find the overlap percentage between two histograms
'''
def overlap(setA_df, nor_df, num_bins,feature_names, path_plot_save,label):
    overlap=pd.DataFrame(columns=feature_names,data=np.zeros((1, len(feature_names))))
    p_value_df=pd.DataFrame(columns=feature_names,data=np.zeros((1, len(feature_names))))
    t_value_df=pd.DataFrame(columns=feature_names,data=np.zeros((1, len(feature_names))))
    discr_power_df =pd.DataFrame(columns=feature_names,data=np.zeros((1, len(feature_names))))
    label=label.capitalize()
    for i in range(setA_df.shape[1]):
        
        title='%s'%feature_names[i]
        title = title.replace(':','_')
        data1 =setA_df.iloc[:,i]
        data2=nor_df.iloc[:,i]
#        sm_ratio = similarity(data1,data2)
        t,p = p_value(data1,data2)
#        b_dist = bhattacharyya(data1,data2)
        xmin = min(data1.min(), data2.min())
        xmax = max(data1.max(), data2.max())
        bins = np.linspace(xmin, xmax, num_bins)
        weights1 = np.ones_like(data1) / float(len(data1))
        weights2 = np.ones_like(data2) / float(len(data2))
        
        hist_1 = np.histogram(data1, bins, weights=weights1)[0]
        hist_2 = np.histogram(data2, bins, weights=weights2)[0]
        
        tvd = 0.5*sum(abs(hist_1 - hist_2))
#        minus_tvd = 1-tvd
        print("overlap: {:2.2f} percent".format((1-tvd)*100))
        overlap_perc=((1-tvd)*100)
        discriminative_power = 100-overlap_perc
        discr_power_df.iloc[:,i]=discriminative_power
        overlap.iloc[:,i]=overlap_perc
        p_value_df.iloc[:,i]=p
        t_value_df.iloc[:,i]=t
        plt.figure()
        ax = plt.gca()
        ax.hist(data1, bins, weights=weights1, color='red', edgecolor='white', alpha=0.5)[0]
        ax.hist(data2, bins, weights=weights2, color='blue', edgecolor='white', alpha=0.5)[0]
        plt.legend(('%s'%label,'Control'))
        plt.title("{}, Discriminative power: {:2.2f} %".format(title, (100-(1-tvd)*100)))
        plt.savefig(path_plot_save+'%s_hist.png'%title)
        plt.close()
    return overlap,discr_power_df,p_value_df,t_value_df

'''
DATA VISUALIZATION
'''
def density_plot(setA_df,nor_df,feature_names, path_plot_save, label):
    label=label.capitalize()

    '''
    1. DENSITY PLOTS
    '''
#path_plot_save ='/home/irem/Desktop/ACDC_Test/AF_CLF_FUSION_PLOTS/Most_common_feats_density_plots/'
### plot of normal and minf subjects with the most frequent feature
    for i in range(nor_df.shape[1]):
#        [overlap_perc, discriminative_power] = overlap(setA_df, nor_df, 100,feature_names, path_plot_save,label)
        p1 =sns.kdeplot(setA_df.iloc[:,i],kernel = 'gau', legend=True,  shade = True, color='r', label='%s'%label)
        p2 =sns.kdeplot(nor_df.iloc[:,i],kernel = 'gau', legend=True,   shade = True, color='b', label='Control')
        title ='%s'%(feature_names[i])
        title = title.replace(':','_')

        plt.title('%s'%title)
        plt.savefig(path_plot_save+'%s.png'%title)
        plt.close()
   
    return p1, p2
    
def dist_plot (setA_df,nor_df_training,feature_votes_sorted, path_plot_save, label):
    label=label.capitalize()
   
    '''
    2 . Flexibly plot a univariate distribution of observations.
    '''
#path_plot_save ='/home/irem/Desktop/ACDC_Test/AF_CLF_FUSION_PLOTS/Most_common_feats_density_plots/'
### plot of normal and minf subjects with the most frequent feature
    for i in range(nor_df_training.shape[1]):
        p1 =sns.distplot(setA_df.iloc[:,i], color='r', label='%s'%label)
        p2 =sns.distplot(nor_df_training.iloc[:,i],color='b', label='Control')
        title ='%s'%feature_votes_sorted[i]
        plt.title('%s'%title)
        plt.savefig(path_plot_save+'%s_dist.png'%title)
        plt.close()
    return p1, p2


'''
BOX PLOT
'''
def box_plot(setA_df, nor_df, feature_names, path_to_save):
    fig, ax = plt.subplots()
    green_diamond = dict(markerfacecolor='g', marker='D')
    ax.set_title('%s'%feature_names[i])

    data_A = [setA_df , nor_df  ]
    ax.boxplot(data_A, flierprops=green_diamond)
    plt.xticks([1, 2], ['R1', 'Control'])
    fig.savefig(path_to_save+'%s_boxplot.png' %feature_names[i], dpi=150)

def find_max_cv_scores (cv_score):
    id_max_feat = np.argmax(cv_score.values)
    max_feat_name = cv_score.columns[id_max_feat]
    max_feat_accuracy = cv_score.iloc[:,id_max_feat]
    max_feat_accuracy =max_feat_accuracy[0]
    return max_feat_name,max_feat_accuracy
    
def svm_acc_feature(data1,data2, label, clf):
#    kf = KFold(n_splits=10)
    data=pd.concat([data1,data2])
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    Feature_scl = scaler.fit_transform(data)
    cv_score = pd.DataFrame(columns=data.columns, data=np.zeros((1, data.shape[1])))
#    data=data.iloc[:,1:]
    for i in range(data.shape[1]):
        f = Feature_scl[:,i]
        f = f[:,newaxis]
        print(i)
        cv_score.iloc[:,i] = np.mean(cross_val_score(clf, f, label, cv=20))
    max_feat_name,max_feat_accuracy = find_max_cv_scores(cv_score)
    return cv_score, max_feat_name, max_feat_accuracy               
        
