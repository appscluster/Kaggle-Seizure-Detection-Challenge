# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 09:49:14 2014

@author: SpectralMD2
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 04 12:43:11 2014

@author: Valued Customer
"""
#%%import scipy.io
import numpy as np
import os
import pylab as plt
import math
import pandas as pd
import time
import scipy.io

import pandas as pd
from scipy import signal
from scipy.fftpack import fft
import matplotlib.backends.backend_pdf
#trueSample = showFFT(trainTrue,len(train_1),trueSample,channels)
import statsmodels.api as sm
from sklearn.decomposition import FastICA, PCA
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import time
totalTime = time.clock()
output1 = []
Sample = []
pathSum = 'M:/Data to Analysis/'
dogFolder = os.listdir(pathSum)
count = 0

#%% save as hashtable to dog and channels
#dog1 =  dataClassify(dogFolder[0],pathSum,16)
#dog2 =  dataClassify(dogFolder[1],pathSum,16)
#dog3 =  dataClassify(dogFolder[2],pathSum,16) 
#dog4 =  dataClassify(dogFolder[3],pathSum,16) 
#dog5 =  dataClassify(dogFolder[4],pathSum,15) 
#patient1 =  dataClassify(dogFolder[5],pathSum,15)     
#patient2 =  dataClassify(dogFolder[6],pathSum,24) 

#item = dogFolder[0]
channel = [16,16,16,16,15,15,24]
hashItem = {}
for i in range(len(channel)):
    item =  dogFolder[i]
    channels = channel[i]

## get the name 

    startTime = time.time()
    print 'new patients or dogs begin %s' % item
    path = pathSum + item
    classTypepath = os.listdir(path)
    m = len(classTypepath)
    
    train_1 = []  # has ictal
    train_0 = [] # has interictal
    test = [] # has test
    
    
    for i in range(m):
        name = classTypepath[i].split('_')
        for j in range(len(name)):
            if name[j] == 'preictal': 
                train_1.append(i)
            elif name[j] == 'interictal':
                train_0.append(i)                
            elif name[j]== 'test':
                test.append(i)
           
    ica = FastICA(whiten =True,max_iter =30)
    #%% get the data     dataTest = getdata(path,test,channels,'testdata')
    def getdata(path,ClassType,channels,name):
        featureNumber = 8
        featureStore = pd.DataFrame(columns= ['channel' + str(i+1) for i in range(channels*featureNumber)])
        # how many features we will extract
        classTypepath = os.listdir(path)
        time_start = time.clock()
        for i in range(len(ClassType)):      
            data = scipy.io.loadmat(path+'/'+ classTypepath[ClassType[i]],struct_as_record=True)
            keys = data.keys()
            for item in keys:     
                item1 = item.split('_')
                if 'segment' in item1:
                    name = item
                    fftN = 4096
                    fs = 400
                    dataStore = pd.DataFrame(data[name]['data'][0][0])
#                    dataStore = pd.DataFrame(ica.fit_transform(data[name]['data'][0][0].T).T)
#                    for i in range(channels):                  
#                    dataStore1 = pd.rolling_mean(dataStore,window=20*400)
#                    dataStore[~dataStore.isnull()]
                    yChannel = abs(fft(dataStore,fftN)) #%% fft, from time domain to frequency domain
                    #%% spectral edge frequency:
#                    yChannel[:,6:512] = 0

                                            
                    f = np.linspace(0,fs,num = fftN)
                    maxSave = np.zeros((1,channels*featureNumber))
                    # 0 to 7
                    maxSave[0,0:channels] = 2*f[yChannel.argmax(axis=1)]
        ## find the maximum location
                    maxSave[0,channels:channels*2] = np.amax(yChannel,axis=1)  ## find the maximum value
                    #%% get the different bands of signal
#                    p = map(lambda x: x*x,np.sum(yChannel,axis=1))
                   # power = [x*x for x in np.sum(yChannel,axis=1)]
                    
                    # 0.5 Hz to 4 Hz: 0.5/fs * 1024 = 1.28 to 4/fs* 1024 = 10.25
                    power = np.amax(yChannel[:,1:256],axis=1)**(2)
                    maxSave[0,channels*2:channels*3] = np.sum(yChannel[:,5:41],axis=1)**(2)
                    
                    # 4 Hz to 8 Hz  4/fs * 1024 = 10.25 to 8/fs* 1024 = 20.5
                    maxSave[0,channels*3:channels*4] = np.sum(yChannel[:,41:82],axis=1)**(2)
                    # 8 Hz to 13 Hz  8/fs * 1024 = 20.5 to 13/fs* 1024 = 33.125
                    maxSave[0,channels*4:channels*5] = np.sum(yChannel[:,82:133],axis=1)**(2)
                    # 13 Hz to 30 Hz  13/fs * 1024 = 33.125 to 30/fs* 1024 = 76.875
                    maxSave[0,channels*5:channels*6] = np.sum(yChannel[:,133:308],axis=1)**(2)
                    # 30 Hz to 100 Hz 30/fs * 1024 = 76.875 to 100/fs* 1024 = 256.25
                    maxSave[0,channels*6:channels*7] = np.sum(yChannel[:,308:1024],axis=1)**(2)
                    
                    #%% matrix to each power band:
                    powerBand = maxSave[0,channels*2:channels*7].reshape([5,channels])
                    # caculate total power
                    p = map(lambda x: x*x,np.sum(yChannel,axis=1))
#                    maxValue = np.amax(powerBand,axis=0)
                    #
                    maxSave[0,channels*2:channels*7] = np.divide(powerBand,p).reshape(1,5*channels)
               #     np.divide(maxSave[0,32:].reshape([5,16]),np.amax(maxSave[0,32:].reshape([5,16]),axis=0)).reshape(1,80)
#                    maxSave[0,32:] = aaa.reshape(1,80)  
                    #%% 0-40 Hz, 
                    
                    p40Hz = (np.sum(yChannel[:,0:410],axis=1)**(2))*0.5
                    
                    spectralEdge = []
                    for jj in range(channels):
                        for ii in range(fftN/2):
                            N = float(ii)/fs*fftN
                            power = np.sum(yChannel[jj,:N])**(2)
                            if power>=p40Hz[jj]:
                                spectralEdge.append(N*fs/fftN)
                                break
                    maxSave[0,channels*7:channels*8] = np.array(spectralEdge)               
                    featureStore.loc[i] = maxSave
                    
                    break
        print  'the time to get the %s output is %0.2f s' %  (name, time.clock() - time_start)
        return featureStore
    dataTest = getdata(path,test,channels,'testdata')
    dataTrue = getdata(path,train_1,channels,'dataTrue')
    dataFalse = getdata(path,train_0,channels,'dataFalse')
    #    test = None
    #%% data 
    label = pd.DataFrame(columns=['label'])
    m = len(dataTrue)
    n = len(dataFalse)
    
    for i in range(m+n):
        if i<m:
            label.loc[i] = 1
        else:
            label.loc[i] = 0
    trainData = pd.concat([dataTrue,dataFalse])
    #%% free the memory
    dataFalse = None
    dataTrue = None
    #%% do the cross validation:
    from sklearn.cross_validation import StratifiedKFold
    from scipy import interp
    from sklearn.cross_validation import KFold
    from sklearn.cross_validation import LeavePOut
    from sklearn.metrics import roc_curve, auc
    from sklearn.cross_validation import StratifiedShuffleSplit
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    
    
    lpo = LeavePOut(len(label), p=20)
    
    sss = StratifiedShuffleSplit(label, 10, test_size=0.3, random_state=1)
    
    classifier = svm.SVC(probability=True)
    from sklearn import preprocessing
    # cd '' call the function, z-score, mean = 0, normal = 1
    
    X= preprocessing.scale(trainData)  
    y = label.as_matrix().ravel()
    clf_gnb = GaussianNB()
    
    i = 0
    localMin = 0
    similiarDis = []
    
    from sklearn.metrics import jaccard_similarity_score
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.grid_search import GridSearchCV
    if channels <=16:
        anova_filter = SelectKBest(f_regression, k=int(0.65*(trainData.shape[1])))
    else:
        anova_filter = SelectKBest(f_regression, k=int(0.55*(trainData.shape[1])))
    #%% filter, univaraince analys, lasso
    X = anova_filter.fit(X,y).transform(X)
#    X_new = LinearSVC(C=1, penalty="l1", dual=False,class_weight ='auto').fit_transform(X, y)
#    print 'the feature we selected to data processing %d ' % (X_new.shape[1])

#    clfSVM = svm.SVC(probability=True,class_weight='auto',C=1000,gamma=0.0001,shrinking=True)
#    anova_svm = Pipeline([('anova', anova_filter), ('svc', clfSVM)])
    cv = StratifiedKFold(y, n_folds=3)
    est = svm.SVC(probability=True,class_weight='auto',shrinking=True)
    gamma_range = 10.0**np.arange(-5,4)
    C_range = 10.0**np.arange(-2,9)
    kernel_range = ('poly','rbf','sigmoid')
    param_grid = dict(gamma=gamma_range,C=C_range,kernel=kernel_range)
    grid = GridSearchCV(est,param_grid=param_grid,cv = cv,refit =True)
    grid.fit(X, y.astype(str)) 
    clf = grid.best_estimator_
    print 'the best score is %f ' %  grid.best_score_  
#    clf_para = grid.best_params_
    #%% get the best parameters to construct new classifiers
#    clf1 = svm.SVC(clf_para)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    plt.figure()
    for train_index, test_index in sss: #    print("TRAIN:", train_index, "TEST:", test_index)
        i +=1
        clf.fit(X.iloc[train_index], np.ravel(y.iloc[train_index]))
#        grid = GridSearchCV(est,param_grid=param_grid,cv = sss,refit =True)
        probas_ = clf.predict_proba(X.iloc[test_index])
        fpr, tpr, thresholds = roc_curve(y.iloc[test_index], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        localMax = max(roc_auc,localMin)
        
        #%% get the Jaccard coeffcient
        y_pred = list(clf.predict(X.iloc[test_index]))
        y_true = y.iloc[test_index][0].values.tolist()
        similiarDis.append(jaccard_similarity_score(y_pred,y_true))
        
        # get the best one classfiers from the model selection
        if localMax == roc_auc:
            clf_opt = clf
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    output1.append(clf_opt.predict_proba(anova_filter.transform(preprocessing.scale(dataTest)))[:,1])
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(sss)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

print 'finish the time to get the output'
print  'the total time to get the output is %0.2f s' %  (time.clock() - totalTime )
#%% to write csv to submit
a = output1[0]
for i in range(1,len(output1)):
    print 'the %d data size is %d' % (i, len(output1[i]))
    a = np.concatenate((a,output1[i]))
sample = pd.read_csv('M:/data processing/sampleSubmission.csv')
sample['preictal'] = a
sample.to_csv('seizureDetection.csv', index = False)



#%% data to the model
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), \
#                         algorithm="SAMME",  \
#                         n_estimators=200)
#
#
#from sklearn.metrics import confusion_matrix
#def classProb(clf,trainData, label):
#    clf.fit(trainData, label) 
#    y_pred = clf.predict(trainData)
#    print 'the confusion matrix is: '
#    print confusion_matrix(list(label.squeeze().T), list(y_pred.T))
#    return clf
#
#clf_svm = svm.SVC(probability=True)
#
##%% for svm
## training data cross validatioin, comparing the parameters
## try differnt kernel function (RBF, LINEAR, ETC)
## FROM 10*^(-4) TO 10*^(2)...LAMDA VALUE.  
## MODEL VERIFICATION, VALIDATON, 
#clf_svm = classProb(clf_svm,trainData, label)
#clf_svm.predict(dataTest)
#a = clf_svm.predict_proba(dataTest)
#
##%% for the gnb
#clf_gnb = GaussianNB()
#clf_gnb = classProb(clf_gnb,trainData, label)
#b = clf_gnb.predict(dataTest)
#gnb_prob = clf_gnb.predict_proba(dataTest)
#
#bdt = classProb(bdt,trainData, label)
#c = bdt.predict_proba(dataTest)
#
#plt.figure()
#plt.plot(clf_gnb.predict_proba(testSample)[:,1],label = 'GNB')
#
#
#plt.hold(True)
#
#plt.plot(c[:,1],label = 'bdt',color= 'r')
#plt.hold(True)
#plt.plot(0.5*np.ones([len(y_pred)]),label = '0.5',color = 'g')
#
#plt.legend()



#%% data 
#label = pd.DataFrame(columns=['label'])
#m = len(train_1)
#n = len(train_0)
#
#for i in range(m+n):
#    if i<m:
#        label.loc[i] = 1
#    else:
#        label.loc[i] = 0
#trainData = pd.concat([trueSample,falseSample])
#

#%% data to the model




#def classProb(clf):
#    clf.fit(trainData, label)    
#    return clf
#11
#clf_svm = svm.SVC(probability=True)
#
##%% for svm
#clf_svm = classProb(clf_svm)
#clf_svm.predict(testSample)
#a = clf_svm.predict_proba(testSample)
#
##%% for the gnb
#clf_gnb = GaussianNB()
#clf_gnb = classProb(clf_gnb)
#b = clf_gnb.predict(testSample)
#print 'the time to caculcate the feature is %d s'%	(time.time() - startTime)














