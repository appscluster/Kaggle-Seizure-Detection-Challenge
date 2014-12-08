# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 13:41:39 2014

@author: Valued Customer
"""

import pandas as pd
import pylab as plt
import scipy
from scipy.fftpack import fft
import numpy as np
import time
#data = pd.read_csv('C:/Users/Valued Customer/Desktop/data for fun/EEG/train (1)/Data_S02_Sess01.csv')
time_start = time.clock()
#aa = getData(path)
def getData(path):
    featureNumber = 9
    channels = 57
    fftN = 1024
    fs = 200
    j = 0
    # how many features we will extract
    
    # path = 'C:/Users/Valued Customer/Desktop/data for fun/EEG/train (1)/Data_S02_Sess01.csv'
    data = pd.read_csv(path)
    feedBackEvent = data['FeedBackEvent']
    index = np.where(feedBackEvent==1)
    featureStore = pd.DataFrame(columns= ['channel' + str(i+1) for i in range(channels*featureNumber)])
    for dataIndex in index[0]:
        if dataIndex-10*200<=0:
            begin = dataIndex
        else:
            begin = dataIndex -10*200
        if dataIndex +200*10 > data.shape[0]:
            end = dataIndex
        else:
            end = dataIndex + 200*10
#        print 'the begin %d and end is %d' % (begin,end)
 
        dataStore = data[data.keys()[1:-1]].iloc[begin:end,:].T
#        featureStore =None
        
        
        yChannel = abs(fft(dataStore,fftN)) #%% fft, from time domain to frequency domain
                                           
        f = np.linspace(0,fs,num = fftN)
        maxSave = np.zeros((1,channels*featureNumber))
                    # 0 to 7
        maxSave[0,0:channels] = 2*f[yChannel[:,int(0.5/fs * fftN):int(40./fs * fftN)].argmax(axis=1)\
                                                    +int(0.5/fs * fftN)]
                    ## find the maximum location
        maxSave[0,channels:channels*2] = np.amax(yChannel[:,int(0.5/fs * fftN):int(40./fs * fftN)],axis=1)  ## find the maximum value
    
        maxSave[0,channels*2:channels*3] = np.sum(np.power(yChannel[:,int(0.5/fs * fftN):int(4./fs * fftN)],2),axis=1)
                    
                    # 4 Hz to 8 Hz  4/fs * 1024 = 10.25 to 8/fs* 1024 = 20.5
        maxSave[0,channels*3:channels*4] = np.sum(np.power(yChannel[:,int(4./fs * fftN):int(8./fs * fftN)],2),axis=1)
                    # 8 Hz to 13 Hz  8/fs * 1024 = 20.5 to 13/fs* 1024 = 33.125
        maxSave[0,channels*4:channels*5] = np.sum(np.power(yChannel[:,int(8./fs * fftN):int(13./fs * fftN)],2),axis=1)
                    # 13 Hz to 30 Hz  13/fs * 1024 = 33.125 to 30/fs* 1024 = 76.875
        maxSave[0,channels*5:channels*6] = np.sum(np.power(yChannel[:,int(13./fs * fftN):int(30./fs * fftN)],2),axis=1)
                    # 30 Hz to 100 Hz 30/fs * 1024 = 76.875 to 100/fs* 1024 = 256.25
        maxSave[0,channels*6:channels*7] = np.sum(np.power(yChannel[:,int(30./fs * fftN):int(40./fs * fftN)],2),axis=1)
                    
                    #%% matrix to each power band:
        powerBand = maxSave[0,channels*2:channels*7].reshape([5,channels])
                    # caculate total power
        p = map(lambda x: x*x,np.sum(yChannel,axis=1))
    #                    maxValue = np.amax(powerBand,axis=0)
                    #
        maxSave[0,channels*2:channels*7] = np.divide(powerBand,p).reshape(1,5*channels)
               #     np.divide(maxSave[0,32:].reshape([5,16]),np.amax(maxSave[0,32:].reshape([5,16]),axis=0)).reshape(1,80)
    #%%                    maxSave[0,32:] = aaa.reshape(1,80)  
                    #%% 0-40 Hz, 
        p40Hz = np.sum(np.power(yChannel[:,int(0.5/fs * fftN):int(40./fs * fftN)],2),axis=1)*0.5
        maxSave[0,channels*8:channels*9] = np.array(p40Hz)
    #                p40Hz = (np.sum(yChannel[:,int(0.5/fs * 4095):int(40./fs * 4095)],axis=1)**(2))*0.5               
        spectralEdge = []
        for jj in range(channels):
            for ii in range(int(0.5/fs * fftN),fftN/2):
                N = float(ii)/fs*fftN
                power = np.sum(np.power(yChannel[jj,int(0.5/fs * fftN):N],2))
                if power>=p40Hz[jj]:
                    spectralEdge.append(N*fs/fftN)
                    break
        maxSave[0,channels*7:channels*8] = np.array(spectralEdge)  
    
        featureStore.loc[j] = maxSave
        j +=1
#        
        
    return featureStore

import os
#path = 'C:/Users/Valued Customer/Desktop/data for fun/EEG/train (1)/Data_S02_Sess01.csv'
path = 'C:/Users/Valued Customer/Desktop/data for fun/EEG/train (1)'
dogFolder = os.listdir(path)    
featureMatrix = getData(path + '/' +  dogFolder[0])
for i in range(1,len(dogFolder)):
    print 'the index %d data begins ' %i
    featureMatrix = pd.concat([featureMatrix,getData(path + '/' +  dogFolder[i])],ignore_index=True)
print  'the time to get the output is %0.2f s' %  (time.clock() - time_start)    
#    featureMatrix.append(getData(path + '/' +  dogFolder[2]),ignore_index = True)
featureMatrix.to_csv('C:/Users/Valued Customer/Desktop/data for fun/EEG/TrainfeatureMatrix.csv', index = False)    
        
#%%
path = 'C:/Users/Valued Customer/Desktop/data for fun/EEG/test (1)'
dogFolder = os.listdir(path)    
testMatrix = getData(path + '/' +  dogFolder[0])
for i in range(1,len(dogFolder)):
    print 'the index %d data begins ' % i
    testMatrix = pd.concat([testMatrix,getData(path + '/' +  dogFolder[i])],ignore_index=True)
print  'the time to get the output is %0.2f s' %  (time.clock() - time_start)     
testMatrix.to_csv('C:/Users/Valued Customer/Desktop/data for fun/EEG/TestfeatureMatrix.csv', index = False)  



labels = pd.read_csv('C:/Users/Valued Customer/Desktop/data for fun/EEG/TrainLabels (1).csv')
submission = pd.read_csv('C:/Users/Valued Customer/Desktop/data for fun/EEG/SampleSubmission (5).csv')
#%%
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
L1 = LinearSVC(C=0.01, penalty="l1", dual=False)
X_new = L1.fit_transform(featureMatrix.values, y)
x_test = L1.transform(testMatrix.values)

clf = svm.SVC()
clf.fit(X_new,y)
a = clf.predict(x_test)
#%%
rf = RandomForestClassifier(n_estimators=150, max_features = 100,random_state=42)

y = labels['Prediction']
clf = svm.SVC()
rf.fit(featureMatrix, y)
a = rf.predict(testMatrix)
b = [str(b) for b in a]
print b.index('0')

submission['Prediction'] = a
submission.to_csv('C:/Users/Valued Customer/Desktop/data for fun/EEG/benchmark.csv', index = False)
#training_files = []
#for filename in labels.IdFeedBack.values:
#    training_files.append(filename[:-6])  
#
#testing_files = []
#for filename in submission.IdFeedBack.values:
#    testing_files.append(filename[:-6])  












