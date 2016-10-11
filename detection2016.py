# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from operator import itemgetter
import random
import os
import time
import glob
import re
from multiprocessing import Process
import copy
import timeit
from scipy.fftpack import fft
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)


import pylab as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import scipy.signal

standard_scaler = StandardScaler()

from scipy import signal

start = timeit.default_timer()


random.seed(2016)
np.random.seed(2016)


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def filter(X,sampleingRate, locCut, highCut):
    nyf = 0.5*sampleingRate
    b, a = signal.butter(5, np.array([locCut, highCut]) / nyf, btype='band')
    x_filt = signal.lfilter(b, a, X)
    return np.float32(x_filt)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]), sequence


def create_simple_csv_train(patient_id):
    # p[1] = Process(target=create_simple_csv_train, args=(1,))
    out = open("simple_train_" + str(patient_id) + ".csv", "w")
    out.write("Id,sequence_id,patient_id")
    for i in range(7*16):
        if i < 16:
            out.write(",avg_" + str(i))
        else:
            out.write(",power_feature" + str(i))


    out.write(",file_size,result\n")

    # TRAIN (0)
    pathData = "/Users/weizhili/Desktop/kaggle_egg/"
    out_str = ''
    files = sorted(glob.glob(pathData + "train_" + str(patient_id) + "/*0.mat"), key=natural_key)
    sequence_id = 0
    total = 0
    fftN = 4096
    fs = 400

    for fl in files:
        total += 1
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(sequence_id) + "," + str(patient)
        for f in sorted(list(tables.columns.values)):
            tables[f] = filter(tables[f].values, 400, 1, 50)
            mean = tables[f].mean()
            yChannel = abs(fft(tables[f], fftN))
            maxPowerFrequency = yChannel.argmax(axis=0)
            pow1_4Hz = np.sum(np.power(yChannel[int(0.5/fs * fftN):int(4./fs * fftN)], 2))
            pow4_8Hz = np.sum(np.power(yChannel[int(4./fs * fftN):int(8./fs * fftN)], 2))
            pow8_13Hz = np.sum(np.power(yChannel[int(8./fs * fftN):int(13./fs * fftN)], 2))
            pow13_30Hz = np.sum(np.power(yChannel[int(13./fs * fftN):int(30./fs * fftN)], 2))
            pow30_40Hz = np.sum(np.power(yChannel[int(30./fs * fftN):int(40./fs * fftN)], 2))
            featureExtract = "," + str(mean) + ',' + str(maxPowerFrequency) + ',' + str(pow1_4Hz) + \
                             "," + str(pow4_8Hz) + ',' + str(pow8_13Hz) + ',' + str(pow13_30Hz) + ',' + str(pow30_40Hz)

            out_str += featureExtract
        out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        if total % 6 == 0:
            if int(sequence_from_mat) != 6:
                print('Check error! {}'.format(sequence_from_mat))
                exit()
            sequence_id += 1

    out.write(out_str)

    # TRAIN (1)
    out_str = ''
    pathData = "/Users/weizhili/Desktop/kaggle_egg/"
    files = sorted(glob.glob(pathData + "train_" + str(patient_id) + "/*1.mat"), key=natural_key)
    sequence_id += 1
    total = 0
    for fl in files:
        total += 1
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(sequence_id) + "," + str(patient)
        for f in sorted(list(tables.columns.values)):
            tables[f] = filter(tables[f].values, 400, 1, 70)
            mean = tables[f].mean()
            yChannel = abs(fft(tables[f], fftN))
            maxPowerFrequency = yChannel.argmax(axis=0)
            pow1_4Hz = np.sum(np.power(yChannel[int(0.5/fs * fftN):int(4./fs * fftN)], 2))
            pow4_8Hz = np.sum(np.power(yChannel[int(4./fs * fftN):int(8./fs * fftN)],2))
            pow8_13Hz = np.sum(np.power(yChannel[int(8./fs * fftN):int(13./fs * fftN)],2))
            pow13_30Hz = np.sum(np.power(yChannel[int(13./fs * fftN):int(30./fs * fftN)],2))
            pow30_40Hz = np.sum(np.power(yChannel[int(30./fs * fftN):int(40./fs * fftN)],2))
            featureExtract = "," + str(mean) + ',' + str(maxPowerFrequency) + ',' + str(pow1_4Hz) + \
                             "," + str(pow4_8Hz) + ',' + str(pow8_13Hz) + ',' + str(pow13_30Hz) + ',' + str(pow30_40Hz)

            out_str += featureExtract
        out_str += "," + str(os.path.getsize(fl)) + "," + str(result) + "\n"
        if total % 6 == 0:
            if int(sequence_from_mat) != 6:
                print('Check error! {}'.format(sequence_from_mat))
                exit()
            sequence_id += 1

    out.write(out_str)
    out.close()
    print('Train CSV for patient {} has been completed...'.format(patient_id))


def create_simple_csv_test(patient_id):

    # TEST
    out_str = ''
    pathData = "/Users/weizhili/Desktop/kaggle_egg/"
    files = sorted(glob.glob(pathData + "test_" + str(patient_id) + "/*.mat"), key=natural_key)
    out = open("simple_test_" + str(patient_id) + ".csv", "w")
    out.write("Id,patient_id")
    for i in range(7*16):
        if i < 16:
            out.write(",avg_" + str(i))
        else:
            out.write(",power_feature" + str(i))

    out.write(",file_size\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        new_id = patient*100000 + id
        fftN = 4096
        fs = 400
        try:
            tables, sequence_from_mat = mat_to_pandas(fl)
        except:
            print('Some error here {}...'.format(fl))
            continue
        out_str += str(new_id) + "," + str(patient)
        for f in sorted(list(tables.columns.values)):
            tables[f] = filter(tables[f].values, 400, 1, 70)
            mean = tables[f].mean()
            yChannel = abs(fft(tables[f], fftN))
            maxPowerFrequency = yChannel.argmax(axis=0)
            pow1_4Hz = np.sum(np.power(yChannel[int(0.5/fs * fftN):int(4./fs * fftN)], 2))
            pow4_8Hz = np.sum(np.power(yChannel[int(4./fs * fftN):int(8./fs * fftN)],2))
            pow8_13Hz = np.sum(np.power(yChannel[int(8./fs * fftN):int(13./fs * fftN)],2))
            pow13_30Hz = np.sum(np.power(yChannel[int(13./fs * fftN):int(30./fs * fftN)],2))
            pow30_40Hz = np.sum(np.power(yChannel[int(30./fs * fftN):int(40./fs * fftN)],2))
            featureExtract = "," + str(mean) + ',' + str(maxPowerFrequency) + ',' + str(pow1_4Hz) + \
                             "," + str(pow4_8Hz) + ',' + str(pow8_13Hz) + ',' + str(pow13_30Hz) + ',' + str(pow30_40Hz)

            out_str += featureExtract

        out_str += "," + str(os.path.getsize(fl)) + "\n"
        # break

    out.write(out_str)
    out.close()
    print('Test CSV for patient {} has been completed...'.format(patient_id))


def run_single(train, test, features, target, random_state=1):
    eta = 0.1
    max_depth = 3
    subsample = 0.92
    colsample_bytree = 0.9
    start_time = time.time()

    num_boost_round = 1000
    early_stopping_rounds = 100
    test_size = 0.2

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=int(round(1/test_size, 0)), shuffle=True, random_state=random_state)
    train_seq_index, test_seq_index = list(kf)[0]
    print('Length of sequence train: {}'.format(len(train_seq_index)))
    print('Length of sequence valid: {}'.format(len(test_seq_index)))
    train_seq = unique_sequences[train_seq_index]
    valid_seq = unique_sequences[test_seq_index]

    X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    X_test = test[features]


    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

    countLabel = {'0':0, '1':0}
    for item in y_train:
        if item==1:
            countLabel['1'] +=1
        else:
            countLabel['0'] +=1

    ratioOne = float(countLabel['0']/countLabel['1'])
    ratioTwo = float(countLabel['1']/countLabel['0'])
    ratio = max(ratioOne, ratioTwo)
    params = {
            "objective": "binary:logistic",
            "booster" : "gbtree",
            "eval_metric": "auc",
            "eta": eta,
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": random_state,
            "scale_pos_weight": ratio
        }


    print('Length train:', len(X_train))
    print('Length valid:', len(X_valid))

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
    yfull_test = copy.deepcopy(test[['Id']].astype(object))

    yfull_test["1"] = test_prediction

    # get the losigitc regression
    # add logistic regression and random forest

    rm_lm = LogisticRegression(C=10, penalty='l1', tol=0.01)
    rm_lm.fit(X_train,y_train)
    check_lm = rm_lm.predict_proba(X_valid)
    scoreTwo = roc_auc_score(y_valid, check_lm[:,1])
    print('Check losigitc regression error value: {:.6f}'.format(scoreTwo))

    test_prediction_lm = rm_lm.predict_proba(X_test)
    yfull_test['2'] = test_prediction_lm[:, 1]


    # get the random forest
    n_estimator = 300
    rf = RandomForestClassifier(max_depth=10, n_estimators = n_estimator, class_weight= 'balanced')
    rf.fit(X_train,y_train)
    check_rf = rf.predict_proba(X_valid)
    scoreThree = roc_auc_score(y_valid, check_rf[:,1])

    test_prediction_rf = rf.predict_proba(X_test)
    print('Check random forest error value: {:.6f}'.format(scoreThree))
    yfull_test['3'] = test_prediction_rf[:,1]

    # get the gradientboost tree
    gbt = GradientBoostingClassifier(n_estimators=n_estimator, subsample= 0.9, max_features = "auto")
    gbt.fit(X_train, y_train)
    check_gbt = gbt.predict_proba(X_valid)
    scoreFour = roc_auc_score(y_valid, check_gbt[:,1])
    test_prediction_gbt = gbt.predict_proba(X_test)
    print('Check gradient boost error value: {:.6f}'.format(scoreFour))
    yfull_test['4'] = test_prediction_gbt[:,1]

    # get the gradientboost tree
    forest = ExtraTreesClassifier(n_estimators=n_estimator, bootstrap= True, max_features = "auto", class_weight= "balanced")
    forest.fit(X_train, y_train)
    check_forest = forest.predict_proba(X_valid)
    scoreFive = roc_auc_score(y_valid, check_forest[:,1])
    test_prediction_check_forest = forest.predict_proba(X_test)
    print('Check extral tree value: {:.6f}'.format(scoreFive))
    yfull_test['5'] = test_prediction_check_forest[:,1]
    merge = []
    for i in range(1, 6):
        merge.append(str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)
    test_prediction = yfull_test['mean'].values

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score

def run_singleDeepLearning(train, test, features, target, random_state=49):
    eta = 0.1
    max_depth = 3
    subsample = 0.9
    colsample_bytree = 0.9
    start_time = time.time()

    num_boost_round = 1000
    early_stopping_rounds = 30
    test_size = 0.2

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=int(round(1/test_size, 0)), shuffle=True, random_state=random_state)
    train_seq_index, test_seq_index = list(kf)[0]
    print('Length of sequence train: {}'.format(len(train_seq_index)))
    print('Length of sequence valid: {}'.format(len(test_seq_index)))
    train_seq = unique_sequences[train_seq_index]
    valid_seq = unique_sequences[test_seq_index]

    X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    X_test = test[features]

    inputDim = X_test.shape[1]

    input_img = Input(shape=(inputDim,))
    encoded1= Dense(100, activation='relu', activity_regularizer= regularizers.activity_l1(10e-5))(input_img)
    encoded2 = Dense(90, activation='relu')(encoded1)
    encoded3 = Dense(70, activation='relu')(encoded2)
    encoded4 = Dense(40, activation='relu')(encoded3)

    decoded0= Dense(70, activation='relu')(encoded4)

    decoded1= Dense(90, activation='relu')(decoded0)
    decoded2 = Dense(100, activation='relu')(decoded1)
    decoded3 = Dense(inputDim, activation='relu')(decoded2)
    autoencoder = Model(input=input_img, output=decoded3)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    X_train = standard_scaler.transform(X_train)
    X_test = standard_scaler.transform(X_test)
    X_valid = standard_scaler.transform(X_valid)

    autoencoder.fit(X_train, X_train,
                    nb_epoch=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_valid, X_valid))
    decoded_test = autoencoder.predict(X_valid)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_valid[i,:100].reshape(10, 10))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_test[i,:100].reshape(10, 10))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


    path = '/Users/weizhili/Desktop/data analysis part_suning/giftCard/gift/'

    plt.savefig(path + 'autoEncoding_1_deep_auto_encoding.png', bbox_inches='tight')
    X_train = autoencoder.predict(X_train)
    X_test = autoencoder.predict(X_test)
    X_valid = decoded_test

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

    countLabel = {'0':0, '1':0}
    for item in y_train:
        if item==1:
            countLabel['1'] +=1
        else:
            countLabel['0'] +=1

    ratioOne = float(countLabel['0']/countLabel['1'])
    ratioTwo = float(countLabel['1']/countLabel['0'])
    ratio = max(ratioOne, ratioTwo)
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
        "scale_pos_weight": ratio
    }

    print('Length train:', len(X_train))
    print('Length valid:', len(X_valid))

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    # get the losigitc regression

    rm_lm = LogisticRegression()
    rm_lm.fit(X_train,y_train)
    check_lm = rm_lm.predict_proba(X_valid)
    scoreTwo = roc_auc_score(y_valid, check_lm[:,1])
    print('Check losigitc regression error value: {:.6f}'.format(scoreTwo))

    test_prediction_lm = rm_lm.predict_proba(X_test)


    # get the random forest
    n_estimator = 300
    rf = RandomForestClassifier(max_depth=4, n_estimators = n_estimator, class_weight= 'balanced')
    rf.fit(X_train,y_train)
    check_rf = rf.predict_proba(X_valid)
    scoreThree = roc_auc_score(y_valid, check_rf[:,1])

    test_prediction_rf = rf.predict_proba(X_test)
    print('Check random forest error value: {:.6f}'.format(scoreThree))

    # get the gradientboost tree
    gbt = GradientBoostingClassifier(n_estimators=n_estimator, subsample= 0.9, max_features = "auto")
    gbt.fit(X_train, y_train)
    check_gbt = gbt.predict_proba(X_valid)
    scoreFour = roc_auc_score(y_valid, check_gbt[:,1])
    test_prediction_gbt = gbt.predict_proba(X_test)
    print('Check gradient boost error value: {:.6f}'.format(scoreFour))

    # get the gradientboost tree
    forest = ExtraTreesClassifier(n_estimators=n_estimator, bootstrap= True, max_features = "auto", class_weight= "balanced")
    forest.fit(X_train, y_train)
    check_forest = forest.predict_proba(X_valid)
    scoreFive = roc_auc_score(y_valid, check_forest[:,1])
    test_prediction_check_forest = forest.predict_proba(X_test)
    print('Check extral tree value: {:.6f}'.format(scoreFive))

    # do the merge of the predict probablity
    SumScore = score + scoreTwo + scoreThree + scoreFour + scoreFive
    test_prediction = test_prediction*(score/SumScore) + (scoreTwo / SumScore)*test_prediction_lm[:,1]\
                      + (scoreThree / SumScore)*test_prediction_rf[:,1] + \
    (scoreFour / SumScore)*test_prediction_gbt[:,1] + (scoreFive / SumScore)*test_prediction_check_forest[:,1]

    return test_prediction.tolist(), SumScore


def run_kfold(nfolds, train, test, features, target, random_state=2016):
    eta = 0.1
    max_depth = 3
    subsample = 0.9
    colsample_bytree = 0.90
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

    num_boost_round = 4000
    early_stopping_rounds = 100

    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['Id']].astype(object))

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_seq_index, test_seq_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        train_seq = unique_sequences[train_seq_index]
        valid_seq = unique_sequences[test_seq_index]
        print('Length of train people: {}'.format(len(train_seq)))
        print('Length of valid people: {}'.format(len(valid_seq)))

        X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
        y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
        X_test = test[features]

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        countLabel = {'0':0, '1':0}
        for item in y_train:
            if item==1:
                countLabel['1'] +=1
            else:
                countLabel['0'] +=1

        ratioOne = float(countLabel['0']/countLabel['1'])
        ratioTwo = float(countLabel['1']/countLabel['0'])
        ratio = max(ratioOne, ratioTwo)
        print(ratio)


        params = {
            "objective": "binary:logistic",
            "booster" : "gbtree",
            "eval_metric": "auc",
            "eta": eta,
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "seed": random_state,
            "scale_pos_weight": ratio
        }

   #     gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=1000)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        yhat = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        # Each time store portion of precicted data in train predicted values
        for i in range(len(X_valid.index)):
            yfull_train[X_valid.index[i]] = yhat[i]

        print("Validating...")
        check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
        score = roc_auc_score(y_valid.tolist(), check)
        print('Check error value: {:.6f}'.format(score))

        imp = get_importance(gbm, features)
        print('Importance array: ', imp)

        print("Predict test set...")
        test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
        yfull_test['kfold_' + str(num_fold)] = test_prediction

    # kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    # num_fold = nfolds
    # # get the losigitc regression
    # for train_seq_index, test_seq_index in kf:
    #     num_fold += 1
    #     print('Start fold {} from {}'.format(num_fold, nfolds))
    #     train_seq = unique_sequences[train_seq_index]
    #     valid_seq = unique_sequences[test_seq_index]
    #     print('Length of train people: {}'.format(len(train_seq)))
    #     print('Length of valid people: {}'.format(len(valid_seq)))
    #
    #     X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    #     y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    #     X_test = test[features]
    #     print('Length train:', len(X_train))
    #     print('Length valid:', len(X_valid))
    #     rm_lm = LogisticRegression(C=10, penalty='l1', tol=0.01)
    #     rm_lm.fit(X_train, y_train)
    #     check_lm = rm_lm.predict_proba(X_valid)
    #     scoreTwo = roc_auc_score(y_valid, check_lm[:,1])
    #     print('Check losigitc regression error value: {:.6f}'.format(scoreTwo))
    #     test_prediction_lm = rm_lm.predict_proba(X_test)
    #     yfull_test['kfold_' + str(num_fold)] = test_prediction_lm[:,1]
    #
    # # get the random forest
    # kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    # num_fold = 2*nfolds
    # # get the losigitc regression
    # for train_seq_index, test_seq_index in kf:
    #     num_fold += 1
    #     print('Start fold {} from {}'.format(num_fold, nfolds))
    #     train_seq = unique_sequences[train_seq_index]
    #     valid_seq = unique_sequences[test_seq_index]
    #     print('Length of train people: {}'.format(len(train_seq)))
    #     print('Length of valid people: {}'.format(len(valid_seq)))
    #
    #     X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    #     y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    #     X_test = test[features]
    #     print('Length train:', len(X_train))
    #     print('Length valid:', len(X_valid))
    #     n_estimator = 300
    #     rf = RandomForestClassifier(max_depth=4, n_estimators = n_estimator, class_weight= 'balanced')
    #     rf.fit(X_train,y_train)
    #
    #     check_rf = rf.predict_proba(X_valid)
    #     scoreThree = roc_auc_score(y_valid, check_rf[:,1])
    #     print('Check random forest error value: {:.6f}'.format(scoreThree))
    #     test_prediction_rf = rf.predict_proba(X_test)
    #     yfull_test['kfold_' + str(num_fold)] = test_prediction_rf[:,1]
    #
    #     kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    # num_fold = 3*nfolds
    # # get the gradientboost tree
    # for train_seq_index, test_seq_index in kf:
    #     num_fold += 1
    #     print('Start fold {} from {}'.format(num_fold, nfolds))
    #     train_seq = unique_sequences[train_seq_index]
    #     valid_seq = unique_sequences[test_seq_index]
    #     print('Length of train people: {}'.format(len(train_seq)))
    #     print('Length of valid people: {}'.format(len(valid_seq)))
    #
    #     X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    #     y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    #     X_test = test[features]
    #     print('Length train:', len(X_train))
    #     print('Length valid:', len(X_valid))
    #     n_estimator = 300
    #     gbt = GradientBoostingClassifier(n_estimators=n_estimator, subsample= 0.9, max_features = "auto")
    #     gbt.fit(X_train, y_train)
    #     check_gbt = gbt.predict_proba(X_valid)
    #     scoreFour = roc_auc_score(y_valid, check_gbt[:,1])
    #     test_prediction_gbt = gbt.predict_proba(X_test)
    #     print('Check gradient boost error value: {:.6f}'.format(scoreFour))
    #     yfull_test['kfold_' + str(num_fold)] = test_prediction_gbt[:,1]
    # num_fold = 4*nfolds
    # # get the gradientboost tree
    # for train_seq_index, test_seq_index in kf:
    #     num_fold += 1
    #     print('Start fold {} from {}'.format(num_fold, nfolds))
    #     train_seq = unique_sequences[train_seq_index]
    #     valid_seq = unique_sequences[test_seq_index]
    #     print('Length of train people: {}'.format(len(train_seq)))
    #     print('Length of valid people: {}'.format(len(valid_seq)))
    #
    #     X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], train[train['sequence_id'].isin(valid_seq)][features]
    #     y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], train[train['sequence_id'].isin(valid_seq)][target]
    #     X_test = test[features]
    #     print('Length train:', len(X_train))
    #     print('Length valid:', len(X_valid))
    #     n_estimator = 300
    #     forest = ExtraTreesClassifier(n_estimators=n_estimator, bootstrap= True, max_features = "auto", class_weight= "balanced")
    #     forest.fit(X_train, y_train)
    #     check_forest = forest.predict_proba(X_valid)
    #     scoreFive = roc_auc_score(y_valid, check_forest[:,1])
    #     test_prediction_check_forest = forest.predict_proba(X_test)
    #     print('Check extral tree value: {:.6f}'.format(scoreFive))
    #     yfull_test['kfold_' + str(num_fold)] = test_prediction_check_forest[:,1]
    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds + 1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return yfull_test['mean'].values, score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    pathFeature = "/Users/weizhili/Downloads/python_10_06/kaggle_EGG_Frequency/"
    sub_file = pathFeature + 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')
    total = 0
    for id in test['Id']:
        patient = id // 100000
        fid = id % 100000
        str1 = str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('Id')
    # output.remove('file_size')
    return sorted(output)


def read_test_train():
    print("Load train.csv...")
    pathFeature = "/Users/weizhili/Downloads/python_10_06/kaggle_EGG_Frequency/"
    train1 = pd.read_csv(pathFeature + "simple_train_1.csv")
    train2 = pd.read_csv(pathFeature + "simple_train_2.csv")
    train3 = pd.read_csv(pathFeature + "simple_train_3.csv")
    train = pd.concat([train1, train2, train3])
    # Remove all zeroes files
    train = train[train['file_size'] > 55000].copy()
    # Shuffle rows since they are ordered
    train = train.iloc[np.random.permutation(len(train))]
    # Reset broken index
    train = train.reset_index()
    print("Load test.csv...")
    test1 = pd.read_csv(pathFeature + "simple_test_1.csv")
    test2 = pd.read_csv(pathFeature + "simple_test_2.csv")
    test3 = pd.read_csv(pathFeature + "simple_test_3.csv")
    test = pd.concat([test1, test2, test3])
    print("Process tables...")
    features = get_features(train, test)
    return train, test, features


if __name__ == '__main__':
    print('XGBoost: {}'.format(xgb.__version__))
    if 1:
        # Do reading and processing of MAT files in parallel
        p = dict()
        p[1] = Process(target=create_simple_csv_train, args=(1,))
        p[1].start()
        p[2] = Process(target=create_simple_csv_train, args=(2,))
        p[2].start()
        p[3] = Process(target=create_simple_csv_train, args=(3,))
        p[3].start()
        p[4] = Process(target=create_simple_csv_test, args=(1,))
        p[4].start()
        p[5] = Process(target=create_simple_csv_test, args=(2,))
        p[5].start()
        p[6] = Process(target=create_simple_csv_test, args=(3,))
        p[6].start()
        p[1].join()
        p[2].join()
        p[3].join()
        p[4].join()
        p[5].join()
        p[6].join()
    train, test, features = read_test_train()
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    test_predictionXgb, scoreXgb = run_single(train, test, features, 'result')
    test_predictionDP, scoreDP = run_singleDeepLearning(train, test, features, 'result')
    test_predictionXgb2, scoreXgb2 = run_kfold(5, train, test, features, 'result')

    plt.figure()
    plt.hist(test_predictionXgb,100)
    plt.title("test_predictionXgb _1")
    plt.figure()
    plt.hist(test_predictionDP,100)
    plt.title("test_predictionDP _1")

    plt.figure()
    plt.hist(test_predictionXgb2,100)
    plt.title("test_predictionXgb2 _2")

    # test_prediction = 0.15*np.asarray(test_predictionXgb) + 0.15*np.asarray(test_predictionDP) + 0.7*test_predictionXgb2
    test_prediction = 0.2*np.asarray(test_predictionXgb) + 0.8*test_predictionXgb2

    plt.figure()
    plt.hist(test_prediction,100)
  #  score = (scoreXgb + scoreDP/5 + scoreXgb2)/3
    score = (scoreXgb  + scoreXgb2)/2

    create_submission(scoreXgb2, test, test_predictionXgb2)


stop = timeit.default_timer()

print stop - start
