# -*- coding: utf-8 -*-
from __future__ import print_function

import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.svm import SVR, LinearSVR

from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import xam

from scipy.special import boxcox, inv_boxcox

import random
import copy

class Modeling( object):
    def __init__(self, data):
        # np.random.seed(1)

        self.data           = data
        self.trainX         = data.trainX
        self.testX          = data.testX
        #pre_assigned
        self.testY          = data.testY[:, 0]
        self.trainY         = data.trainY[:, 0]

        # a test of transform
        self.LAMBDA = .001

        # # to test on transform domain, activate this block
        # self.TRANSFORM      = False
        # self.trainY         = self.target_transform( self.trainY)
        # self.testY          = self.target_transform( self.testY)

        #for boxcox transform
        self.TRANSFORM      = False

        #for plottting
        self.N_JOBS = 4 #for scikit multi cores
        self.scoring = 'neg_mean_absolute_error'
        self.BOUNDARY = .03
        # self.OUTLIER_BOUNDARY = .05

        #some parameters for training
        self.NORMALIZE_GP   = False
        self.INCLUDE_OLD_FEATURES = False
        self.STACK_FOLD     = 10
        self.CV_FOLD        = 10
        self.LOG_FILE       = 'log1.log'
        self.SAVE_PATH      = os.path.join('../Res/MCS Features Tuned Redelivery update new measurement/')
        if not os.path.isdir( self.SAVE_PATH):
            os.makedirs( self.SAVE_PATH)

        self.dt_hyper   = {'max_features' : 1,
                           'max_depth' : 1
        }


        # # # MCS param -TR15 -15MV
        self.rf_hyper = {'n_estimators': 1,'max_depth' : 1,
                         'max_features' : 1,
                         'n_jobs':-1, 'warm_start': False, 'random_state': None}
        self.ada_hyper  = {'n_estimators' : 1, 'learning_rate' : 1}
        self.xgb_hyper = {'n_estimators': 1, 'max_depth': 1, 'learning_rate': 1,
                          }


        # self.xgb_hyper = {'n_estimators': 100, 'max_depth':8, 'learning_rate': 0.35, 'reg_alpha': .3, 'reg_lambda': 2.5, 'gamma' : 0}

        #kernel stacking
        self.NU_STACK   = .1 * 1e-1
        self.kernel_s   = 2. * Matern(length_scale=10., length_scale_bounds=(1e-5, 1e5),
                                        nu= self.NU_STACK)

        self.base_models = {
            # 'KNN'       : KNeighborsRegressor( **self.k_hyper),
            # 'L-SVM'     : LinearSVR( epsilon= 1e-2, C= 1000),
            # 'NL-SVM'    : SVR(**self.svm_hyper ),
            'RF'        : RandomForestRegressor( **self.rf_hyper),
            # 'Adaboost'  : AdaBoostRegressor( base_estimator=SVR( **self.svm_hyper), **self.ada_hyper), #repace weak SVM by a tree
            'Adaboost': AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**self.dt_hyper),
                                          **self.ada_hyper),
            'XGBoost'   : XGBRegressor(**self.xgb_hyper)
            # 'XGBoost'   : XGBRegressor( n_estimators= 10, max_depth = 2)
        }

 
        self.models     = self.base_models
        self.names      = self.models.keys()
        self.regressors = self.models.values()

        self.cross_validation()
        self.learning_curve()
        self.stacking_test()
        self.feature_importance()

        print("Done!")

    def log(self, a):
        with open( self.LOG_FILE, 'a') as file_handle:
            print( a, file = file_handle)
            print('\n', file=file_handle)

    def target_transform(self, Y):
        #an empirical parameter
        if self.TRANSFORM:
            Y_t    = boxcox( Y, self.LAMBDA)

            # Y_t     = np.log(Y)
        else: #no transform
            Y_t    = Y

        return Y_t

    def target_inverse_transform(self, Y_t):
        #an empirical parameter
        if self.TRANSFORM:
            #clip the boundary
            #0 -> -1/lambda
            #1 -> 0
            eps     = 1e-7
            index0 = Y_t < -1./self.LAMBDA
            Y_t[index0] = (-1. +eps)/self.LAMBDA

            index1 = Y_t > 0
            Y_t[index1] = 0
            Y      = inv_boxcox( Y_t, self.LAMBDA)

            # Y = np.exp( Y_t)
        else: #no transform
            Y    = Y_t

        return Y

# # # 1st hierachical functions # # #

    def cross_validation(self):
        trainX = self.trainX
        trainY = self.trainY

        # trainX = np.vstack( (trainX, self.testX))
        # trainY = np.hstack((trainY, self.testY))

        #self.knn_cv()
        # self.svm_cv( trainX, trainY)
        self.rf_cv( trainX, trainY)
        self.xgb_cv( trainX, trainY)
        self.ada_cv( trainX, trainY)
        # self.gp_cv()

    def learning_curve(self):
        # names = ['SVR', 'RF', 'AdaBoost', 'XGB', ]
        L_names = len( self.names)
        # f       = [0.05,  .1, .5, .7, 1]
        f       = np.array(range(1,21)) *.05
        N       = len( f)
        L       = len( self.trainY)

        TrainX  = np.copy( self.trainX)
        TrainY  = np.copy( self.trainY)

        alg_indices = [0,1,2]

        for m in range( N):
            print ("repeat m: \n", m)
            np.random.seed( m)
            res_train = np.zeros((L_names, N), dtype = float)
            res_test = np.zeros((L_names, N), dtype=float)

            for p_i in range( N):
                Li  = int( f[p_i] * L)
                print( "portion i: \n", p_i, Li)
                ind = range(L)
                np.random.shuffle( ind)
                trainX          = TrainX[ ind[:Li]]
                trainY          = TrainY[ ind[:Li]]

                for alg_i in alg_indices:
                    print ("Regressor: ", self.names[alg_i])
                    clf             = self.regressors[alg_i]
                    (clf, res_train[alg_i, p_i], res_test[alg_i, p_i], max_train, max_test) = 
                                    self.learning_curve_individual(clf, trainX, trainY)

                #save  train test
                np.savetxt( os.path.join( self.SAVE_PATH, str(m) + "_learning_curve_train.txt"), res_train)
                np.savetxt(os.path.join(self.SAVE_PATH, str(m) + "_learning_curve_test.txt"), res_test)

            #figure
            for r in range( L_names):
                plt.figure(m, figsize= (16., 6.))
                ax  = plt.subplot( 2, (L_names +1)/2, r + 1)
                ax.plot( 1. *L/N* np.array( range(1, N+1)), res_train[r])
                ax.plot( 1. *L/N* np.array( range(1, N+1)), res_test[r])
                ax.set_xlabel( 'Number of samples')
                ax.set_ylabel('MAE')
                ax.set_title( self.names[r])
            plt.savefig( os.path.join( self.SAVE_PATH, str(m) + "_learning_curve.png"))

    def feature_importance(self):
        trainX = self.trainX
        trainY = self.trainY

        # names = ['RF' , 'XGB']
        algs = [0,1,2]
        for i in algs: #range( len(self.names)):
            clf = self.regressors[i]
            trainY_t = self.target_transform(trainY)
            clf.fit( trainX, trainY_t)

            plt.figure(i + 10)
            ax1     = plt.subplot(1, 1, 1)
            # print clf.f
            ind     = np.argsort( clf.feature_importances_)
            bars    = clf.feature_importances_[ind]
            labs    = self.data.data_train.columns[2:-2]
            labs    = labs[ind]
            print ([str(lab) for lab in labs])
            ax1.barh( range( len( bars)), bars)
            print (clf.feature_importances_[ind])
            plt.yticks( range( len( bars)), labs)
            ax1.set_title( self.names[i])
            # plt.suptitle( ' Feature Importance')
            plt.savefig(self.SAVE_PATH + self.names[i] + '_ feature _ importance.png')
        # plt.pause(1000)
        return -1

    def stacking_test(self):
        trainY = self.trainY
        testY = self.testY
        trainX = self.trainX
        testX = self.testX

        print( len( self.regressors))
        regressors = list( self.regressors)
        for i in range( len( regressors)):
            clf = regressors[i]

            trainY_t = self.target_transform(trainY)
            clf.fit( trainX, trainY_t)

            pred_trainY_t   = clf.predict( trainX)
            pred_trainY     = self.target_inverse_transform( pred_trainY_t)
            err_train       = pred_trainY - trainY

            pred_testY_t    = clf.predict( testX)
            pred_testY      = self.target_inverse_transform( pred_testY_t)
            err_test        = pred_testY - testY

            # # # TRAINING # # #
            plt.figure(1, figsize= (16., 6.))
            ax1             = plt.subplot( 2, (len( self.regressors) +1)/2, i + 1)
            ax1.scatter(  trainY, pred_trainY)
            ax1.plot( trainY, trainY + self.BOUNDARY)
            ax1.plot( trainY, trainY - self.BOUNDARY)
            ax1.plot( trainY, trainY)
            ax1.set_title( self.names[i])

            plt.suptitle('Train prediction')
            plt.savefig( self.SAVE_PATH + 'Train_prediction.png')

            np.savetxt( self.SAVE_PATH + self.names[i] + '_train_prediction.csv', pred_trainY, delimiter=',')

            # HISTOGRAM FOR TRAINING
            plt.figure( 2, figsize= (16., 6.))
            ax2             = plt.subplot(2, (len( self.regressors) +1)/2, i+1)
            ax2.hist( err_train)
            ax2.set_title( self.names[i] + 'max_err =' + str( np.around( max (abs( err_train)), 2)))

            plt.suptitle('Train  error histogram')
            plt.savefig( self.SAVE_PATH + 'Train err histogram.png')

            # # # TESTING # # #
            plt.figure(3, figsize= (16., 6.0))
            ax3             = plt.subplot( 2, (len( self.regressors) +1)/2, i+1)
            ax3.scatter( testY, pred_testY)
            ax3.plot( testY, testY + self.BOUNDARY)
            ax3.plot( testY, testY - self.BOUNDARY)
            ax3.plot( testY, testY )
            ax3.set_title( self.names[i])

            plt.suptitle( 'Testing prediction')
            plt.savefig( self.SAVE_PATH + 'Test_prediction.png')

            np.savetxt(self.SAVE_PATH + self.names[i] + '_test_prediction.csv', pred_testY, delimiter=',')

            # HISTOGRAM FOR TRAINING
            plt.figure(4, figsize=(16., 6.))
            ax4 = plt.subplot(2, (len(self.regressors) + 1) / 2, i + 1)
            ax4.hist(err_test)
            ax4.set_title(self.names[i] + 'max_err =' + str(np.around(max(abs(err_test)), 2)))

            plt.suptitle('Test error histogram')
            plt.savefig(self.SAVE_PATH + 'Test err histogram.png')

            #large train error
            ind_sort = np.argsort( -abs(err_train))
            print("Large Train error:")
            print (self.names[i])
            try:
                print (self.data.data_train.iloc[ind_sort[0:10], 0:2])
                print (err_train[ind_sort[0:10]])
                print ("Mean: ", np.mean(abs(err_train)))
            except:
                print ("Out of bound may be because test in train")

            #print the largest error beam
            print("Large Test error:")
            ind_sort = np.argsort( -abs(err_test))
            print (self.names[i])
            print (self.data.data_test.iloc[ind_sort[0:10], 0:2])
            print (err_test[ind_sort[0:10]])
            print("Mean: ", np.mean(abs(err_test)))

# # # 2nd hierachical functions # # #

    def rf_cv(self, trainX, trainY):
        '''
        self.rf_hyper   = {'max_features' : 29,
                           'n_estimators' : 800, 'max_depth' : 20}
        :param trainX:
        :param trainY:
        :return:
        '''
        trainY_t          = self.target_transform( trainY)
        param_grids = {
            "n_estimators" : [100, 400, 800, 1000],
            "max_depth" : [3, 7, 9, 11, 13, 15, 17, 19, 21],
            'max_features' : [5, 7, 9, 11, 13, 15]
        }

        grid = GridSearchCV( RandomForestRegressor(
                             param_grid= param_grids,
                             n_jobs= self.N_JOBS, verbose=1, cv= self.CV_FOLD, scoring= self.scoring)

        grid.fit( trainX, trainY_t)

        #plot the cv labels wise, not heat map
        plot_cv( tuned_parameters= param_grid, clf = grid,
                 clf_name= 'RF' + str( param_grids["n_estimators"][1]),
                 obj = self)

        print ("RF: The best params are %s with score %f\n" % (grid.best_params_, grid.best_score_))

        clf = RandomForestRegressor( **grid.best_params_)
        clf.fit(  trainX, trainY_t)



        (mean_train, mean_test, max_train, max_test) = self.compute_error(clf, trainX, trainY)
        print ("mean_train err, mean_test err, followed by max: ", ( mean_train, mean_test, max_train, max_test))
        self.log( ( mean_train, mean_test, max_train, max_test))


        self.regressors[0] = clf

    def xgb_cv(self, trainX, trainY):

        trainY_t          = self.target_transform( trainY)
        param_grids = {
            "n_estimators" : [50, 100, 200, 400],
            "max_depth" : [1, 2, 4, 5, 6, 8, 10],
            'learning_rate': [.1, .2, .25, .3, .5]
        }

        grid = GridSearchCV( XGBRegressor( 
                             param_grid= param_grids,
                             n_jobs= self.N_JOBS, verbose=1, cv= self.CV_FOLD, scoring= self.scoring)

        grid.fit( trainX, trainY_t)


        #plot the cv labels wise, not heat map
        plot_cv( tuned_parameters= param_grid, clf = grid,
                 clf_name= 'XGB' + str( param_grids["n_estimators"][1]),
                 obj = self)

        print ("XGB: The best params are %s with score %f\n" % (grid.best_params_, grid.best_score_))

        clf = XGBRegressor( **grid.best_params_)
        clf.fit(  trainX, trainY_t)

        (mean_train, mean_test, max_train, max_test) = self.compute_error(clf, trainX, trainY)
        print ("mean_train err, mean_test err, followed by max: ", ( mean_train, mean_test, max_train, max_test))
        self.log( ( mean_train, mean_test, max_train, max_test))


        self.regressors[2] = clf

    def ada_cv(self, trainX, trainY):

        trainY_t          = self.target_transform( trainY)
        param_grids = {
            "n_estimators" : [10, 25, 50, 100, 200],
            'learning_rate': [.1, .2, .25, .3, .5]
        }

        grid = GridSearchCV( AdaBoostRegressor( base_estimator=DecisionTreeRegressor(**self.dt_hyper)),
                             param_grid= param_grids,
                             n_jobs= self.N_JOBS, verbose=1, cv= self.CV_FOLD, scoring= self.scoring)

        grid.fit( trainX, trainY_t)
 
        #plot the cv labels wise, not heat map
        plot_cv( tuned_parameters= param_grid, clf = grid,
                 clf_name= 'Ada' ,
                 obj = self)

        print ("ADA: The best params are %s with score %f\n" % (grid.best_params_, grid.best_score_))

        clf = AdaBoostRegressor( base_estimator=DecisionTreeRegressor(**self.dt_hyper), **grid.best_params_)
        clf.fit(  trainX, trainY_t)

        (mean_train, mean_test, max_train, max_test) = self.compute_error(clf, trainX, trainY)
        print ("mean_train err, mean_test err, followed by max: ", ( mean_train, mean_test, max_train, max_test))
        self.log( ( mean_train, mean_test, max_train, max_test))

        self.regressors[1] = clf

    def learning_curve_individual(self, clf, trainX, trainY):
        trainY_t = self.target_transform(trainY)
        clf.fit( trainX, trainY_t)

        (mean_train, mean_test, max_train, max_test) = self.compute_error(clf, trainX, trainY)
        print ("mean_train err, mean_test err, followed by max: ", ( mean_train, mean_test, max_train, max_test))
        self.log( ( mean_train, mean_test, max_train, max_test))

        return (clf, mean_train, mean_test, max_train, max_test)

# # # helper functions # # #

    def compute_error(self, clf, trainX, trainY):
        # trainX      = self.trainX
        testX       = self.testX
        # trainY_o    = self.trainY_o
        testY     = self.testY

        pred_trainY_t   = clf.predict( trainX)
        pred_testY_t    = clf.predict( testX)

        pred_trainY     = self.target_inverse_transform( pred_trainY_t)
        pred_testY      = self.target_inverse_transform( pred_testY_t)
         #, pred_testY) = self.target_inverse_transform( pred_trainY_t, pred_testY_t)

        mean_train  = metrics.mean_absolute_error(trainY, pred_trainY)
        mean_test   = metrics.mean_absolute_error(testY, pred_testY)
        max_train   = np.amax(abs(trainY - pred_trainY))
        max_test    = np.amax(abs(testY - pred_testY))

        return ( mean_train, mean_test, max_train, max_test)


class Data():
    def __init__(self, training_path, testing_path, need_preprocess):
        '''

        :param training_path:
        :param testing_path:
        :param pre_processed: True: need some process
        '''
        self.training_path = training_path
        self.testing_path = testing_path
        self.need_preprocess = need_preprocess

        self.data, self.data_train, self.data_test = [[]] * 3 #raw data, self.data doesn't have meaning

        #all those train test must be in ndarray
        self.ratio = .9
        self.trainX, self.trainY, self.testX, self.testY = [[]] * 4  # just to clarify what Data have
        self.trainXu, self.testXu = [[]] * 2 #unscale feature, to see how scaling work

        #set seed
        np.random.seed(1)
        self.indices = [] #keep  track of the randomness order

        if training_path == testing_path: #split data into 8/2 9/1
            self.split_data()
        else:
            if self.need_preprocess:
                self.load_raw_data()
            else:
                self.load_data()

        #rescale trainX
        self.sc = self.rescale_train()
        #rescale testX
        self.rescale_test()

    def load_data(self):
        #currently train and set are in single file
        self.data_train = pd.read_csv( self.training_path,  header= None)
        self.data_test = pd.read_csv( self.testing_path, header = None)

        #split process data, unscaled in numpy format
        self.trainXu             = np.array( self.data_train.iloc[ :,0: -2])
        self.trainY             = np.array( self.data_train.iloc[ :, [-2, -1]])
        self.testXu              = np.array( self.data_test.iloc[ :, 0: -2]) #3 becasue of index column saved
        self.testY              = np.array( self.data_test.iloc[ :, [-2, -1]])

        #print self.trainXu[100, :]
        #print self.data_train.iloc[1, :]
        #quit()

    def load_raw_data(self):
        '''
        :return:
        '''
        self.data_train_org = pd.read_csv( self.training_path)
        self.data_train = self.preprocess_dummy_more_features( self.data_train_org)

        self.data_test_org = pd.read_csv(self.testing_path)
        self.data_test = self.preprocess_dummy_more_features( self.data_test_org)

        self.trainXu    = np.array( self.data_train.iloc[:, 2: -2])
        self.trainY     = np.array( self.data_train.iloc[:, [-2, -1]])

        self.testXu     = np.array( self.data_test.iloc[:, 2: -2])
        self.testY      = np.array( self.data_test.iloc[:, [-2, -1]])

    def split_data(self):
        ''''''
        self.data_org = pd.read_csv( self.training_path)

        #randomize original data
        N = len(self.data_org)
        self.indices                 = np.random.permutation( N )
        self.data                    = self.data_org.iloc[self.indices,:] #randomize

        if self.need_preprocess:
            data_num = self.preprocess_dummy_more_features()

        else:
            data_num = self.data

        #split data to 8/2
        split_point             = long(  N * self.ratio )
        # self.data_train         = self.data.loc[:split_point] #slice works, but index is not
        # self.data_test          = self.data.loc[split_point:]
        self.data_train         = data_num.iloc[:split_point, :] #slice doesn't work, but index is not
        self.data_test          = data_num.iloc[split_point:,:]

        #reindex for later use  to trace back ID, error
        self.data_train.reset_index(drop=True, inplace = True)
        self.data_test.reset_index( drop=True, inplace= True)

        #split process data, unscaled in numpy format
        self.trainXu             = np.array( data_num.iloc[ :split_point,2: -2])
        self.trainY             = np.array( data_num.iloc[ :split_point, [-2, -1]])
        self.testXu              = np.array( data_num.iloc[ split_point:, 2: -2])
        self.testY              = np.array( data_num.iloc[ split_point:, [-2, -1]])

        # print self.trainXu[100, :]
        # print self.data_train.iloc[100, :]
        # quit()

    def preprocess_dummy_more_features(self, data):
        '''
        No dictionary needed, map directly to binary features
        Downside is need to index by counting the number
        :return:
        '''
        self.data = data

        self.data['TreatID'] = self.data['TreatID'].str.upper()
        self.data['Energy_Plan'] = self.data['Energy_Plan'].apply(int).apply(str) #tricky need in Linux but not in Windows pandas
        # self.data = pd.get_dummies( self.data, columns=['TreatID'], prefix=[''], prefix_sep='')
        self.data = pd.get_dummies(self.data, columns=['Energy_Plan'], prefix=['MV-'], prefix_sep='')

        # TrueBeam to replace machine name
        ind1 = self.data['TreatID'] == 'TR 4'
        ind2 = self.data['TreatID'] == 'TR 6'
        self.data = self.data.assign( TrueBeam = 1) #every this TrueBeam
        self.data.loc[ ind1 | ind2, 'TrueBeam'] = 0 #excep 4 and 6

        #Tr15
        ind15 = self.data['TreatID'] == 'TR 15'
        self.data = self.data.assign( aSi1200 = 0)
        self.data.loc[ ind15, 'aSi1200'] = 1 #TR15

        #0,1: ID and beam, #19,20: Gamma
        #number index is so easy to make mistake so converting to fieldname:
        feature_names = [
                         'PatientID', 'Beam_ID',
                            # 'MU',
                            'BA', 'BI', 'BM', 'UAA',
                          'MSAS2', 'MSAS5', 'MSAS10', 'MSAS20',
                          'MaxSAS2', 'MaxSAS5', 'MaxSAS10', 'MaxSAS20',
                          'MFA', 'MAD', 'MUCP',
                          'MLO', 'MLO2', 'MLO3', 'MLO4', 'MLO5',
                          'minAP_h', 'maxAP_h', 'minAP_v', 'maxAP_v',
                          'maxRegs',
                          'AAJA',
                            'MAXJ',
                            #'Energy_Plan',
                            #'6', '10', '15',
                            'MV-6', 'MV-10',
                          #'TR 14', 'TR 15', 'TR 4', 'TR 5', 'TR 6',
                            'TrueBeam',
                            'aSi1200',
                          #new features
                          'MCS', 'EM',
                          'Gamma_22', 'Gamma_33'
                         ]
        #data_num = self.data.iloc[:, [0 ,1] + range( 5, 18) + range(21,25) + [19, 20] ]
        data_num = self.data.loc[:, feature_names]
        data_num['Gamma_22'] = data_num['Gamma_22']
        data_num['Gamma_33'] = data_num['Gamma_33']

        return data_num

    def rescale_train(self):
        # sc = preprocessing.StandardScaler()
        # # sc = preprocessing.RobustScaler()
        # # sc  = preprocessing.MinMaxScaler()
        # sc.fit( self.trainXu)
        # self.trainX = sc.transform( self.trainXu)
        # return sc

        self.trainX = self.trainXu

    def rescale_test(self):
        # self.testX = self.sc.transform( self.testXu)

        self.testX = self.testXu

    def save_train_test(self, train_file = 'train_data.csv', test_file = 'test_data.csv'):
        '''
        save from data_train i.e no normalization
        :param train_file:
        :param test_file:
        :return:
        '''
        self.data_train.to_csv( train_file, sep =',' , index = False)
        self.data_test.to_csv( test_file, sep = ',', index = False)

    def save_train_test_num_format(self, train_file = 'train_data_num.csv', test_file = 'test_data_num.csv'):
        '''
        With normalization
        :param train_file:
        :param test_file:
        :return:
        '''
        np.savetxt( train_file, np.hstack( (self.trainX, self.trainY) ), delimiter="," )
        np.savetxt(test_file, np.hstack( (self.testX, self.testY) ), delimiter="," )

if __name__ == '__main__':

    training_path   = 'MCS_Train-Tr15.csv'
    testing_path    = 'MCS_Test-Tr15.csv'
    data = Data( training_path, testing_path, need_preprocess = True) #load raw data
    data.save_train_test('MCS_test.csv')
    res = Modeling( data)