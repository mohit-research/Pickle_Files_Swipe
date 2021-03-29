from tqdm.notebook import tqdm
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from math import sqrt
from scipy.stats import gaussian_kde
from operator import itemgetter
import shutil
import math
import numpy as np
import statistics as stat
import random
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
# from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE
from authWithGAN import *

def HTER(y, pred):
        '''
        Params: (Expected Binary Labels)
        y: Original Labels
        pred: Predicted Labels

        Returns:
        --------------
        FAR, FRR, HTER respectively
        '''
        far = frr = 0
        for i in range(len(y)):
                if y[i] == 0 and pred[i] == 1:
                        far += 1
                if y[i] == 1 and pred[i] == 0:
                        frr += 1
        far /= len(y)   
        frr /= len(y)   
        hter = (frr + far)/2
        return far, frr, hter



def pickling(fname, obj):
        f = open(fname, "wb")
        pickle.dump(obj, f)
        f.close()

def unpickling(fname):
        f = open(fname, 'rb')
        g = pickle.load(f) 
        f.close()
        return g

def create_sliding_window(X, Y, n):
        final_X = []
        final_Y = []
        for i in range(len(X)- n):
                temp = []
                for j in range(i, i + n):
                        temp += X[j]
                final_X.append(temp)
                final_Y.append(Y[i+n])
        return final_X, final_Y

def binarize(u, id):

        ans = []
        for i in u:
                if int(i) == int(id):
                        ans.append(1)
                else:
                        ans.append(0)
        return ans


def compare_classification(label_name, model, device, user):
        ''' Function to process the data, apply oversampling techniques (SMOTE) and run the classification model specified using GridSearchCV
        Input:  label_name: The task to be performed (Gender, Major/Minor, Typing Style)
                feature_type: The feature set to be used (Desktop, Phone, Tablet, Combined)
                top_n_features: Thu number of features to be selected using Mutual Info criterion
                model: The ML model to train and evaluate
        Output: accuracy scores, best hyperparameters of the gridsearch run'''

        user = str(user)+"GANTab"
        if model == "SVM":      
                # Set the parameters by cross-validation
                
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',SVC())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid={'selector__k':[15, 25, 35, 47] }, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + ".pkl", clf)

                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_, far, frr, hter

        if model == "DTree":
                tuned_parameters = {
                        'selector__k':[15, 25, 35, 47]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',DecisionTreeClassifier(random_state=42))
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + ".pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_, far, frr, hter

        if model == "RForest":
                tuned_parameters = {
                        'selector__k':[15, 25, 35, 47]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',RandomForestClassifier())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + ".pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_, far, frr, hter

        if model == "XGBoost":
                tuned_parameters = {
                'selector__k': [15, 25, 35, 47],
                'model__min_child_weight': [1, 5, 10]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',xgb.XGBClassifier())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)

                pickling(model + '_' + str(user) + ".pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_, far, frr, hter

        if model == "ABoost":
                tuned_parameters = {'selector__k': [15, 25, 35, 47], 'model__n_estimators':[10, 50, 100],'model__learning_rate':[ 0.001, 0.01, 0.1], 'model__algorithm':["SAMME", "SAMME.R"]}

                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model',AdaBoostClassifier())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)
                pickling(model + '_' + str(user) + ".pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_, far, frr, hter

        if model == "MLP":
                tuned_parameters = {
                        'selector__k':[15, 25, 35, 47],
                        'model__hidden_layer_sizes': [(75,), (100,), (125,), (150,)]
                }
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model', MLPClassifier())
                ]
                )
                clf = GridSearchCV(
                         estimator=pipeline, param_grid=tuned_parameters, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)
                pickling(model + '_' + str(user) + ".pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_, far, frr, hter

        if model == "NB":
                pipeline = Pipeline(
                [
                        ('selector',SelectKBest(mutual_info_classif)),
                        ('model', GaussianNB())
                ]
                )
                clf = GridSearchCV(
                        estimator=pipeline, param_grid = {'selector__k':[15, 25, 35, 47]}, scoring='accuracy', return_train_score=True
                )
                clf.fit(X_train, y_train)
                pickling(model + '_' + str(user) + ".pkl", clf)
                y_true, y_pred = y_test, clf.predict(X_test)
                far, frr, hter = HTER(y_true, y_pred)
                return accuracy_score(y_true, y_pred), None, None, far, frr, hter


results = []
file_ptr = open("results_GAN_Tab.out", "w")
for val in range(2, 7):

        X = unpickling("Pickle_Files_Swipe/tab_features_X.pkl")
        y = unpickling("Pickle_Files_Swipe/tab_features_Y.pkl")
        u = unpickling("Pickle_Files_Swipe/tab_Ids_U.pkl")

        print ("Running for User", val)
        file_ptr.write(str(val)+"\n")
    
        select_user = binarize(u, str(val))

        X, y = create_sliding_window(X, select_user, 5)
        X = np.array(X)
        y = np.array(y)

        legit = []
        adversary = []

        '''for i in range(len(X)):
                if y[i] == 1:
                        legit.append(X[i])
                else:
                        adversary.append(X[i])'''

        #X1 = generate_samples(legit, 1000 - len(legit)) + legit
        #print("HERE")
        #X0 = generate_samples(adversary, 1000)

        #X_matrix, y_vector = X1 + X0, [1]*1000 + [0]*1000

        X_matrix, y_vector = SVMSMOTE().fit_resample(X, y)
        #X_matrix, y_vector = X, y
        X_train, X_test, y_train, y_test = train_test_split(
                        X_matrix, y_vector, test_size=0.4, stratify = y_vector, random_state=0)
        X_test = unpickling("tab_X"+str(val)+".pkl")
        y_test = unpickling("tab_Y"+str(val)+".pkl")

        for i in range(len(X_train)):
            if y_train[i] == 1:
                legit.append(X_train[i])
            else:
                adversary.append(X_train[i])

        #print("LOGS MOHIT:",len(legit))
        X1 = np.concatenate((generate_samples(legit, 3500),legit), axis=0)
        #print(X1.shape)
        #print("LOGS MOHIT:",len(adversary))
        X0 = np.concatenate((generate_samples(adversary, 3500),adversary), axis=0)
        #print(X0.shape)
        X_matrix, y_vector = np.concatenate((X1, X0),axis=0), [1]*len(X1)+[0]*len(X0)

        #X_matrix, y_vector = SVMSMOTE().fit_resample(X, select_user)
        scaler = preprocessing.StandardScaler()
        X_matrix = scaler.fit_transform(X_matrix)
        X_train = X_matrix
        y_train = y_vector
        # Split the dataset in two equal parts to remove unseen data

        #X_train, X_test, y_train, y_test = train_test_split(
        #X_matrix, y_vector, test_size=0.4, stratify = y_vector, random_state=0)
        #pickling("X"+str(val)+".pkl", X_test)
        #pickling("Y"+str(val)+".pkl", y_test)

        res = []
        from warnings import simplefilter
        from sklearn.exceptions import ConvergenceWarning
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)
        simplefilter(action='ignore', category=ConvergenceWarning)

        # function to call the compare_classification function for the specified model, feature_type and task
        def classification_results(problem, model, device, val):
                ac, setup, vali, far, frr, hter = compare_classification(problem, model, device,val)
                res.append([ac, frr, far, hter])
                print ("Accuracy: ", ac)
                file_ptr.write("Accuracy: "+str(ac)+"\n")
                print ("FRR:", frr)
                file_ptr.write("FRR:"+str(frr)+"\n")
                print ("FAR:", far)
                file_ptr.write("FAR:"+str(far)+"\n")
                print ("HTER", hter)
                file_ptr.write("HTER:"+str(hter)+"\n")
                #print(setup)
                #print(val)

        device = ["Phone"]
        class_problems = ["Authentication"]
        models = ["NB", "RForest", "XGBoost", "MLP", "SVM"]
        # models = ["SVM"]

        for model in models:
                print("###########################################################################################")
                print(model)
                file_ptr.write("Model is:"+str(model)+"\n")
                for class_problem in class_problems:
                        print(class_problem)
                        for dev in device:
                                print(dev)
                                classification_results(class_problem, model, dev, val)
                                print()
                                print("-----------------------------------------------------------------------------------------")
        results.append(res)

print (results)
file_ptr.write("Results:"+str(results)+"\n")

def avg_utility(r):
        final = []
        for m in range(len(r[0])):
                tmp = []
                for k in range(len(r[0][0])):
                        ans = 0
                        for i in range(len(r)):
                                ans += r[i][m][k]
                        ans /= len(r)
                        tmp.append(ans)
                final.append(tmp)
        return final

print (avg_utility(results))
file_ptr.write("Average:"+str(avg_utility(results))+"\n")




