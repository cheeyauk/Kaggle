# Leaf Classification
# By : Kee Chee Yau
# Last edited : October 2, 2017

# Recommended software versions
# Python 3.6
# sklearn 0.18

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import math
from sklearn import preprocessing

# extracting train and test pre-processed features file
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Analyse the data
species = train['species'].unique()

y_train = np.array(pd.get_dummies(train['species']))

train.corr().iloc[1,:]

del train['species']
del train['id']
del test['id']

# scale the output
sc = StandardScaler()
X_train = sc.fit_transform(train)
X_test = sc.transform(test)

ytest = []

import keras
from keras.models import Sequential
from keras.layers import Dense ,Dropout
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score 

def create_SingularNN():
    # Initialising the ANN
    classifier = Sequential()
    #Relu = keras.layers.advanced_activations.ThresholdedReLU(init='zero', weights=None)
    # Adding the input layer and the first hidden layer
    Relu = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
    classifier.add(Dense(output_dim = 700, init = 'uniform', activation = Relu, input_dim = 192))
    classifier.add(Dropout(0.3))

    # Adding the output layer
    classifier.add(Dense(output_dim = 99, init = 'uniform', activation = 'softmax'))
    # Compiling the ANN
 
    classifier.compile(optimizer = 'adagrad', loss = 'categorical_crossentropy',  metrics = ['accuracy'])
    return classifier

def runNN1H(X_train,X_test,y_train):
    classifier = create_SingularNN()
    classifier.fit(X_train, y_train, nb_epoch = 100)
    return classifier, (classifier.predict_proba(X_test) >0.5).astype(int)

# Bagging ANN classifier using keras
from sklearn.model_selection import KFold
nSplit = 5
kf = KFold(n_splits=nSplit,shuffle=True)
kf.get_n_splits(X_train,y_train)
acc = []
clflist = []
y_test = []
for train_index, test_index in kf.split(X=X_train):  
    X_tr =[]
    X_tr = X_train[train_index]
    X_ts = X_train[test_index]
    y_tr, y_ts = y_train[train_index], y_train[test_index]
    X_tr.astype(int);X_ts.astype(int);
    clf, y_pr= runNN1H(X_tr,X_ts,y_tr)
    acc.append(accuracy_score(y_pr,y_ts))
    clflist.append(clf)

for i in range(len(clflist)):
    clf = clflist[i]
    ytest = clf.predict_proba(X_test)
    y_test.append(ytest)

ynn= (np.mean(y_test,axis=0) > 0.5).astype(int) 

np.average(acc)

# Save results into a csv file (please change this to a correct headings based on the sample_submission.csv)
Result = pd.DataFrame(ynn)
Result.to_csv('LeafResult.csv',index=False)
    