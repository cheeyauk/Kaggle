# Titanic dataset
# Code used for preprocessing and modeling for Kaggle Titanic dataset
# By : Kee Chee Yau
# Last edited : September 28, 2017

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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
actual_result = pd.read_csv('gender_submission.csv')

# view distribution of name
full = train.append(test)
full['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
full['Family_Name'] = full['Name'].map(lambda name: name.split(' ')[0].strip(','))
popularNames = full.groupby('Family_Name').count().reset_index()
popularNames = popularNames['Family_Name'][popularNames['Pclass']>=5]
full['PopularName'] = popularNames.isin(full['Family_Name'])
full['Family_Name'][full['PopularName']!=True] ='Others'
full['fSize'] = full['SibSp']+full['Parch']+1

# fill with most frequent
full["Embarked"] = full["Embarked"].fillna("S")

title = pd.get_dummies(full.groupby('Title').agg('count'))
title['Pclass'].plot.bar(rot=90)

Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Mr",
                    "Ms":         "Miss",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

# visualize distribution of fare
full['Fare'].hist(bins=20)
full['Age'].hist()

# distribution of age versus survived, identify age group
def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

plot_distribution( full , var = 'Age' , target = 'Survived' , row = 'Sex' )
plot_distribution( full , var = 'fSize' , target = 'Survived' )

# check distribution of data
full.corr()
sns.pairplot(full.fillna(train.mean()))

#https://www.kaggle.com/c/titanic/data
full.isnull().sum()
full.describe() # 512 max, std = 49.6
nSurvived = full['Survived'][full['Survived']==1].size
farePrice = full.groupby(['Pclass'])['Fare'].mean()

# view distribution of dataset
full.hist()
plt.tight_layout()
full['TicketType'] = full['Ticket'].apply(lambda x: x[0])
tickettype = pd.get_dummies(full.groupby('TicketType').agg('count'))
tickettype['Pclass'].plot.bar(rot=90)
ttSummary = pd.crosstab(full['TicketType'], full['Survived']).reset_index()
ttSummary['BadTicket'] = ttSummary.iloc[:,1] > 3*ttSummary.iloc[:,2]

BadTickets = ttSummary['TicketType'][ttSummary['BadTicket']==True]

# map names
full['Title']=full['Title'].map( Title_Dictionary )
full['Cabin'] = full['Cabin'].fillna('Unkwown')
full['Cabin_Unknown'] = full['Cabin'].apply(lambda c: 1 if c[0] == 'U' else 0)
full['Cabin_n'] = full['Cabin'].apply(lambda c: c[1] if ((c[0] != 'U') & (len(c) > 1)) else 'U')
full['Cabin_n'] = full['Cabin_n'].apply(lambda c: c if c != ' ' else 'U')
full['Cabin'] = full['Cabin'].map(lambda c:c[0])

# If some one has sibling and parents they will save them? will count confuse the model?
full['HasSiblings'] = full['SibSp'].apply(lambda x: 1 if x>=1 else 0)
full['HasParentsChildren'] = full['Parch'].apply(lambda x: 1 if x>=1 else 0)

# convert class to string so dummy will work
full['Class'] = full['Pclass'].apply(str)

# fill in missing values
index_NaN_age = list(full["Age"][full["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = full["Age"].median()
    age_pred = full["Age"][((full['SibSp'] == full.iloc[i]["SibSp"]) & (full['Parch'] == full.iloc[i]["Parch"]) & (full['Pclass'] == full.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        full.loc[i,'Age'] = age_pred
    else :
        full.loc[i,'Age'] = age_med
               
full['Fare']=full['Fare'].fillna(full['Fare'].median())

# number of unique tickets
print(full['Ticket'].nunique()," unique tickets out of ", len(full), "passengers")
full['Ticket_group'] = full.groupby('Ticket')['Name'].transform('count')
full['Fare_eff'] = full['Fare']/full['Ticket_group']
full['Ticket_numbers'] = full['Ticket'].map(lambda name: name.split(' ')[1].strip() if name.find(' ') >0 else name)
full['Ticket_len'] = full['Ticket_numbers'].apply(lambda tic:len(tic))
full['Tic_1st_n'] = full['Ticket_numbers'].apply(lambda tic:tic[0])
#full['Tic_2nd_n'] = full['Ticket_numbers'].apply(lambda tic:tic[1])
shared = full.groupby('Ticket').count().reset_index()
sharedTickets = shared[shared['Pclass']>1]['Ticket']

full['SharedTicket'] = sharedTickets.isin(full['Ticket'])
full['SharedTicket'] = full['SharedTicket'].apply(lambda x: 1 if x == True else 0)
full['BadTicket'] = BadTickets.isin(full['TicketType'])
full['BadTicket'] = full['BadTicket'].apply(lambda x: 1 if x == True else 0)

# select columns
full_p = full.loc[:,['Family_Name','Cabin_Unknown','Cabin_n','Ticket_group','Ticket_len','Tic_1st_n','TicketType','fSize','Cabin','Title','SibSp','Parch','Class','Age','Sex','Fare','Fare_eff','Embarked','SharedTicket','BadTicket']]

# Age Group
def ageGroupCalc(age):
    if age<3:
        return '0'
    elif age < 15: 
        return '1'
    elif age < 40:
        return '2'
    elif age < 60:
        return '3'
    else:
        return '4'
        
# Family Size
def fSizeCalc(size):
    if size==1:
        return '0'
    elif  size == 2 :
        return '1'
    elif 3<= size <= 4:
        return '2'
    elif size > 4:
        return '3'

full_p['AgeGroup'] = full_p['Age'].apply(ageGroupCalc)
full_p['fSizeGroup'] = full_p['fSize'].apply(fSizeCalc)
full_p['Child'] = full_p['Age'].apply(lambda x: 1 if x<10 else 0)
full_p['isAlone'] = full_p['fSizeGroup'].apply(lambda x: 1 if x=='0' else 0)
      
# Fare categories
def fareCalc(fare):
    if fare <=10:
        return '0'
    elif 10 < fare <=100: 
        return '1'
    elif 100 < fare <= 300:
        return '2'
    elif fare > 300:
        return '3'
full_p['FareGroup'] = full_p['Fare'].apply(fareCalc)

y_train = train['Survived']
full_p = pd.get_dummies(full_p)


#delete unneeded columns.
del full_p['Age']
#del full_p['Class_1']
#del full_p['Sex_female']
#del full_p['Embarked_S']
#del full_p['Title_Royalty']
#del full_p['Cabin_U']
#del full_p['AgeGroup_4']
#del full_p['fSizeGroup_2']
del full_p['fSize']
del full_p['Fare']
#del full_p['FareGroup_2']


X_train = full_p[:891]
X_test = full_p[891:]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ytest = []

# Use PCA to extract important components
from sklearn.decomposition import PCA
pca = PCA(n_components =10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
explained_variance.sum()

# Kernel PCA
#from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components = 10, kernel = 'rbf')
#X_train = kpca.fit_transform(X_train)
#X_test = kpca.transform(X_test)

#===========================================================
# Bayes
#===========================================================
from sklearn.naive_bayes import GaussianNB

def runBayes(X_train,X_test,y_train):
    gnb = GaussianNB() 
    gnb.fit(X_train, y_train)
    y_test = gnb.predict(X_test)
    return y_test

#===========================================================
# Random Forest
#===========================================================
from sklearn.ensemble import RandomForestClassifier
def runRandomForest(X_train,X_test,y_train):
# https://hubpages.com/technology/Adaboost-in-Python
    rfc = RandomForestClassifier(n_estimators=300,criterion='gini',bootstrap = False, max_features = 3,min_samples_split=2,min_samples_leaf=10)
    rfc.fit(X_train, y_train)
    print(rfc.score(X_train, y_train))
    y_test=rfc.predict(X_test)
    
    return rfc,y_test

#===========================================================
# SVC
#===========================================================
from sklearn.svm import SVC
def runSVC(X_train,X_test,y_train):
# https://hubpages.com/technology/Adaboost-in-Python
    clf = SVC(kernel = 'rbf', random_state = 0)
    clf.fit(X_train, y_train)
    #print(clf.score(X_train, y_train))
    y_test=clf.predict(X_test)
    return clf, y_test

#===========================================================
# Solution 1 : Creates a bagging Keras model 
#===========================================================

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
    classifier.add(Dense(output_dim = 100, init = 'uniform', activation = Relu, input_dim = 10))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(output_dim = 100, init = 'uniform', activation = Relu, input_dim = 100))
    classifier.add(Dropout(0.3))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adagrad', loss = 'binary_crossentropy',  metrics = ['accuracy'])
    return classifier

def runNN1H(X_train,X_test,y_train):
    classifier = create_SingularNN()
    classifier.fit(X_train, y_train, nb_epoch = 200)
    return classifier, (classifier.predict(X_test) > 0.5).astype(int)


# Bagging model
from sklearn.model_selection import KFold
nSplit = 5
kf = KFold(n_splits=nSplit)
kf.get_n_splits(X_train)
acc = []
clflist = []
ytest = []
for train_index, test_index in kf.split(X_train):  
    X_tr =[]
    X_tr = X_train[train_index]
    X_ts = X_train[test_index]
    y_tr, y_ts = y_train[train_index], y_train[test_index]
    X_tr.astype(int);X_ts.astype(int);
    #clf, y_pr= runSVC(X_tr,X_ts,y_tr)
    clf, y_pr= runNN1H(X_tr,X_ts,y_tr)
    #clf, y_pr= runRandomForest(X_tr,X_ts,y_tr)
    #clf, y_pr= run_Ada(X_tr,X_ts,y_tr)
    acc.append(accuracy_score(y_pr,y_ts))
    clflist.append(clf)

for i in range(len(clflist)):
    clf = clflist[i]
    ytesti = (clf.predict(X_test) > 0.5).astype(int) 
    ytest.append(ytesti)

ynn= (np.mean(ytest,axis=0) > 0.5).astype(int) 

np.average(acc)
#ytest = runBayes(X_train,X_test,y_train)
#ytest = runRandomForest(X_train,X_test,y_train)

accuracy1 = accuracy_score(actual_result['Survived'],ytest[0])
accuracy2 = accuracy_score(actual_result['Survived'],ytest[1])
accuracy3 = accuracy_score(actual_result['Survived'],ytest[2])
accuracy4 = accuracy_score(actual_result['Survived'],ytest[3])
accuracy5 = accuracy_score(actual_result['Survived'],ytest[4])

# bagging average
accuracynn  = accuracy_score(actual_result['Survived'],ynn)

Result = pd.DataFrame(test['PassengerId'])
Result['Survived'] = ynn
Result.to_csv('TitanicResult.csv',index=False)

#===========================================================
# Solution 2 : ENSEMBLE with Grid Search
#===========================================================
# Model 1. Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
rf_param_grid = {"max_depth": [None],
              "max_features": [7,5,3],
              "min_samples_split": [2,5,10,15],
              "min_samples_leaf": [10,15,20],
              "bootstrap": [False],
              "n_estimators" :[150],
              "criterion": ["gini"]}

grid_search = GridSearchCV(estimator = clf,
                           param_grid = rf_param_grid,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = 6,verbose=1)
rfc_gs = grid_search.fit(X_train,y_train)
rfc_best = rfc_gs.best_estimator_
rfc_param = rfc_gs.best_params_
rfc_score = rfc_gs.best_score_

# Model 2. SVC
from sklearn.svm import SVC
clf = SVC(probability=True)
from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.5,0.8,1,1.2,1.5,2.0], 'kernel':['rbf'], 'gamma':[0.2,0.15, 0.1,0.09]}
svm_gs = GridSearchCV(estimator = clf,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = 6,
                           verbose = 1)
svm_gs = svm_gs.fit(X_train,y_train)
svc_param = svm_gs.best_params_
svc_best =  svm_gs.best_estimator_
svc_score = svm_gs.best_score_

# Model 3. Adaboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[15,25,100],
              "learning_rate":  [ 0.01, 0.05, 0.1, 0.15]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=5, scoring="accuracy", n_jobs= 6, verbose = 1)

gsadaDTC.fit(X_train,y_train)
ada_params = gsadaDTC.best_params_
ada_best = gsadaDTC.best_estimator_
ada_score = gsadaDTC.best_score_

# Model 4. MLP Classifier 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=300)

ann_param_grid = {"activation":["relu","logistic"],
                  "solver" : ["lbfgs", "sgd","adam"],
                  "hidden_layer_sizes":[150],
              "alpha" : [0.001,0.005],
             "learning_rate_init" : [0.001,0.01,0.05],
             "batch_size" : [50,100,200]
              }
ann_gs = GridSearchCV(mlp,param_grid = ann_param_grid, cv=5, scoring="accuracy", n_jobs= 6, verbose = 1)
ann_gs = ann_gs.fit(X_train,y_train)

ann_acc = ann_gs.best_score_
ann_param = ann_gs.best_params_
ann_best =  ann_gs.best_estimator_

# Model 5. Gradient Boosting trees
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8,10],
              'min_samples_leaf': [50,100,150],
              'max_features': [0.3, 0.1,0.5] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=5, scoring="accuracy", n_jobs= 6, verbose = 1)

gsGBC.fit(X_train,y_train)
gbc_param = gsGBC.best_params_
gbc_param = gsGBC.best_score_
GBC_best = gsGBC.best_estimator_

# Final Step. Create a Voting classifier based to ensemble results
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('rfc', rfc_best), ('svc', svc_best),('ada',ada_best),('ann',ann_best),('gbc',GBC_best)], voting='soft', n_jobs=6)
ann_best.fit(X_train, y_train)

votingC = votingC.fit(X_train, y_train)

y_vote = votingC.predict(X_test).reshape(-1,1)

accuracyf  = accuracy_score(actual_result['Survived'],y_vote)

Result = pd.DataFrame(test['PassengerId'])
Result['Survived'] = y_vote
Result.to_csv('TitanicResult.csv',index=False)
help(pd.DataFrame.to_csv)
