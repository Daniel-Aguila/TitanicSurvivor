from cgi import test
import os
from pickle import FALSE
from random import random
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
#read data with Pandas
train_data = pd.read_csv("/Users/daguila/Documents/ML/Titanic_Comp/train.csv")
test_data = pd.read_csv("/Users/daguila/Documents/ML/Titanic_Comp/test.csv")

#clean data
print("data size pre-cleaning(rows):", train_data.shape[0])
def percentageOfMissingData(missing_values, total_values):
    return (missing_values/total_values)*100

#number of missing data pointss
missing_values_count = train_data.isnull().sum()
#total data
total_cells = np.product(train_data.shape)
total_missing = missing_values_count.sum()
print("Percentage of Missing Data:",percentageOfMissingData(total_missing,total_cells))



print("Missing Values:",missing_values_count[0:10])
train_data=train_data.drop(columns=['PassengerId','Name', 'Cabin','Ticket']) #dropping 'Name' column since survival is not dependent on Name
test_data=test_data.drop(columns=['PassengerId','Name','Cabin','Ticket'])

#average age
average_age = train_data['Age'].mean()
mode_age = train_data['Age'].mode()
print("Average age:",average_age)
print("Mode Age:",mode_age)
train_data=train_data.fillna(float(mode_age))
test_data=test_data.fillna(float(mode_age))
train_data=train_data.replace({"female":0, "male":1}) #changing the quantitative value to numerical for females and males with 0,1 respectavely 
test_data=test_data.replace({"female":0,"male":1})
train_data=train_data.replace({"S":0,"Q":1,"C":2})
test_data=test_data.replace({"S":0,"Q":1,"C":2})
#Sex groups
male_total_data = train_data[(train_data['Sex']==1)]
female_total_data = train_data[(train_data['Sex']==0)]
#Split data for training (ie: features and labels)
train_features = train_data.copy()
train_labels = train_features.pop('Survived')
print("train_feature describe:",train_features.describe())
print(train_features.head())
#graph
def hist_Age():
    male_total_data = train_data[(train_data['Sex']==1)]
    female_total_data = train_data[(train_data['Sex']==0)]
    plt.hist(male_total_data['Age'],bins=10,color="green",label="Male",ec='k')
    plt.hist(female_total_data['Age'],bins=10,color="red",label="Female",ec='k')
    plt.legend()
    plt.xlabel("Age")
    plt.ylabel("People")
    plt.show()

def sex_to_survival():
    male_survived_data = male_total_data[(male_total_data['Survived']==1)]
    male_no_survived_data = male_total_data[(male_total_data['Survived']==0)]
    female_survived_data = female_total_data[(female_total_data['Survived']==1)]
    female_no_survived_data = female_total_data[(female_total_data['Survived']==0)]
    labels = ['Survived', 'Died']
    plt.pie([len(male_survived_data),len(male_no_survived_data)],autopct='%1.1f%%',labels=labels)
    plt.title("Males")
    plt.show()
    plt.pie([len(female_survived_data),len(female_no_survived_data)], autopct='%1.1f%%', labels=labels)
    plt.title("Females")
    plt.show()

#train
knn = KNN(n_neighbors=5)
RFC = RandomForestClassifier()
treeClass = tree.DecisionTreeClassifier()

#search for best parameters
params = [{
    'n_estimators': [100,200,300,400,500,600,700,800],
    'max_depth': ['None',2,5,10,15,20,30,50],
    'min_samples_split': [2,3,4,5],
    'min_samples_leaf': [1,2]
}]

param_to_test =[{
    'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
    'max_depth': [2],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}]
gs_rfc = GridSearchCV(RFC,param_grid=param_to_test,scoring='accuracy',cv=5,verbose=5)



gs_rfc.fit(train_features,train_labels)
print('Best parameters for RFC/n',gs_rfc.best_params_)
#current best parameters
# Max_depth: 50,
# Min_samples_leaf: 2,
# min_samples_split: 5,
# n_estimators: 400
y_pred_train = gs_rfc.predict(train_features)
print(metrics.accuracy_score(train_labels,y_pred_train))
#predict for the test data now
y_pred_test = gs_rfc.predict(test_data)

#fill out the submission
submission = pd.read_csv('/Users/daguila/Documents/ML/Titanic_Comp/gender_submission.csv')
submission['Survived'] = y_pred_test
submission.to_csv('/Users/daguila/Documents/ML/Titanic_Comp/submissions/Submission.csv', index = False)


#Next submission is ready, put that you changed the Age from average to  mode again but with the model improved against overfitting