
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as pl
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import preprocessing

warnings.filterwarnings('ignore')
data = pd.read_csv('data/raw/insurance.csv')
le = preprocessing.LabelEncoder()
le.fit(data["sex"])
data["Sex"] = le.transform(data["sex"])
le.fit(data["smoker"])
data["Smoker"] = le.transform(data["smoker"])
le.fit(data["region"])

data["Region"] = le.transform(data["region"])
y = data["charges"]
x = data[["age", "bmi", "children", "Sex", "Smoker", "Region"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)
predictions = linreg.predict(X_test)
linreg.score(X_test,y_test)

with open('data/model/trained_model.pkl', 'wb') as f:
    pickle.dump(linreg, f)
with open('data/model/trained_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)
model = pickle.load(open('data/model/trained_model.pkl','rb'))
print(model.predict([[55, 18, 0, 1, 1, 1]]))