
"""
Steps in ML project
1. data collection(csv,web scraping, json, database)
2. clean data by processing it to become usable to train the model, data preparation
(data preparation cannot be automated as dataset varies with project)
3. [Train-test-split, label encoding(or say categorical data handling),
normalization(or called feature scaling)]
4. Train model
5. Performance evaluation
6. Deployment
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv("student_scores.csv")

features = dataset['Hours'].values
labels = dataset['Scores'].values
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
features=features.reshape(25,1)
regressor.fit(features, labels)
x=[9]
x=np.array(x)
x=x.reshape(1,1)
regressor.predict(x)
#to convert 9 into d array

#drawing the best fit line for this dataset
import matplotlib.pyplot as plt
plt.scatter(features, labels)
plt.plot(features, regressor.predict(features)) #in second parameter we need to pass values of predicted
#the points closer to the best fit line will have lower error 

#Train-test-split
dataset = pd.read_csv("student_scores.csv")

features = dataset['Hours'].values
labels = dataset['Scores'].values

from sklearn.model_selection  import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2)
len(features_train)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features_train, labels_train)
features_train=features_train.reshape(20,1)
features_test=features_test.reshape(5,1)
pred = regressor.predict(features_test)

labels_test

pd.DataFrame(zip(pred, labels_test))
#zip is in built method in Python. pd.DataFrame->typecast this into a dataframe

#train score
regressor.score(features_train, labels_train)

#test score
regressor.score(features_test, labels_test)

hours=[10]
type(hours)

# we cant pass this list data to prediction, we need to convert it into numpy
hours = np.array(hours)
#this data should be in horizontal format unlike vertical as in current.
#so convert this data to horizontal i.e change the shape

hours = hours.reshape(1,1)

#now pass this data to regressor.predict
regressor.predict(hours)

















