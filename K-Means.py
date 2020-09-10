''' For more visit
https://www.codersarts.com 
'''


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# read the train and test dataset

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

print(train_data.head())

# shape of the dataset

print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)

# Predict the missing target variable in the test data


train_x = train_data.drop(columns=['Sales_item'],axis=1)
train_y = train_data['Sales_item']

# seperate the independent and target variable on training data

test_x = test_data.drop(columns=['Sales_item'],axis=1)
test_y = test_data['Sales_item']

model = KMeans()  

# fit the model with the training data

model.fit(train_data)

# Number of Clusters
model.n_clusters

# predict the clusters on the train dataset

predict_train_data = model.predict(train_data)

predict_train_data 

# predict the target on the test dataset

predict_test_data = model.predict(test_data)

predict_test_data

# Now, we will train a model with n_cluster = 3

model = KMeans(n_clusters=3)

# fit the model with the training data

model.fit(train_data)

# Number of Clusters

model.n_clusters

# predict the clusters on the train dataset

predict_train_data_model = model.predict(train_data)

predict_train_data_model 

# predict the target on the test dataset

predict_test_data_model = model.predict(test_data)

predict_test_data_model 