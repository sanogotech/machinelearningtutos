''' For more visit
https://www.codersarts.com 
'''


import pandas as pd
from sklearn.linear_model import LinearRegression
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

model = LinearRegression()

# fit the model with the training data

model.fit(train_x,train_y)

# coefficeints of the trained model

model.coef_

# intercept of the model

model.intercept_

# predict the target on the test dataset

predict_train = model.predict(train_x)

predict_train 

# Root Mean Squared Error on training dataset

rmse_train = mean_squared_error(train_y,predict_train)**(0.5)

rmse_train

# predict the target on the testing dataset

predict_test = model.predict(test_x)

predict_test 

# Root Mean Squared Error on testing dataset

rmse_test = mean_squared_error(test_y,predict_test)**(0.5)

rmse_test