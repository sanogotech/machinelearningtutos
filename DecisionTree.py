''' For more visit
https://www.codersarts.com 
'''


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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

model = DecisionTreeClassifier()

# fit the model with the training data
model.fit(train_x,train_y)

# coefficeints of the trained model

model.coef_

# intercept of the model

model.intercept_

# predict the target on the train dataset

predict_train_data = model.predict(train_x)

predict_train_data 

# Accuray Score on train dataset

accuracy_train_data = accuracy_score(train_y,predict_train)

accuracy_train_data

# predict the target on the test dataset

predict_test_data = model.predict(test_x)

predict_test_data

# Accuracy Score on test dataset

accuracy_test_data = accuracy_score(test_y,predict_test)

accuracy_test_data