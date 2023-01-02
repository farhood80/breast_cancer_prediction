#!/usr/bin/env python
# coding: utf-8

# <b> Breast Cancer Prediction Project

# In[76]:


# importing dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[77]:


# data collection and organizing

# loading the Data from sklearn

breast_cancer = sklearn.datasets.load_breast_cancer()


# In[78]:


print(breast_cancer)


# In[79]:


#loading the data into pandas standard dataframe

breast_cancer1 = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)


# In[80]:


#printing the n rows of the data frame
breast_cancer1.head(10) 


# In[81]:


#getting info about the data

breast_cancer1.info() #cehck the type of the data and if there is any null


# In[82]:


# checking the shape of the data
breast_cancer1.shape


# In[83]:


#adding the target columns to he data frame

breast_cancer1['result'] = breast_cancer.target


# In[84]:


breast_cancer1.head(10)


# In[85]:


# checking the shape of the data
breast_cancer1.shape


# In[86]:


# measuring dataset
breast_cancer1.describe()


# In[87]:


# checking for missing values

breast_cancer1.isnull().sum() #its great data there is no missing values


# In[88]:


# checking the distribution of the target variables

breast_cancer1['result'].value_counts()


# <b> 1 => benign
#     
#   <b>  2 => malignant

# In[89]:


breast_cancer1.groupby('result').mean()


# In[90]:


# seprating the features and target

x = breast_cancer1.drop(['result'], axis=1) # or x = breast_cancer1.drop(columns = 'result', axis=1)
y = breast_cancer1['result']


# In[91]:


print(x)
print(y)


# In[92]:


#spliting data int otraining and test set

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.2, random_state =2)


# In[93]:


# check the division 

print(x.shape , x_train.shape , x_test.shape)


# In[94]:


#model training with logistic regression
model = LogisticRegression()


# In[95]:


#training the logistic regression model
model.fit(x_train , y_train)


# In[96]:


#model evalution and accuracy score of the training data

x_train_prediction = model.predict(x_train)
training_data_accuray = accuracy_score(y_train , x_train_prediction)


# In[97]:


# checking the precentage of accuracy training data
print("the  of the training data is : " , training_data_accuray) #wow its too good


# In[98]:


# accuracy score of the test data

x_test_prediction = model.predict(x_test)
test_data_accuray = accuracy_score(y_test , x_test_prediction)


# In[99]:


# checking the precentage of accuracy test data
print("the  of the test data is : " , test_data_accuray) #its great very nice datset


# <B> #Note :  the lesser the drop of accuracy in test the more accurate the data is

# In[ ]:


#building a predictive system

input_data = input("please enter the data: ")



# In[ ]:


# convert the entered data into standard pandas dataset

input_data_as_np = np.asanyarray(input_data)

#reshape the np array to accepting datapoint

input_data_reshaped = input_data_as_np.reshape(1,-1) 

prediction = model.predit(input_data_reshaped)
print(prediction)


# In[ ]:


if (prediction == [0]):
    print("the breast cancer is malignent")
else:
    print("the breast cancer is benign")


# In[ ]:




