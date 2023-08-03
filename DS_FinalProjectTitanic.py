#!/usr/bin/env python
# coding: utf-8

# Nayri Tagmazian
# PHYS 247 Final Project (Titanic)

# In[229]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential as sq
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[230]:


# importing training data to dataframe
train_df = pd.read_csv('train.csv')
train_df.head()


# In[231]:


train_df.shape # train_df has 891 rows and 12 columns


# In[232]:


# checking to see which columns have null values, if any, and the data type of each column
train_df.info() # the only columns that have null values are Age, Cabin, and Embarked


# In[233]:


# counting null values of each column
train_df.isna().sum()


# In[234]:


# Cabin column is missing a majority of values (687/891) and have no way to fill in missing values, will have to drop that column
train_df = train_df.drop(columns = ['Cabin'], axis = 1)
train_df.head()


# In[235]:


train_df.shape # now have 891 rows and 11 columns


# In[236]:


# bar plot counting the number of passengers who survived and died 
sns.countplot(x = "Survived", data = train_df)
plt.title("Passenger Count by Survived", size = 15)
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
print(train_df["Survived"].value_counts()) # of the 891 passengers on the Titanic, 549 died while only 342 survived


# In[237]:


# bar plot counting the number of male and female passengers
sns.countplot(x = "Sex", data = train_df)
plt.title("Passenger Count by Sex", size = 15)
plt.xlabel("Gender of Passengers")
plt.ylabel("Count")
plt.show()
print(train_df["Sex"].value_counts()) # there were 577 males and 314 females passengers on the Titanic


# In[238]:


# bar plot counting the number of passengers in each class (Pclass)
sns.countplot(x = "Pclass", data = train_df)
plt.title("Passenger Count by Pclass", size = 15)
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()
print(train_df["Pclass"].value_counts()) # there were 216 passengers in the upper class, 184 passengers in middle class, and 491 passengers in lower class


# In[239]:


# bar plot counting the number of passengers who Embarked at each location
sns.countplot(x = "Embarked", data = train_df)
plt.title("Passenger Count by Embarked", size = 15)
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()
print(train_df["Embarked"].value_counts()) # 644 passengers embarked at Southampton, 168 at Cherbourg, and 77 at Queenstown


# In[240]:


# bar plot counting the number of passengers who survived and died (Survived) by gender (Sex)
sns.countplot(x = "Sex", data = train_df, hue = "Survived")
plt.title("Survival Count by Sex", size = 15)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show() # survival rate of females > survival rate of males
survial_by_sex = train_df.groupby(['Sex'])['Survived'].value_counts()
print("Survival by Sex:\n", survial_by_sex)


# In[241]:


# bar plot counting the number of passengers who survived and died (Survived) by Pclass
sns.countplot(x = "Pclass", data = train_df, hue = "Survived")
plt.title("Survival Count by Pclass", size = 15)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show() # survival rate of upper class > survival rate of lower class
survial_by_class = train_df.groupby(['Pclass'])['Survived'].value_counts()
print("Survival by Pclass:\n", survial_by_class)


# In[242]:


# bar plot counting the number of passengers who survived and died (Survived) by embarking location (Embarked)
sns.countplot(x = "Embarked", data = train_df, hue = "Survived")
plt.title("Survival Count by Embarked", size = 15)
plt.xlabel("Emarked Location")
plt.ylabel("Count")
plt.show() # most passengers embarked at Southampton, which had the lowest survival rate
# Cherbourg had the highest survival rate
survial_by_embarked = train_df.groupby(['Embarked'])['Survived'].value_counts()
print("Survival by Embarked:\n", survial_by_embarked)


# In[243]:


# bar plot of PClass by Sex
sns.countplot(x = "Sex", data = train_df, hue = "Pclass")
plt.title("Pclass by Sex", size = 15)
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show() # corresponds to bar plots above that survival rate of females higher than survival rate of males 
# bc more males in lower class and more of lower class perished from Titanic than upper class
class_by_sex = train_df.groupby(['Pclass'])['Sex'].value_counts()
print("Pclass by Sex:\n", class_by_sex)


# In[244]:


# bar plot counting the number of passengers of each Pclass by embarking location (Embarked)
sns.countplot(x = "Embarked", data = train_df, hue = "Pclass")
plt.title("Pclass by Embarked", size = 15)
plt.xlabel("Emarked Location")
plt.ylabel("Count")
plt.show() # most passengers embarked at Southampton, which had the highest count of lower class passengers
# from above we know that lower class had the least likelihood to survive, so Southamptom should have low survival rate
class_by_embarked = train_df.groupby(['Pclass'])['Embarked'].value_counts()
print("Pclass by Embarked:\n", class_by_embarked)


# In[245]:


# bar plot counting the number of passengers of each gender (Sex) by embarking location (Embarked)
sns.countplot(x = "Embarked", data = train_df, hue = "Sex")
plt.title("Sex by Embarked", size = 15)
plt.xlabel("Emarked Location")
plt.ylabel("Count")
plt.show() # most passengers embarked at Southampton, which had the highest count of male passengers
# from above we know that men had a lower likelihood of survival, so Southamptom should have low survival rate
sex_by_embarked = train_df.groupby(['Sex'])['Embarked'].value_counts()
print("Sex by Embarked:\n", sex_by_embarked)


# In[246]:


# calculating mean, median, and mode of Age column to see which value is best to use to fill null Age values
mean = train_df["Age"].mean()
median = train_df["Age"].median()
mode = train_df["Age"].mode()
mean, median, mode


# In[247]:


# age distribution graph of passengers 
plt.figure(figsize=(12,6))
sns.kdeplot(x = "Age", data = train_df.dropna())
plt.plot([mean, mean], [0, 0.04], color = "red", label = "mean")
plt.plot([median, median], [0, 0.04], color = "green", label = "median")
plt.plot([mode, mode], [0, 0.04], color = "purple", label = "mode")
plt.title("Age Distribution", size = 15)
plt.xlabel("Age")
plt.ylabel("Distribution")
plt.legend()
plt.show() # most passengers were young, in their 20s-40s
# median value looks like it's the best fit for missing values, will use median to fill in null values in Age


# In[248]:


# age distribution graph based off survival (Survived)
plt.figure(figsize=(20,10))
sns.kdeplot(x = "Age", data = train_df[train_df['Survived'] == 0].dropna(), shade = True, alpha = 1, label = "Died")
sns.kdeplot(x = "Age", data = train_df[train_df['Survived'] == 1].dropna(), shade = True, alpha = 0.5, label = "Survived")
plt.title("Age Distribution by Survival", size = 15)
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.show() # the older you are, the less likely you are to survive


# In[249]:


# calculating median age by PClass and Sex columns to use these values to fill in null age values
mean_age = train_df.groupby(['Pclass', 'Sex', 'Survived'])['Age'].mean()
print("Mean Age by Pclass and Survived:\n", mean_age)


# In[250]:


# creating function to fill in null age values based off PClass, Sex, and Survived
def mean_age_input(row):
    age = row[0]
    pass_class = row[1]
    gender = row[2]
    survival = row[3]
    
    if np.isnan(age):
        if pass_class == 1 and gender == 'female' and survival == 0:
            return 25.67
        elif pass_class == 1 and gender == 'female' and survival == 1:
            return 34.95
        elif pass_class == 1 and gender == 'male' and survival == 0:
            return 43.63
        elif pass_class == 1 and gender == 'male' and survival == 1:
            return 36.66
        elif pass_class == 2 and gender == 'female' and survival == 0:
            return 36.0
        elif pass_class == 2 and gender == 'female' and survival == 1:
            return 28.08
        elif pass_class == 2 and gender == 'male' and survival == 0:
            return 33.11
        elif pass_class == 2 and gender == 'male' and survival == 1:
            return 17.67
        elif pass_class == 3 and gender == 'female' and survival == 0:
            return 23.27
        elif pass_class == 3 and gender == 'female' and survival == 1:
            return 20.08
        elif pass_class == 3 and gender == 'male' and survival == 0:
            return 26.62
        else:
            return 22.8      
    else:
        return age
        
train_df["Age"] = train_df[["Age", "Pclass", 'Sex','Survived']].apply(mean_age_input, axis = 1)


# In[251]:


# counting null values of each column to ensure that all null values in Age column have been filled
train_df.isna().sum()


# In[252]:


train_df.shape # now have 891 rows and 11 columns


# In[253]:


train_df['Embarked'].value_counts()


# In[254]:


selected_rows = train_df[train_df['Embarked'].isna()]
selected_rows


# In[255]:


# looking for avg Embarked values based on Fare's that are approx $80
train_df_2 = train_df.loc[train_df['Fare'] == 81.8583, 'Embarked']
train_df_2


# In[256]:


train_df_3 = train_df.loc[train_df['Fare'] == 79.65, 'Embarked']
train_df_3


# In[257]:


# replacing the remaining two null values in Embarked column with S
train_df['Embarked'] =  train_df['Embarked'].fillna('S')


# In[258]:


# counting null values of each column to ensure that all null values in Embarked column have been dropped
train_df.isna().sum()


# In[259]:


# dropping PassengerID, Name, and Ticket columns because all three of them provide same, repeated information
# also because they are all unique inputs that do not provide insightful information 
train_df = train_df.drop(columns = ['PassengerId', 'Name', 'Ticket'], axis = 1)
train_df.head()


# In[260]:


# converting all remaining categorical variables in the dataset to numerical variables and putting into new columns
train_df['new_Sex'] = pd.factorize(train_df.Sex)[0]
train_df['new_Embarked'] = pd.factorize(train_df.Embarked)[0]


# In[261]:


# removing all categorical columns and keeping all numerical columns
train_df = train_df.drop(columns = ['Sex', 'Embarked'], axis = 1)
pd.set_option('display.max_columns', None)
train_df.head()


# In[262]:


train_df.shape


# In[263]:


# splitting data into 70% training and 30% testing
features = list(train_df.columns)
X = train_df.drop('Survived', axis = 1)
y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = .30)
print("shape of original dataframe :", train_df.shape)
print("shape of input - training set", X_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", X_test.shape)
print("shape of output - testing set", y_test.shape)


# In[264]:


# starting model building 


# In[265]:


# normalizing features using MinMaxScaler to prepare for Artifical Neural Network Model
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train) 
scaled_X_test = scaler.fit_transform(X_test) 


# In[266]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[267]:


# building the Artifical Neural Network model
nn_model = keras.Sequential()

# first layer
nn_model.add(Dense(30, input_dim = 7, activation = 'relu'))
nn_model.add(Dropout(0.5))

# second layer
nn_model.add(Dense(15, activation = 'relu'))
nn_model.add(Dropout(0.25))

# third layer
nn_model.add(Dense(1, activation = 'sigmoid'))


# In[268]:


# compiling the Artifical Neural Network model
nn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
nn_model.summary()


# In[269]:


# training the Artifical Neural Network Model
history = nn_model.fit(scaled_X_train, y_train, epochs = 125, batch_size = 32, 
                    callbacks = keras.callbacks.EarlyStopping(monitor = 'loss', mode = 'min', patience = 1), verbose = 1,
                    validation_data = (scaled_X_test, y_test))


# In[270]:


# evaluating Artifical Neural Network Model performance
nn_model_score = nn_model.evaluate(scaled_X_test, y_test, verbose = 0)


# In[271]:


# printing accuracy of Artifical Neural Network Model
print('Test Loss:', nn_model_score[0])
print('Test Accuracy:', nn_model_score[1])
print(nn_model.summary())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("ANN Test Accuracy", size = 15)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()
# ann method accuracy score of 78.36% which isn't too bad, curious to see if another method will get me a higher score
# will try logistic regression method next


# In[272]:


# using training data to train Logistic Regression Model 
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)


# In[273]:


# testing accuracy of Logistic Regression Model on both training and testing data
y_train_pred = logistic_regression_model.predict(X_train)
y_test_pred = logistic_regression_model.predict(X_test)
accuracy_training_data = accuracy_score(y_train, y_train_pred)
accuracy_test_data = accuracy_score(y_test, y_test_pred)
print('Accuracy of training data: ', accuracy_training_data)
print('Accuracy of test data: ', accuracy_test_data)
# logreg method accuracy score of 82.09% which is better, will keep this model


# In[274]:


confusion_matrix = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix) # we have 148+20 correct predictions, and 28+72 incorrect predictions


# In[275]:


print(classification_report(y_test, y_test_pred))


# In[276]:


logistic_regression_roc_auc = roc_auc_score(y_test, logistic_regression_model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logistic_regression_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logistic_regression_roc_auc)
plt.plot([0, 1], [0, 1], '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = "lower right")
plt.savefig('Log_ROC')
plt.show()

