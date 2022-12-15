#!/usr/bin/env python
# coding: utf-8

# # Assignment 4 - Combine Data Cleaning, Feature Selection, Modeling and Interpretability

# ## Abstract

# A tour & travels company is offering travel insurance package to their customers. The new insurance package also includes covid cover. The company wants to know which customers would be interested to buy it based on their database history. The insurance was offered to some of the customers in 2019 and the given data has been extracted from the performance/sales of the package during that period. The data is provided for almost 2000 of its previous customers and we are required to build an intelligent model that can predict if the customer will be interested to buy the travel insurance package.

# ## Information about the Dataset

# ##### What is our ultimate aim with this notebook?
# - Our ultimate aim is to build a few machine learning models, namely a linear model,tree-based model and AutoML model that can predict if the customer will be interested in buying the travel insurance package based on certain parameters. We will also run Shap Analysis for all of three models and explore the results/opservations. Before we create a model we need to do some data cleaning, feature selection and exploratory data analysis of the kaggle dataset.
# 
# <i><b>Kaggle Dataset Link: </b>  https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data</i>

# ***
# #### Column Description for our Dataset
# ***
# <b><i><u>Target Variable/Dependent Variable</u><i></b>
#     
# 1. <b>TravelInsurance</b> - Did the customer buy travel insurance package during introductory offering held in the year 2019. This is the variable we have to predict
# 
# <b><i><u>Predictor Variables/Independent Variables </u><i></b>
# 1. <b>Unnamed: 0</b> - The row number
# 2. <b>Age</b> - Age of the customer
# 3. <b>Employment Type</b> - The sector in which customer is employed
# 4. <b>GraduateOrNot</b> - Whether the customer is college graduate or not
# 5. <b>AnnualIncome</b> - The yearly income of the customer in indian rupees[rounded to nearest 50 thousand rupees]
# 6. <b>FamilyMembers</b> - Number of members in customer's family
# 7. <b>ChronicDiseases</b> - Whether the customer suffers from any major disease or conditions like Diabetes/high BP or Asthama,etc
# 8. <b>FrequentFlyer</b> - Derived data based on customer's history of booking air tickets on atleast 4 different instances in the last 2 years[2017-2019]
# 9. <b>EverTravelledAbroad</b> - Has the customer ever travelled to a foreign country
# 
#     
# 

# In[214]:


#loading the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
import statsmodels.api as sd
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import random

import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

# importing shap
import shap


# In[215]:


#loading the file
df_train=pd.read_csv('TravelInsurancePrediction.csv')

#defining the response and the predictors
y_total=df_train['TravelInsurance']
data = df_train.drop(['Unnamed: 0'],axis=1)
df_train=df_train.drop(['TravelInsurance', 'Unnamed: 0'],axis=1)
df_train.head()


# In[216]:


#Checking if ChronicDiseases has 1 and 0 only(binary)
data['ChronicDiseases'].value_counts()
plt.figure(figsize=(10,5))
plt.barh(['No','Yes'],data['ChronicDiseases'].value_counts(), color ="#cc6600")
plt.title('Distribution for Chronic Diseases', size =14)
plt.xlabel("Count", size =14)
plt.ylabel("Chronic Disease", size =14)


# Observations:
# 1. Chronic Diseases has only binary data so we are assuming that '1' means that the travellers has a chronic Disease and '0' means the traveller is free of any chronic disease.

# In[217]:


#function for coverting binary to yes/no

def convert_binary_to_yesno(x):
    if x == 1:
        return "Yes"
    else:
        return "No"


# In[218]:


#function for coverting yes/no to binary

def convert_yesno_to_binary(x):
    if x == "Yes":
        return 1
    else:
        return 0


# In[219]:


#Converting ChronicDiseases to a Yes/No field for keeping it consistent with other Yes/No fields for our analysis  
data['ChronicDiseases'] = data['ChronicDiseases'].apply(convert_binary_to_yesno)


# In[220]:


#Reading first 5 rows after coversion
data.head(5)


# In[221]:


# Temporarily converting our dependent variable to a Yes/No field to get a clear categorical & numerical column list
data['TravelInsurance'] = data['TravelInsurance'].apply(convert_binary_to_yesno)


# <h2><i> What are the data types of our dataset variables?</i></h2>

# In[222]:


#Getting the list of categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()


# In[223]:


#Getting the list of numerical columns
numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()


# In[224]:


#Printing the list of categorical and numerical columns
print("--------------------------------------------------------")
print("                 Categorical Variables                  ")
print("--------------------------------------------------------")
print(f'Total number of categorical variables in our dataset: {len(categorical_cols)}')
for row,col in enumerate(categorical_cols):
    print(f'{row+1}. {col}')
print("\n")
print("--------------------------------------------------------")
print("                 Numerical Variables                  ")
print("--------------------------------------------------------")
print(f'Total number of num variables in our dataset: {len(numerical_cols)}')
for row,col in enumerate(numerical_cols):
    print(f'{row+1}. {col}')


# <h2><i>Are there missing values? Which independent variables have missing data? How much?</h2></i>

# In[225]:


#Checking missing values in our data
data.isnull().sum()


# <u><b><i>Observations</i><b><u>
# * We have 0% missing values both in our independent variables as well as dependent variable 

# In[226]:


# Splitting our dataset into two dataframes - 1. Predictor Variables 2. Target Variable
y = data['TravelInsurance']
x = data.drop('TravelInsurance', axis = 1)
data_total = data


# <h2><i>What are the likely distributions of the numeric variables?</h2></i>

# #### 1. Age

# In[227]:


plt.figure(figsize=(15,7))
sns.distplot(data['Age'], bins=10, color = "green")
print("The mean of Age is ",round(data['Age'].mean(),2))
print("The median of Age is ",data['Age'].median())
print("The Mode of Age is ",data['Age'].mode())
plt.xlabel("Age", size=14)
plt.ylabel("Density", size=14)
plt.title('Distribution curve for Age', size=20)


# Observations-
# 1. The mean and median are almost similar
# 2. The density of people with age 28 is the highest
# 3. The age range lies between 25 - 35 years

# #### 2. Annual Income

# In[228]:


plt.figure(figsize=(15,7))
sns.distplot(data['AnnualIncome'], color = "darkgreen")
print("The mean income is ",round(data['AnnualIncome'].mean(),2))
print("The median income is ",data['AnnualIncome'].median())
plt.title('Distribution curve for Annual Income')
plt.xlabel("Annual Income(in Millions)", size=14)
plt.ylabel("Density", size=14)
plt.title('Distribution curve for Annual Income', size=20)


# Observations:
# 1. As per the graph the income distribution follows modified normal distribution
# 2. There is a significant drop in the number of people from an income greater than 1.6M to 1.8M
# 3. Here also the mean and median are almost the same

# #### 3. Family Members

# In[229]:


plt.figure(figsize=(15,7))
sns.distplot(data['FamilyMembers'], color = "#00b3b3")
print("Average family members for a traveller is ",round(data['FamilyMembers'].mean(),2))
print("The median of family members is ",data['FamilyMembers'].median())
print("The mode of family members is ",data['FamilyMembers'].mode())
plt.xlabel("FamilyMembers", size=14)
plt.ylabel("Density", size=14)
plt.title('Distribution curve for FamilyMembers', size=20)


# Observations:
# 1. Average number of family members for the ones travelling is around 4
# 2. We have atleast 2 family members for any traveller with maximum of 9 members for 10% of the travellers
# 3. Travellers that have 8 or 9 family members are low in our dataset which might mean that travellers with big families don't travel that often

# #### 4. Employment Type

# In[230]:


plt.figure(figsize=(10,5))
plt.barh(['Private Sector/Self Employed','Government Sector'],data['Employment Type'].value_counts(), color = "#ffa64d")
plt.title('Distribution curve for type of employment', size=14)
plt.ylabel("Type of Employment", size=14)
plt.xlabel("No of Appearances", size=14)


# Insights from the distribution graph above:
# 
# 1. There are only two categories of employment type of travellers in our dataset - Private Sector and Government Sector 
# 2. No of private sector travellers are almost double of Government sector travellers

# #### 5. GraduateOrNot

# In[231]:


plt.figure(figsize=(10,5))
plt.barh(['Yes','No'],data['GraduateOrNot'].value_counts(), color="#33cccc")
plt.title('Distribution of travellers holding a graduate degree vs not holding a degree', size=14)
plt.ylabel("Graduate or Not", size=14)
plt.xlabel("No of Travellers", size=14)
print(data['GraduateOrNot'].value_counts())
print(295/1987*100)


# Observations:
# 1. 85% of the travellers in our dataset hold a graduate degree

# #### 6. FrequentFlyers

# In[232]:


plt.figure(figsize=(10,5))
plt.barh(['No','Yes'],data['FrequentFlyer'].value_counts())
plt.title('Distribution of frequent flyers', size=14)
plt.ylabel("Frequent flyer or not", size=14)
plt.xlabel("No of counts", size=14)
data['FrequentFlyer'].value_counts()


# Observations:
# * More than 60% of the travellers do not travel than often so we will have to check if they would even consider opting for a travel insurance

# #### 7. ChronicDiseases

# In[233]:


plt.figure(figsize=(10,5))
plt.barh(['No','Yes'],data['ChronicDiseases'].value_counts(), color ="#cc6600")
plt.title('Distribution for Chronic Diseases', size =14)
plt.xlabel("Count", size =14)
plt.ylabel("Chronic Disease", size =14)


# Observations:
# * Majority of the travellers do not have any chronic disease

# #### 8. EverTravelledAbroad

# In[234]:


plt.figure(figsize=(10,5))
plt.barh(['No','Yes'],data['EverTravelledAbroad'].value_counts(), color ="#660066")
plt.title('Distribution showing distribution of people who have travelled abroad or not', size =14)
plt.xlabel("Travelled abroad or not", size =14)
plt.ylabel("No of counts", size =14)
print(data['EverTravelledAbroad'].value_counts())


# Observations:
# 1. Only 20% of the travellers have a travel history travelling abroad
# ***

# Before we move further, we need to do some data cleaning for our analysis and also for building the model.
# 
# * We will covert 'Yes/No' to 1/0 - GraduateOrNot, FrequentFlyer, EverTravelledAbroad, ChronicDiseases,TravelInsurance
# 
# * For employment type we will follow the rule: Governemnt Sector -> 1, Private Sector -> 0

# In[235]:


# Coverting our independent categorical variables from Yes/No to Binary
x['GraduateOrNot'] = x['GraduateOrNot'].apply(convert_yesno_to_binary)
x['FrequentFlyer'] = x['FrequentFlyer'].apply(convert_yesno_to_binary)
x['EverTravelledAbroad'] = x['EverTravelledAbroad'].apply(convert_yesno_to_binary)
x['ChronicDiseases'] = x['ChronicDiseases'].apply(convert_yesno_to_binary)

# Converting the target variable from Yes/No to Binary
y = y.apply(convert_yesno_to_binary)


# In[236]:


#function for converting Employement type to binary

def convert_employmenttype_to_binary(employmenttype):
    if employmenttype == "Government Sector":
        return 1
    else:
        return 0


# In[237]:


# Converting EmployementType to Binary
x['Employment Type'] = x['Employment Type'].apply(convert_employmenttype_to_binary)


# In[238]:


# Reading our independent variables dataframe after data preparation
x.head()


# <h2><i>Do the range of the predictor variables make sense?</i></h2>

# In[239]:


#Checking the range of the predictor variables
plt.figure(figsize=(20,7))
sns.boxplot(data=x, palette="Set3")


# <b>Observations:</b>
# * The range of AnnualIncome is way too high as compared to other predictor variables. Therefore, we are not able to visualize the ranges for other predictor variables.
# 
# <b>So what should be do next to visualize ranges of other predictor variables?</b>
# * We will plot the charts separately to get the ranges of each one of them

# In[240]:


#Checking the Ranges of the predictor variables individually
cols = list(x.columns)
f, axs = plt.subplots(2,4,figsize=(20,10))
for i in range(len(cols)):
  plt.subplot(2,4,i+1)
  plt.title('Box plot of '+ cols[i])
  sns.boxplot(x[cols[i]], palette = 'Set3')
plt.show()


# Observations:
# * Majority of the travellers have an age greater than the median age of ~29 years
# * No of family above and below the median is almost same
# * The income group appears to be normally distributed as the 25-50 percentile and 50-75% have almost similar area
# * Number of travellers without a Graduate degree is low
# * Number of travellers who are frequent flyers or have travelled abroad is also low
# * There are no outliers in our dataset

# #### Normalizing the dataset
# * We need to scale our numerical columns. Although we can use any scaling technique, we will be using normalization as we want to have values in the range of [0,1] and also to detect outliers as normalization is highly affected my outliers
# 

# In[241]:


# list of numerical columns which require normalization
num_cols=['AnnualIncome','Age', 'FamilyMembers']

# Importing required library from sklearn for normalization
from sklearn import preprocessing
feature_to_scale = num_cols

# Preparing for normalizing
min_max_scaler = preprocessing.MinMaxScaler()

# Transform the data to fit minmax processor
x[feature_to_scale] = min_max_scaler.fit_transform(x[feature_to_scale])


# In[242]:


# Checking our predictor variables after normalization
x.describe()


# Observations:
# * We can see that the numerical variables - age, annual income and family members are now in the range of 0 to 1

# In[243]:


#Checking the Ranges of the predictor variables together after normalization of numerical variables
plt.figure(figsize=(20,7))
sns.boxplot(data=x, palette="Set3")
plt.title("Box plot of predictor variables of the dataset", size=14)


# <h2><i>Are the predictor variables independent of all other predictor variables?</i></h2>

# In[244]:


# Lets check the correlation among the predictor variables using a correlation matrix
x.corr()


# Observations:
# * At first glance we can see that that the variables have very less collinearity. To visualize the values lets check the heatmap next

# In[245]:


#the heat map of the correlation
plt.figure(figsize=(20,7))
sns.heatmap(x.corr(), annot=True, cmap='RdYlGn')


# Observations:
# * It is very clear from the heatmap that most of the variables are not dependent on each other
# * Degree of collinearity is significantly less that 0.1 for most variables
# * The Annual income and the international travel history of the travellers have a degree of collinearity of 0.49 which still is not a significant dependecy of variables on each other

# #### Creating a Train - Test split for our model training and predictions

# In[246]:


from sklearn.model_selection import  train_test_split

#Spliting data into Training 80% and Test set 20%

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# <h2><i>Do the training and test sets have the same data?</i></h2>

# In[247]:


# Looking the data for test and training set
X_test_plot = X_test
X_train_plot = X_train

#Plotting the data to see the histogram
for c in X_test_plot.columns[:]:
    plt.figure(figsize=(6,7))
    plt.hist(X_train_plot[c], bins=10, alpha=0.5, label="Train Set", color="darkgreen")
    plt.hist(X_test_plot[c], bins=10, alpha=0.5, label="Test Set", color ="red")
    plt.xlabel(c, size=14)
    plt.ylabel("Count", size=14)
    plt.legend(loc='upper right')
    plt.title("{} distribution".format(c))
    plt.show()


 


# Observations:
# * The ratio of 80%-20% for train-test split appears to be distributed correctly for all the variables except for Annual Income
# * For Annual Income, from 0.8 - 0.9, the count of test instances is more than the count of train instances. For the rest of the counts, test count is lower than the train count

# <h1><i>Feature Selection, Modeling and Interpretability</i></h1>
# <h4><i>Folllowing section will cover questions like -<br> 
#     <br>1. Which features are Important?<br>
#     <br>2. Which Models we are using? <br>
#     <br>3. Interpretability of our models? - Shap Analysis<br>
#     <br>4. How well did our models perform?</i></h4>

# In[248]:


#loading the file
df_train=pd.read_csv('TravelInsurancePrediction.csv')

#defining the response and the predictors
y_total=df_train['TravelInsurance']
df_train=df_train.drop(['TravelInsurance', 'Unnamed: 0'],axis=1)
num_cols=list(df_train._get_numeric_data().columns)
cat_cols=list(set(df_train.columns)-set(num_cols))
#encoding the categorical variables
mapping_dict={}
for col in cat_cols:
    mapping_dict[col]={k: i for i, k in enumerate(df_train[col].unique())}
    df_train[col] = df_train[col].map(mapping_dict[col])


# In[249]:


df_train.head()


# In[250]:


# Normalizing our data for proper analysis

# list of numerical columns which require normalization
num_cols=['AnnualIncome','Age', 'FamilyMembers']

# Importing required library from sklearn for normalization
from sklearn import preprocessing
feature_to_scale = num_cols

# Preparing for normalizing
min_max_scaler = preprocessing.MinMaxScaler()

# Transform the data to fit minmax processor
df_train[feature_to_scale] = min_max_scaler.fit_transform(df_train[feature_to_scale])

df_train.head()


# In[251]:


# Importing the metrics library from sklearn
from sklearn import metrics as sm


# ## <b>1. <u>Fitting a linear model and interpreting the coefficients</u></b>

# In[252]:


import statsmodels.api as sd
train_X, val_X, train_y, val_y = train_test_split(df_train, y_total, random_state=1)
log_reg = sd.Logit(train_y,train_X).fit()
print(log_reg.summary())


# In[253]:


### Create a Pickle file using serialization 
import pickle
pickle_out = open("log_reg.pkl","wb")
pickle.dump(log_reg, pickle_out)
pickle_out.close()


# ### Observations of Coefficients
# 
# 1. From the table we can clearly see that the p value of Age,Employment Type, Annual Income, Family Members,Frequent Flyer and EverTravelledAbroad are less than 5% and hence they are important features
# 2. Features such as Chronic Diseases and GraduateOrNot have p value more tha 5% and hence they are not important features
# 3. Though we can see that the p-values are different for all our features, we can't actually rank them at this point. What we can do next? - Run Shao Analysis

# ### <b>Shap values for Logistic regression</b>

# In[254]:


clf_model = LogisticRegression(random_state=0).fit(train_X, train_y)
explainer = shap.LinearExplainer(clf_model, train_X, feature_perturbation="interventional")
shap_values = explainer.shap_values(val_X)


# In[255]:


shap.summary_plot(shap_values, val_X, feature_names=list(train_X.columns))


# ### Observations from the Shap summmary
# 1. People who have high income,more family members, older and ever travelled abroad seem to more readily buy a travel insurance
# 2. People who fly less,have significantly less annual income, are younger seem to not buy a travel insurance. This makes sense because younger people usually don't earn much so they fly less and therefore might not buy Travel insurance

# ### Shap Analysis for a single point

# In[256]:


shap.initjs()
shap.plots.force(explainer.expected_value,shap_values[17],feature_names=list(train_X.columns))
#shap.force_plot(explainer.expected_value, shap_values, val_X.iloc[1], feature_names=list(train_X.columns))


# ### Observations
# - Annual Income is the most important feature for predicting the probability that a person will buy travel insurance or not
# - Employer Type, GraduateOrNot and FrquentFlyer seem to push the value away from base_value
# - AnnualIncome, Abroad Travel status and Age are the features which push the value towards the base value

# <h3><i>Evaluating our Logistic Regression Model</i></h3>

# In[257]:


# acc=sm.accuracy_score(val_y,log_reg.predict(val_X))
log_loss=sm.log_loss(val_y,log_reg.predict(val_X))
auc=sm.roc_auc_score(val_y,log_reg.predict(val_X))
# confusion_matrix=sm.confusion_matrix(val_y,log_reg.predict(val_X))
print("-------------------------------")
print(f'AUC: {auc:.2f}')
print(f'Log Loss: {log_loss:.2f}')
print("-------------------------------")


# ## <b> 2. <u>Fitting a tree based model and Interpreting the coefficients</u></b>

# In[258]:


my_model = DecisionTreeClassifier(random_state=42).fit(train_X, train_y)


# ### Feature importance of Decision Tree model

# In[259]:


plt.figure(figsize=(15,6))
importances = my_model.feature_importances_
features = list(train_X.columns)
plt.title("Decision Tree Model : Feature Importance")
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.barh(features,importances,color='g')


# ### Observations
# 1. It is clear from the above plot that Annual Income is the most importanct feature for predicting the decision of buying a travel insurance
# 2. Features such as Family Members and Age are also some other important features

# ## Plotting the Decision Tree

# In[260]:



fig = plt.figure(figsize=(20, 14))
vis = tree.plot_tree(my_model, feature_names = features, class_names = ['Insurance Taken', 'Insurance Not - Taken'], max_depth=4, fontsize=9, proportion=True, filled=True, rounded=True)


# ### Observations
# 1. It is clear from the decision tree chart is that Annnual Income is the first node which is causing splitting of samples
# 2. Roughly 83% of the samples have a relatively less salary as compared to a few wealthy ones
# 3. Rest of the nodes in subsequest levels are important as their appearance

# ### Shap Analysis for a single point

# In[261]:


row_to_show = 17
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)


# In[262]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,feature_names=list(val_X.columns))


# ### Observations
# 1. In predicting the class as 0 for the 17th datapoint the features that played an important role are AnnualIncome, Age, with no ChronicDiseases,graduates and frequent flyers
# 2. Annual Income variable is pushing the value towards the base value

# <h3><i>Evaluating our Decision Tree Classifier Model</i></h3>

# In[263]:


acc=sm.accuracy_score(val_y,my_model.predict(val_X))
log_loss=sm.log_loss(val_y,my_model.predict(val_X))
auc=sm.roc_auc_score(val_y,my_model.predict(val_X))
confusion_matrix=sm.confusion_matrix(val_y,my_model.predict(val_X))
print("------------------")
print(f'Accuracy: {acc:.2f}')
print(f'AUC: {auc:.2f}')
print(f'Log Loss: {log_loss:.2f}')
print(f'Confusion Matrix:\n {confusion_matrix}')
print("------------------")


# In[264]:


### Create a Pickle file using serialization 
import pickle
pickle_out = open("tress_model.pkl","wb")
pickle.dump(my_model, pickle_out)
pickle_out.close()


# ## 3. <u>Using an AutoML models and interpretability</u>

# In[265]:


#defining the environment variables of h2o
import psutil
import h2o
from h2o.automl import H2OAutoML
import random, os, sys
from datetime import datetime
import logging
import optparse
import time
import json
min_mem_size=6 
run_time=60
pct_memory=0.5
virtual_memory=psutil.virtual_memory()
min_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
print(min_mem_size)
port_no=random.randint(5555,55555)

#  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
try:
  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
except:
  logging.critical('h2o.init')
  h2o.download_all_logs(dirname=logs_path, filename=logfile)      
  h2o.cluster().shutdown()
  sys.exit(2)


# In[266]:


#function for coverting binary to yes/no

def convert_binary_to_yesno(x):
    if x == 1:
        return "Yes"
    else:
        return "No"


# In[267]:


df_h = h2o.import_file('TravelInsurancePrediction.csv')
df_h = df_h.drop(['C1'],axis=1)

df_h.head()


# In[268]:


pct_rows=0.80
df_train, df_test = df_h.split_frame([pct_rows])
print('Train dataframe size')
print(df_train.shape)
print('Test dataframe size')
print(df_test.shape)
#defining the predictor and response variable for our model
X=df_h.columns
y ='TravelInsurance'
# df_h['TravelInsurance'] = df_h['TravelInsurance'].apply(convert_binary_to_yesno) 
X.remove(y)
print('The predictor variables are as follows')
print(X)
print('The response variable is')
print(y)


# In[269]:


aml = H2OAutoML(max_runtime_secs=run_time, seed=1)
aml.train(x=X,y=y,training_frame=df_train) 
print('Training Successful....')


# In[270]:


aml.explain(df_test,include_explanations=['varimp','shap_summary'])


# ### Observations
# 1. Variable Importance plot - As we have seen in earlier two models , in autoML model also we have AnnualIncome as the most importnat feature
# 2. Variable Importance plot - The next important features are Family Members, Age, Abroad Travel status and so on
# 3. SHAP Summary - In Shap summary also we can see that the features such as Annual Income, Age and FamilyMembers to be some of the most important features for predicting the travel insurance purchase decision for a person
# 4. After seeing shap analysis for all three of our models, we can now very clearly interpreate what features but weight in predicting our model decision which is ultimately the goal of this notebook

# In[271]:


import pickle
pickle_out = open("automl_model.pkl","wb")
pickle.dump(aml, pickle_out)
pickle_out.close()


# ***
# ## Citations

# 1.Many techniques used in this notebook have been adopted from the following github repositories
# 
# * Owner - AI Skunkworks
# * Link - https://github.com/aiskunks/Skunks_Skool
# <br></br>
# * Author name - Prof Nik Bear Brown
# * Link - https://github.com/nikbearbrown/
# 
# 2.The methods and parameters of the models amd code corrections have been adapted from stackoverflow
# 
# * Link - https://stackoverflow.com
# 
# 3.Reference has been taken from the seaborn webpage for charts and visualization
# * Link - https://seaborn.pydata.org
# 
# 4.The methods and parameters of the GLM model have been adapted from the h2o documentation</br>
# * Author - **H2O.ai**
# * Link - https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html</br>

# ***
# ## Licensing

# Copyright 2022 Ankit Goyal
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
