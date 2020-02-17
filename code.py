# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Loa-d the train data stored in path variable
train_data = pd.read_csv(path)
print('The shape of the train data is', train_data.shape)
print(train_data.tail(5))

# Load the test data stored in path1 variable
test_data = pd.read_csv(path1)
print('The shape of the test data is', test_data.shape)
print(test_data.tail(5))

# necessary to remove rows with incorrect labels in test dataset
print(test_data.isnull().sum())
test_data.dropna(axis=0, inplace = True)
print(test_data.shape)

# encode target variable as integer
le = LabelEncoder()
train_data['Target'] = le.fit_transform(train_data['Target'])
test_data['Target'] = le.fit_transform(test_data['Target'])
print(train_data['Target'].value_counts())

# Plot the distribution of each feature
print(train_data.info())
numeric_features = ['Age','fnlwgt','Education_Num','Capital_Gain','Capital_Loss','Hours_per_week']
categorical_features = ['Workclass','Education','Martial_Status','Occupation','Relationship','Race','Sex','Country']

for num in numeric_features:
    plt.figure(figsize=(10,10))
    plt.hist(train_data[num], bins=20)
    plt.show()

for cat in categorical_features:
    plt.figure(figsize=(10,10))
    vc = train_data[cat].value_counts()
    vc.plot(kind='bar')
    plt.show()

# Question : In which country the data is more concentrated and which Race of people are most in that country?
train_data[train_data['Country'] == 'United-States'].Race.value_counts()

# convert the data type of Age column in the test data to int type
test_data['Age'] = test_data['Age'].astype(int)
print(test_data.info())

# cast all float features to int type to keep types consistent between our train and test data


# choose categorical and continuous features from data and print them


# fill missing data for catgorical columns


# fill missing data for numerical columns   

# Dummy code Categoricol features


# Check for Column which is not present in test data


# New Zero valued feature in test data for Holand


# Split train and test data into X_train ,y_train,X_test and y_test data


# train a decision tree model then predict our test data and compute the accuracy


# Decision tree with parameter tuning


# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_


#train a decision tree model with best parameter then predict our test data and compute the accuracy




