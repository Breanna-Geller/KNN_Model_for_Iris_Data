# Name: Breanna Geller
# Date: 02/19/2024
# KNN HW 2 CS 622
# NSHE ID: 5006089018
from ucimlrepo import fetch_ucirepo 
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function to pick color based on class
def ColorPick(y):
    print(y)
    if y == 'Iris-setosa':
        return 'red'
    elif y == 'Iris-versicolor':
        return 'green'
    else:
        return 'blue'
    
# Function to calculate Euclidean Distance between two points in training and testing data
def KNN(X_train, y_train, X_test, y_test, k):
    return 0
    
#################################
### Fetching the Iris Dataset ###
#################################
iris = fetch_ucirepo(id=53) # I'M TIRED OF ERR_CONNECTION GRANDPA
# iris = pd.read_csv('iris/iris.data', header=None)
  
# Import Iris data (as pandas dataframes) 
X = iris.data.features # Features
y = iris.data.targets  # Labels



# Print Iris metadata 
# print(iris.metadata) 
  
# Print Iris Variable information 
# print(iris.variables) 

######################################
### Preprocessing the Iris Dataset ###
######################################

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
y=y.iloc[:,0]


# Plot to visualize the data from PCA
#for i in range(0, len(y)):
#    plt.text(X_pca[i,0], X_pca[i,1], y[i], color='black')
#    plt.scatter(X_pca[i,0], X_pca[i,1], color=ColorPick(y[i]))
#plt.show()
#print(len(y))



##################################################
### Split the Iris Dataset into Train and Test ###
##################################################
# By removing this testing set we can make sure we are not cheating
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y)
# print(y_train)
# print(y_test.value_counts()) # counting the number of each class in the test -> GOOD!!! (10 each)
# print(y_train.value_counts()) # counting the number of each class in the train -> GOOD!!!(40 each)


################################################
### 5-Fold Cross Validation of Training Data ###
################################################
# Split the training data into 5 folds (maintain stratification for proportionality)
X_train, Fold_1_X, y_train, Fold_1_Y = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
# print(Fold_1_Y.value_counts()) # counting the number of each class in the valid -> GOOD!!! (8 each) (left = 32)
X_train, Fold_2_X, y_train, Fold_2_Y = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)
# print(Fold_2_Y.value_counts()) # counting the number of each class in the valid -> GOOD!!! (8 each) (left = 24)
X_train, Fold_3_X, y_train, Fold_3_Y = train_test_split(X_train, y_train, test_size=0.333, stratify=y_train)
# print(Fold_3_Y.value_counts()) # counting the number of each class in the valid -> GOOD!!! (8 each) (left = 16)
X_train, Fold_4_X, y_train, Fold_4_Y = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train)
# print(Fold_4_Y.value_counts()) # counting the number of each class in the valid -> GOOD!!! (8 each) (left = 8)
Fold_5_X = X_train
Fold_5_Y = y_train
# print(Fold_5_Y.value_counts()) # counting the number of each class in the valid -> GOOD!!! (8 each)

##########################################
### KNN Train and Test Chamber (Loops) ###
##########################################

# Maintain sets for each fold in an array
Test_Set_Selection = [Fold_1_X, Fold_2_X, Fold_3_X, Fold_4_X, Fold_5_X]
Test_Set_Labels = [Fold_1_Y, Fold_2_Y, Fold_3_Y, Fold_4_Y, Fold_5_Y]

for i in range(0, 1):#len(Test_Set_Selection)):
    # Selection of testing for fold == i
    X_test = Test_Set_Selection[i]
    y_test = Test_Set_Labels[i]
    # Train on all folds except the one we are testing
    X_train = Test_Set_Selection.copy()
    X_train.pop(i)
    y_train = Test_Set_Labels.copy()
    y_train.pop(i)
    y_train = np.concatenate(y_train)
    print(y_train.value_counts())

    # Call KNN function
    KNN(X_train, y_train, X_test, y_test, 5)