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
from collections import Counter # allows me to keep track of the most common element in a list - useful for KNN labels

# Function to pick color based on class
def ColorPick(y):
    print(y)
    if y == 'Iris-setosa':
        return 'red'
    elif y == 'Iris-versicolor':
        return 'green'
    else:
        return 'blue'

###########
### KNN ###         # HOW TO DEAL WITH A TIE???? ask professor
########### 
# Function to calculate Euclidean Distance between one point in test to training data
def KNN(X_train, y_train, X_test, y_test, k):
    # print(len(X_test)) # X_test passes in 8 points for each type of iris = 24
    # print(len(X_train))# X_train passes in 32 points for each type of iris = 96
    nearest_neighbors = []
    predicted_labels = []
    # Run through test data one at a time
    for i in range(0, len(X_test)):
        distances = []
        for j in range(0, len(X_train)):
            # Keep track of Euclidean Distances and labels from test to each training data point
            distances.append([np.linalg.norm(X_test[i]-X_train[j]), y_train[j]])
        distances.sort()
        nearest_neighbors.append(distances[0:k]) # The sorted k-nearest neighbors (0-5) for this test point
    # Traverse through all neighbors and find the most common label
    for i in range(0, len(nearest_neighbors)):
        # print(nearest_neighbors[i])
        counter = Counter([x[1] for x in nearest_neighbors[i]]) # Need to access the label from list
        #print(counter)
        # Find the predicted label (most common)
        counter.most_common()
        predicted_labels.append(counter.most_common(1)[0][0])
    # print(predicted_labels)
    return predicted_labels

    
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
X_train, X_test_save, y_train, y_test_save = train_test_split(X_pca, y, test_size=0.2, stratify=y)
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
#print(type(Fold_5_Y)) # counting the number of each class in the valid -> GOOD!!! (8 each) - checked type too

##########################################
### KNN Train and Test Chamber (Loops) ###
##########################################

# Maintain sets for each fold in an array
Test_Set_Selection = [Fold_1_X, Fold_2_X, Fold_3_X, Fold_4_X, Fold_5_X]
Test_Set_Labels = [Fold_1_Y, Fold_2_Y, Fold_3_Y, Fold_4_Y, Fold_5_Y]



for i in range(0, len(Test_Set_Selection)):
    predicted_labels = []
    # Selection of testing for fold == i
    X_test = Test_Set_Selection[i]
    y_test = Test_Set_Labels[i]
    # Train on all folds except the one we are testing
    X_train = Test_Set_Selection.copy()
    X_train.pop(i)
    y_train = Test_Set_Labels.copy()
    y_train.pop(i)
    y_train = np.concatenate(y_train)
    X_train = np.concatenate(X_train)
    #print(y_train)
    #print(X_train)
    #print(y_test)
    #print(Fold_1_Y)
    #print(X_test)
    #print(Fold_1_X)

    # Call KNN function 
    predicted_labels = KNN(X_train, y_train, X_test, y_test, 5)
    # print(predicted_labels)

    # Calculate Accuracy for this K-Fold Partition
    accurate_predictions = 0
    for j in range(0, len(predicted_labels)):
        if predicted_labels[j] == y_test.iloc[j]:
            #print(predicted_labels[i], y_test.iloc[i])
            accurate_predictions += 1

    print(f"Accuracy for K-Fold {i+1}: ", accurate_predictions/len(predicted_labels))
    
###############################################################
### Completely New Never-Before Seen Set-Aside Data Testing ###
###############################################################
# Use the set aside OG test data to test the model
predicted_labels_no_cheat = KNN(X_train, y_train, X_test_save, y_test_save, 5)
no_cheat_acc = 0
for i in range(0, len(predicted_labels_no_cheat)):
    if predicted_labels_no_cheat[i] == y_test_save.iloc[i]:
        no_cheat_acc += 1

print(f"Accuracy for the Never-Before Seen Test Data: ", no_cheat_acc/len(predicted_labels_no_cheat))