# Name: Breanna Geller
# Date: 05/2/2025
# KNN HW 4 CS 622
# NSHE ID: 5006089018
from ucimlrepo import fetch_ucirepo 
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter # allows me to keep track of the most common element in a list - useful for KNN labels

# Fetch the Iris dataset from UCI ML Repository
iris = fetch_ucirepo(id=53)
# iris = pd.read_csv('iris/iris.data', header=None)
  
# Import Iris data (as pandas dataframes) 
X = iris.data.features # Features
y = iris.data.targets  # Labels

#Check missing values
print(X.isnull().sum()) # Check for missing values in features