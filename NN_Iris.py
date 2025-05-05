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
import torch
import torch.nn as nn
import optuna
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from optuna.visualization import plot_param_importances
from sklearn.metrics import f1_score
import time


# Fetch the Iris dataset from UCI ML Repository
iris = fetch_ucirepo(id=53)

# If CUDA is available, set the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import Iris data (as pandas dataframes) 
X = iris.data.features # Features
y = iris.data.targets  # Labels
print(y)
print(X)

# Check missing values
# print(X.isnull().sum()) # Check for missing values in features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

x_train=torch.FloatTensor(X_train.values)
x_test=torch.FloatTensor(X_test.values)
# Encode the categorical labels into numerical values
label_encoder = LabelEncoder()
y_train = torch.tensor(label_encoder.fit_transform(y_train.values))
y_test = torch.tensor(label_encoder.transform(y_test.values))

start = time.time()
print("Start time: ", start)
# Feedforward NN 
class Iris_NN_Model(nn.Module):
    def __init__(self,input_feature,hidden_layer,output,neurons_per_layer,activation_func,dropout_rate):
        super(Iris_NN_Model, self).__init__()
        layers=[]
        for i in range(hidden_layer) :
            layers.append(nn.Linear(input_feature,neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(activation_func)
            layers.append(nn.Dropout(dropout_rate))
            input_feature=neurons_per_layer
        layers.append(nn.Linear(neurons_per_layer,output))
        self.model=nn.Sequential(*layers)# unpack the list because sequential 
   
    def forward(self,x):
        return self.model(x)
    
#Hyperparameter Tuning - create the objective function for Optuna
# Global activation function mapping
activation_funct = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "Sigmoid": nn.Sigmoid()
}

def objective(trial):
# Select hyperparameter values from suggested numbers
    num_hidden_layer=trial.suggest_int("num_hidden_layers",1,4)
    neurons_per_layer=trial.suggest_int('neurons_per_layer',16,128,step=8)
    epochs=trial.suggest_int("epoch",250,1000,step=50)
    learning_rate=trial.suggest_float("learning_rate",1e-6,5e-1,log=True)  #means alogrithm scale 
    # batch_size=trial.suggest_categorical("batch_size",[16,32,64,128])
    optimizer_name=trial.suggest_categorical("optimizer",['Adam','SGD','RMSprop','Adagrad'])
    weight_decay=trial.suggest_float("weight_decay",1e-5,1e-3,log=True)
    dropout_rate=trial.suggest_float("Dropout_rate",0.1,0.5,step=0.1)

    activation_name = trial.suggest_categorical("activation_function", ["ReLU", "LeakyReLU", "ELU"])
    
# Activation Functions
    activation_func = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "ELU": nn.ELU(),
        "Sigmoid": nn.Sigmoid()
    }[activation_name]
    
# Model init
    input_dim = 4 #input 4 features 
    output_dim = 3 # Output 1 of 3 numbers
    model=Iris_NN_Model(input_dim,num_hidden_layer,output_dim,neurons_per_layer,activation_func,dropout_rate)
    model.to(device)

# Optimizer selection
    loss_func=nn.CrossEntropyLoss()
    if optimizer_name=='Adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay= weight_decay)  
    elif optimizer_name=='SGD':
        optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay= weight_decay)  
    else:
        optimizer=torch.optim.RMSprop(model.parameters(),lr=learning_rate,weight_decay= weight_decay)  
# Training
    for i in range(epochs): 
        y_pred=model(x_train.to(device))
        loss=loss_func(y_pred,y_train.to(device))
        loss.backward()
        optimizer.step()

# Evaluation
    model.eval()
    predictions=[]
    for i,data in enumerate(x_test.to(device)):
        with torch.no_grad():
            y_pred=model(data.reshape(1, -1))
            predictions.append(y_pred.argmax().item())
    #score=accuracy_score(y_test,predictions)
    score = f1_score(y_test, predictions, average='weighted')

    return score

study=optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=100) # 100 trials test
# print(study.best_value)
# print(study.best_params)

# Create static Neural Network model with best hyperparameters -- F1 score
# input_dim,num_hidden_layer,output_dim,neurons_per_layer,activation_func,dropout_rate)
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()
end_time = time.time()
print("End time: ", end_time)
print("Total time taken: ", end_time - start)
start=time.time()
best_params = study.best_params
best_model = Iris_NN_Model(
    input_feature=4,
    hidden_layer=best_params['num_hidden_layers'],
    output=3,
    neurons_per_layer=best_params['neurons_per_layer'],
    activation_func=activation_funct[best_params['activation_function']],
    dropout_rate=best_params['Dropout_rate']
)
best_model.to(device)

# Optimizer with the best learning rate and weight decay
optimizer = torch.optim.RMSprop(
    best_model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)

# Training loop
loss_func = nn.CrossEntropyLoss()
for epoch in range(best_params['epoch']):
    y_pred = best_model(x_train.to(device))
    loss = loss_func(y_pred, y_train.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

best_model.eval()
predictions = []
with torch.no_grad():
    for data in x_test.to(device):
        y_pred = best_model(data.reshape(1, -1))
        predictions.append(y_pred.argmax().item())

# Calculate the F1 score or other metrics
score = f1_score(y_test, predictions, average='weighted')
end_time = time.time()
print("Total time taken for best model: ", end_time - start)
accuracy_score = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy_score}")
print(f"F1 Score: {score}")
