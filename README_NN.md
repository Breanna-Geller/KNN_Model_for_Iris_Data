# Neural Network for the Iris Dataset
## 1. Data Preprocessing
*Apply the same preprocessing techniques used in previous assignments to prepare your data for training. This may include normalization, handling missing values, or feature selection.*

In Homework 2 (KNN) I did not normalize the dataset as the values range from ~6cm-10cm in all features. 

There are also no missing values in the dataset.
```python
           name     role         type demographic                                        description units missing_values
0  sepal length  Feature   Continuous        None                                               None    cm             no
1   sepal width  Feature   Continuous        None                                               None    cm             no
2  petal length  Feature   Continuous        None                                               None    cm             no
3   petal width  Feature   Continuous        None                                               None    cm             no
4         class   Target  Categorical        None  class of iris plant: Iris Setosa, Iris Versico...  None             no
```
I do not want to perform PCA on the dataset, as I believe the NN will be able to handle all features in training and I don't just want the principal components. I used ```LabelEncoder()``` to change labels into corresponding integer representations. Ex: Iris-versicolor = 1.

## 2. Model Training
- *Construct a neural network architecture suitable for your dataset and problem type (e.g.,classification or regression).*
- *Experiment with different hyperparameters to optimize model performance, including the learning rate, batch size, number of epochs, and layer structure.*

### Optuna
As any libraries are available to use, I am going to train my hyperparameters with Optuna- an open source Python Library that tries to study and optimize for the best parameters over 100 trials. I will describe the objective function it is trying to optimize below. You may need to install the library using: ```pip install optuna``` 

First, I split the testing and training data so there is no cheating.

#### Model for Optuna
The model for hyperparameter tuning needs to be dynamic. This means The number of layers, dropout rate, neurons, etc. must not be hardcoded but set to certain variables/ranges/functions (which will be listed below). 

```python 
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
```

#### Objective Function
```trial.suggest_*``` suggests ranges of values for various hyperparameters. I arbitrarily select a different number of epochs, hidden layers, etc. for the tuning to go through. The model is then initialized with the hyperparameters selected and will run on GPU if available or CPU. 

The model is trained for the specified number of epochs using the cross-entropy loss function (nn.CrossEntropyLoss).
Gradients are computed via backpropagation (loss.backward()), and the optimizer updates the model's parameters (optimizer.step()). 

The model is set to take in the 4 features (sepal length, sepal width, petal length, petal width) and output to 3 possible outputs (Iris-Verginica, setosa and versicolor). 

```python
# Model init
    input_dim=4 #input 4 features 
    output_dim=3 # Output 1 of 3 numbers
```

After training, the model is evaluated on the test dataset (x_test) in evaluation mode (model.eval()).
Predictions are generated, and the accuracy score is calculated using accuracy_score.

First, my Optuna study was created to maximize the accuracy score.
The study.optimize method runs the optimization process for 100 trials, testing different combinations of hyperparameters.


## 3. Testing and Model Evaluation
*Choose an evaluation metric appropriate for your task:*
- *For classification: accuracy, precision, recall, f1-score, etc.*
- *For regression: RMSE, MAE, R-squared, etc.*

*Report the evaluation results and analyze the model’s performance.*

The following were used to evaluate and analyze performance of the model.
- ```accuracy_score```: "Computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true." [Source](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

- ```f1_score```: "The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal." [Source](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

The best score and corresponding hyperparameters:
```python
# Trial 1
[I 2025-05-02 17:07:23,248] Trial 0 finished with value: 1.0 and parameters: {'num_hidden_layers': 2, 'neurons_per_layer': 104, 'epoch': 450, 'learning_rate': 0.027312545062450254, 'optimizer': 'Adam', 'weight_decay': 8.490989936176525e-05, 'Dropout_rate': 0.2, 'activation_function': 'ReLU'}. Best is trial 0 with value: 1.0.
# Trial 2
[I 2025-05-02 17:13:11,741] Trial 2 finished with value: 1.0 and parameters: {'num_hidden_layers': 2, 'neurons_per_layer': 128, 'epoch': 950, 'learning_rate': 0.011559860972206445, 'optimizer': 'SGD', 'weight_decay': 7.004647057267131e-05, 'Dropout_rate': 0.2, 'activation_function': 'ReLU'}. Best is trial 0 with value: 1.0.
# Trial 3
[I 2025-05-02 17:13:13,964] Trial 3 finished with value: 0.9333333333333333 and parameters: {'num_hidden_layers': 4, 'neurons_per_layer': 80, 'epoch': 700, 'learning_rate': 1.7124362619118073e-05, 'optimizer': 'Adam', 'weight_decay': 1.4415973442891097e-05, 'Dropout_rate': 0.4, 'activation_function': 'ELU'}. Best is trial 0 with value: 1.0.
#
# 
# 
# Best Accuracy
1.0
# Best Trial Hyper Params
{'num_hidden_layers': 2, 'neurons_per_layer': 104, 'epoch': 450, 'learning_rate': 0.027312545062450254, 'optimizer': 'Adam', 'weight_decay': 8.490989936176525e-05, 'Dropout_rate': 0.2, 'activation_function': 'ReLU'}
```

Then, I tried to optimize with respect to f1-score:
```python
# Trial 1
[I 2025-05-02 17:16:00,691] Trial 0 finished with value: 1.0 and parameters: {'num_hidden_layers': 3, 'neurons_per_layer': 120, 'epoch': 950, 'learning_rate': 0.004948614912716559, 'optimizer': 'RMSprop', 'weight_decay': 0.0005145438738961834, 'Dropout_rate': 0.4, 'activation_function': 'LeakyReLU'}. Best is trial 0 with value: 1.0.
# Trial 2
[I 2025-05-02 17:16:01,681] Trial 1 finished with value: 1.0 and parameters: {'num_hidden_layers': 2, 'neurons_per_layer': 112, 'epoch': 750, 'learning_rate': 0.07847942848167094, 'optimizer': 'SGD', 'weight_decay': 1.9684521479685785e-05, 'Dropout_rate': 0.5, 'activation_function': 'ReLU'}. Best is trial 0 with value: 1.0.
# Trial 3
[I 2025-05-02 17:16:01,908] Trial 2 finished with value: 0.15343915343915343 and parameters: {'num_hidden_layers': 1, 'neurons_per_layer': 64, 'epoch': 250, 'learning_rate': 1.0863555026413742e-06, 'optimizer': 'Adagrad', 'weight_decay': 0.0007059723296937521, 'Dropout_rate': 0.2, 'activation_function': 'ReLU'}. Best is trial 0 with value: 1.0.
#
#
#
# Best F1 Score
1.0
# Best Hyperparameters
{'num_hidden_layers': 3, 'neurons_per_layer': 120, 'epoch': 950, 'learning_rate': 0.004948614912716559, 'optimizer': 'RMSprop', 'weight_decay': 0.0005145438738961834, 'Dropout_rate': 0.4, 'activation_function': 'LeakyReLU'}
```

From here, I created a new NN models where I set the hyperparameters to those selected by Optuna for the f1-score and accuracy.

```python
#Output to terminal
Accuracy: 1.0
F1 Score: 1.0
```

## Report
**Architecture Details: Provide a detailed description of your NN architecture, including the type and number of layers, number of nodes per layer, activation functions, loss function, and optimizer used.**
**Comparative Analysis: Compare the performance and efficiency of your neural network with a previous model (e.g., a logistic regression or decision tree) used in HW2 or HW3. Discuss differences in results, model complexity, and computational requirements.**
**Results and Visualization: Include code snippets, execution results, and screenshots to clearly demonstrate the process and final outputs of your NN model.**
**Insights and Learning: Reflect on what you learned through this assignment, including any new insights into neural networks, challenges faced in data preprocessing, model training, or parameter tuning, and potential improvements or different approaches you would consider.**
** Additional Observations: Share any other observations or interesting findings, such as unexpected results or patterns in the data.**
There are no restrictions on using external libraries, but make sure to document the libraries you used and their purposes in the report.


### Model Performance for HW2 (KNN):
*Achieve a better performance than the model implemented for HW2 or HW3.*

