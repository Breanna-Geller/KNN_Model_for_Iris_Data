# Classification Pipeline with KNN

Responsibilities for are located in the attached PDF. My goal is to achieve higher than 75% accuracy with my model implementation!

## Step 1: Download the Iris Dataset
Downloaded the Iris Dataset from UC Irvine: [Iris](https://archive.ics.uci.edu/dataset/53/iris)

## Step 2: Data Processing

- **Feature Selection**: The Iris dataset has four features and one label. 
    - *Feature*: Sepal Length (continuous)
    - *Feature*: Sepal Width (continuous)
    - *Feature*: Petal Length (continuous)
    - *Feature*: Petal Width (continuous)
    - *Label*: class (independent variable)

There is no need to normalize the data, as the values range from ~6cm-10cm in all features. There are also no missing values, as verified by the dataset. 

```
           name     role         type demographic                                        description units missing_values
0  sepal length  Feature   Continuous        None                                               None    cm             no
1   sepal width  Feature   Continuous        None                                               None    cm             no
2  petal length  Feature   Continuous        None                                               None    cm             no
3   petal width  Feature   Continuous        None                                               None    cm             no
4         class   Target  Categorical        None  class of iris plant: Iris Setosa, Iris Versico...  None             no
```