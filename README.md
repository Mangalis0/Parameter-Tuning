# Hyperparameter Tuning

**Completed by Mangaliso Makhoba.**

**Overview:** This project is using the UCI Wine Quality Dataset to create a model that will predict the wine quality based on physicochemical tests, after tuning hyperparameters.

**Problem Statement:** Evaluate Support Vector Classifier (SVC) model perfomance after finding the best hyperparamers of 'C' and 'gamma'.

**Data:** [Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Deliverables:** Best hyperameters.

## Topics Covered

1. Machine Learning
3. Support Vector Classification
4. Hyperparameter Tuning
5. Log Loss Function 
6. GridSearchCV

## Tools Used
1. Ppython
1. Scikit-learn
2. Jupyter Notebook

## Installation and Usage

Ensure that the following packages have been installed and imported.

```bash
pip install numpy
pip install pandas
pip install sklearn
```

#### Jupyter Notebook - to run ipython notebook (.ipynb) project file
Follow instruction on https://docs.anaconda.com/anaconda/install/ to install Anaconda with Jupyter. 

Alternatively:
VS Code can render Jupyter Notebooks

## Notebook Structure
The structure of this notebook is as follows:

 - First, we'll load our data to get a view of the predictor and response variables we will be modeling. 
 - We'll then preprocess our data, binarising the target variable and splitting up the data intro train and test sets. 
 - We then model our data using a Support Vector Classifier.
 - Following this modeling, we define a custom metric as the log-loss in order to evaluate our produced model.
 - Using this metric, we then take several steps to improve our base model's performance by optimising the hyperparameters of the SVC through a grid search strategy. 



# Function 1: Data Preprocessing
We would like to classify the wine according to it's quality using binary classification.
We write a function to preprocess the data so we can run it through the classifier. The function should:

* Convert the quality for lower quality wines (quality less than or equal to 5) to 0
* Convert the quality for higher quality wines (quality greater than or equal to 6) to 1
* Split the data into 75% training and 25% testing data
* Set random_state to equal 42 for this internal method. 

_**Function Specifications:**_
* Function Name: data_splitting
* Should take a dataframe
* Standardise the features using sklearn's ```StandardScaler```
* Convert the quality labels into a binary labels
* Should fill nan values with zeros
* Should return two `tuples` of the form `(X_train, y_train), (X_test, y_test)`.

_**Expected Outputs:**_
```python
(X_train, X_test,y_train, y_test)=data_preprocess(df)
print(X_train[:2])
print(y_train[:2])
print(X_test[:2])
print(y_test[:2])


[[-0.57136659  0.07127869 -0.48054096  1.17914161 -0.09303318 -0.79974133
   0.0830898  -0.15472329 -0.36573452  0.13010447  0.06101473  0.25842195]
 [-0.57136659  1.50396711 -0.72301571  0.56008035 -0.63948302 -0.05776881
  -0.70572997  0.62379657  0.16787589 -0.86828773 -0.47467813 -0.99931317]]

[1 0]

[[-0.57136659 -0.15493527 -0.54115965  0.90400327 -0.66050032 -0.31460545
   0.53384396  0.03990667 -1.35291379 -0.26925241 -0.34075491  1.18076103]
 [-0.57136659  0.29749266 -1.20796522  2.8987562  -0.80762143 -0.45729248
  -0.19863155 -0.22549783 -1.03274754 -0.7185289  -0.87644778  0.25842195]]

[1 1]  
``` 



# Function 2: Model Training

Now that we have processed the data, let's jump straight into model fitting. We write a function that should:
* Instantiate a `SVC` model.
* Train the `SVC` model with default parameters.
* Return the trained SVC model. 

_**Function Specifications:**_
* Function Name: train_SVC_model
* Should take two numpy `arrays` as input in the form `(X_train, y_train)`.
* Should return an sklearn `SVC` model which has a random state of 40 and gamma set to 'auto'.
* The returned model should be fitted to the data.

_**Expected Outputs:**_

```python
svc = train_SVC_model(X_train,y_train)
svc.classes_
```
```python
array([0, 1], dtype=int64)
```


# Function 3: Model Testing
Now that we've've trained our model. It's time to test its accuracy, however, we'll be using a custom scoring function for this. Create a function that implements the log loss function:

<img src="https://render.githubusercontent.com/render/math?math=H(p,q)= - \frac{1}{N}\sum_{i=1}^{N} -ylog(\hat{y}_{i}) - (1- y)log(1 - \hat{y}_{i})">

_**Function Specifications:**_
* Should take two numpy `arrays` as input in the form `y_true` and `y_predicted`.
* Should return a `float64` for the log loss value rounded to 7 decimal places.

_**Expected Outputs:**_
```python
print('Log Loss value: ',custom_scoring_function(y_test,y_pred))
print('Accuracy: ',accuracy_score(y_test,y_pred))
```

```python
Log Loss value:  1.2540518
Accuracy:  0.9637
```


# Function 4: Getting model parameters

In order to improve the accuracy of our classifier, we have to search for the best possible model (`SVC` in this case) parameters. However, we first have to find out what parameters can be tuned for the given model. Write a function that returns a list of available hyperparameters for a given model. 

_**Function Specifications:**_
* Should take in an sklearn model (estimator) object.
* Should return a list of parameters for the given model.

_**Expected Outputs:**_

```python
get_model_hyperparams(SVC)
```

```
['C',
 'break_ties',
 'cache_size',
 'class_weight',
 'coef0',
 'decision_function_shape',
 'degree',
 'gamma',
 'kernel',
 'max_iter',
 'probability',
 'random_state',
 'shrinking',
 'tol',
 'verbose']
```


# Function 5: Hyperparameter Search

The next step is define a set of `SVC` hyperparameters to search over. Write a function that searches for optimal parameters using the given dictionary of hyperparameters:

- C_list = [0.1, 1, 10]
- {C: 0.1, 1, 10}
- gamma_list = [0.01, 0.1, 1]
- {gamma: 0.01, 0.1, 1}
- D = {'C':[0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}

and using `custom_scoring_function` from **Question 3** above as a custom scoring function (_**Hint**_: Have a look at at the `make_scorer` object in sklearn `metrics`).

_**Function Specifications:**_
* Should define a parameter grid using the given list of `SVC` hyperparameters
* Should return an sklearn `GridSearchCV` object with a cross validation of 5.

_**Expected Outputs:**_
```python
print('Log Loss value: ',custom_scoring_function(y_test,y_pred))
print('Accuracy: ',accuracy_score(y_test,y_pred))
```

``` python
Log Loss value:  1.2115421
Accuracy:  0.9649
```


# Function 6: Optimal model parameters

Write a function that returns the best hyperperameters for a given model (i.e. the `GridSearchCV`). 

_**Function Specifications:**_
* Should take in an sklearn GridSearchCV object.
* Should return a dictionary of optimal parameters for the given model.

_**Expected Outputs:**_
```python
get_best_params(svc_tuned)
```

``` python
{'C': 1, 'gamma': 1}
```


## Contributing Authors
**Authors:** Mangaliso Makhoba, Explore Data Science Academy

**Contact:** makhoba808@gmail.com

## Project Continuity
This is project is complete

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## License
[MIT](https://choosealicense.com/licenses/mit/)
