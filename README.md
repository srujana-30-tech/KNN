# KNN Classification 

## Project Overview

This project implements a **K-Nearest Neighbors (KNN)** classification algorithm to predict whether a person will purchase a product based on the given dataset.

The project covers data preprocessing, missing-value handling, visualization, categorical encoding, feature scaling, model training, prediction, and evaluation.

## Objective

The main objective of this experiment is to:

* Understand the K-Nearest Neighbors algorithm.
* Preprocess and clean the dataset.
* Handle missing values.
* Detect and visualize outliers.
* Convert categorical data into numerical form.
* Split the dataset into training and testing data.
* Scale numerical features.
* Train a KNN classification model.
* Predict purchase behavior.
* Evaluate the model using different performance metrics.

## Algorithm Used

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning classification algorithm.

It predicts the class of a new data point by looking at the **K closest data points** in the training dataset.

In this project:

```text
K = 5
```

Therefore, the model considers the **5 nearest neighbors** when making a prediction.

## Dataset

The dataset used in this experiment contains information related to customers and their purchase behavior.

The target variable is:

```text
Purchased
```

The target is converted into numerical values:

```text
0 → No
1 → Yes
```

## Technologies and Libraries

The project is implemented using Python and the following libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

The complete workflow followed in the notebook is:

```text
Load Dataset
     ↓
Understand Dataset
     ↓
Check Missing Values
     ↓
Handle Missing Values
     ↓
Visualize Data
     ↓
Identify Outliers
     ↓
Encode Categorical Variables
     ↓
Create Target Variable
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Train KNN Model
     ↓
Make Predictions
     ↓
Evaluate Model
```

## 1. Import Libraries

The required Python libraries are imported for data processing, visualization, machine learning, and evaluation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Scikit-learn is used for preprocessing, model training, and evaluation.

## 2. Load the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("Lab 2 KNN.csv")
```

The dataset is then inspected to understand its structure and contents.

## 3. Data Understanding

The dataset is checked for:

* Number of rows and columns
* Data types
* Missing values
* Statistical information
* Duplicate values
* Distribution of features

This helps understand the quality and structure of the dataset before applying machine learning.

## 4. Handling Missing Values

Missing values are handled during preprocessing.

For categorical variables, the **mode** is used.

For numerical variables, the **median** is used.

This prevents missing values from causing problems during model training.

## 5. Data Visualization

Visualization techniques such as:

* Histograms
* Boxplots
* Count plots

are used to understand the distribution of the data.

Boxplots are particularly useful for identifying possible outliers.

## 6. Target Variable

The `Purchased` column is used as the target variable.

It represents whether the customer purchased the product.

```text
No  → 0
Yes → 1
```

The KNN model learns from the input features and predicts this target value.

## 7. Encoding Categorical Data

Machine learning algorithms require numerical input.

Therefore, categorical variables are converted into numerical form using **One-Hot Encoding**.

One-hot encoding creates separate columns for different categories.

For example:

```text
Gender
Male
Female
```

can be converted into numerical columns representing the categories.

## 8. Train-Test Split

The dataset is divided into:

* **70% training data**
* **30% testing data**

The training data is used to train the KNN model, while the testing data is used to evaluate how well the model performs on unseen data.

## 9. Feature Scaling

KNN is a distance-based algorithm, so feature scaling is important.

The project uses:

```text
RobustScaler
```

RobustScaler scales the features while being less affected by extreme values and outliers.

This ensures that features with larger numerical ranges do not dominate the distance calculation.

## 10. KNN Model

The K-Nearest Neighbors classifier is trained using:

```text
n_neighbors = 5
```

This means that for every new data point, the model finds the 5 closest training points and uses their classes to determine the prediction.

For classification, the class with the majority of votes among the nearest neighbors is selected.

### Simple Example

If the 5 nearest neighbors are:

```text
Yes
Yes
No
Yes
No
```

The majority is:

```text
Yes
```

Therefore, the new customer is predicted as:

```text
Purchased = Yes
```

## 11. Prediction

After training the model, predictions are made using the test data.

The predicted values are compared with the actual values to determine the performance of the model.

## 12. Model Evaluation

The model is evaluated using several metrics.

### Accuracy

Accuracy represents the proportion of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted values.

A lower MSE generally indicates smaller prediction errors.

### Confusion Matrix

The confusion matrix shows the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

It helps understand where the classification model is making correct and incorrect predictions.

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

These metrics provide more detailed information about classification performance.

## Advantages of KNN

* Simple and easy to understand.
* Easy to implement.
* Does not require complex mathematical assumptions.
* Can be used for classification and regression.
* Works well for smaller datasets.

## Limitations of KNN

* Can be computationally expensive for large datasets.
* Sensitive to feature scaling.
* Sensitive to irrelevant features.
* Choosing the correct value of K is important.
* Performance can be affected by noisy data and outliers.

## Why Scaling is Important in KNN

KNN calculates the distance between data points.

If one feature has a much larger numerical range than another feature, it can dominate the distance calculation.

Therefore, feature scaling is performed before training the KNN model.

## Key Concepts

```text
KNN
↓
Find nearest data points
↓
Consider K neighbors
↓
Take majority class
↓
Make prediction
```

For this experiment:

```text
Algorithm       : K-Nearest Neighbors
Type            : Supervised Learning
Task            : Classification
K Value         : 5
Target          : Purchased
Encoding        : One-Hot Encoding
Scaling         : RobustScaler
Train Data      : 70%
Test Data       : 30%
```

## Conclusion

The experiment demonstrates how the **K-Nearest Neighbors classification algorithm** can be used to predict customer purchase behavior.

The dataset is first cleaned and preprocessed, categorical features are encoded, numerical features are scaled using RobustScaler, and the data is divided into training and testing sets.

A KNN model with **K = 5** is then trained and used to make predictions. The performance is evaluated using accuracy, MSE, confusion matrix, and classification report.

## Files

```text
Lab 2 KNN (1).ipynb
Lab 2 KNN.csv
README.md
```

## Viva Short Answer

**What is KNN?**

KNN is a supervised machine learning algorithm that predicts the class of a data point based on the classes of its nearest neighbors.

**Why is scaling required?**

Because KNN uses distance calculations, features with larger values can dominate the distance. Scaling puts features on a comparable scale.

**Why K = 5?**

In this experiment, K is set to 5, so the model considers the five nearest neighbors when making a prediction.

**What is the target variable?**

`Purchased`, which represents whether the customer purchased the product.

**Which scaling method is used?**

RobustScaler.

**Which encoding technique is used?**

One-Hot Encoding.

**What type of machine learning is KNN?**

Supervised learning.

**What type of problem is solved?**

Classification.

**What does accuracy measure?**

It measures the proportion of predictions that are correct.
