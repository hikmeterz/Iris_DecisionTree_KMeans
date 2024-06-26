﻿# Iris_DecisionTree_KMeans
# Decision Tree and K-Means Analysis

## Project Description
This repository contains implementations and analysis of Decision Tree and K-Means clustering algorithms. The project uses the Iris dataset to demonstrate the performance and characteristics of these machine learning algorithms.

## Files
- `Iris.csv`: Dataset containing measurements of Iris flowers.
- `dt.py`: Implementation of a Decision Tree classifier.
- `kmeans.py`: Implementation of K-Means clustering.
- `report.ipynb`: Jupyter notebook containing the analysis and results.
- `README.md`: This file.

## Dataset Description
The Iris dataset contains the following attributes:
- `SepalLengthCm`: Sepal length in centimeters.
- `SepalWidthCm`: Sepal width in centimeters.
- `PetalLengthCm`: Petal length in centimeters.
- `PetalWidthCm`: Petal width in centimeters.
- `Species`: Iris species (Setosa, Versicolor, Virginica).

## Implementation Details

### `dt.py`
This script contains the implementation of a Decision Tree classifier. The classifier is built from scratch and includes functions to fit the model to the data and make predictions.

#### Key Features:
- **Gini Impurity**: Calculation of Gini impurity for evaluating splits.
- **Best Split**: Determination of the best feature and threshold to split the data.
- **Tree Building**: Recursive function to build the decision tree.
- **Prediction**: Function to make predictions on new data points.
