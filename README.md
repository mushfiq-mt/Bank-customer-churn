# Bank Customer Churn Prediction Using ANN

## Overview

This project demonstrates the use of an Artificial Neural Network (ANN) to predict bank customer churn. The goal is to identify which customers are likely to leave the bank based on various features, including demographic and financial data.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Requirements](#requirements)

## Introduction

Customer churn is a significant issue for banks, as retaining customers is generally more cost-effective than acquiring new ones. This project uses machine learning to predict customer churn, allowing the bank to take proactive measures to retain valuable customers.

## Data Preprocessing

1. **Loading the Data**: The dataset is loaded using Pandas.
2. **Dropping Unnecessary Columns**: Removed columns that are not useful for prediction.
3. **Encoding Categorical Variables**: Converted categorical variables to numerical values using one-hot encoding.
4. **Feature Scaling**: Scaled features to ensure the model performs optimally.

## Exploratory Data Analysis (EDA)

Visualized the distribution of customers based on tenure and credit score, and analyzed how these features relate to customer churn.

```python
plt.hist([Tenure_not_Exited, Tenure_Exited], color=['Green', 'Red'], label=['Not Exited', 'Exited'])
plt.legend()
plt.show()
```

## Model Building

Defined an ANN using TensorFlow and Keras with the following architecture:

- Input layer: 19 features
- Hidden layer: 10 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation

```python
model = keras.Sequential([
    keras.layers.Input(shape=(19,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

## Model Training and Evaluation

Compiled the model with the Adam optimizer and binary crossentropy loss function. Trained the model for 30 epochs and evaluated its performance on the test set.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30)
```

## Results

The model achieved significant accuracy in predicting customer churn. The performance was evaluated using a confusion matrix and a classification report.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, yp))
```

## Conclusion

The ANN model effectively predicts customer churn, providing banks with a valuable tool for retaining customers. By identifying at-risk customers, banks can take proactive steps to improve customer satisfaction and reduce churn.

## Usage

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/bank-customer-churn-prediction.git
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook to see the data preprocessing, model building, training, and evaluation steps.

## Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## Acknowledgements

This project is inspired by the need to leverage machine learning for business insights and customer retention strategies.
