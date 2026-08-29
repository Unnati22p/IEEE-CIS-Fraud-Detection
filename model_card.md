# Model Card — IEEE-CIS Fraud Detection

## Model Overview

This project uses an XGBoost classifier to detect potentially fraudulent online transactions using the IEEE-CIS Fraud Detection dataset.

The model pipeline includes data cleaning, feature selection, categorical encoding, missing-value handling, model training, evaluation, threshold optimization, and feature importance analysis.

## Dataset

The dataset contains transaction and identity information from the IEEE-CIS Fraud Detection dataset.

### Dataset Statistics

- Total transactions: 590,540
- Non-fraudulent transactions: 569,877
- Fraudulent transactions: 20,663
- Fraud rate: 3.5%

The dataset is highly imbalanced because fraudulent transactions represent only a small percentage of all transactions.

## Data Processing

The preprocessing workflow included:

1. Loading transaction and identity data.
2. Selecting relevant features.
3. Merging transaction and identity data using `TransactionID`.
4. Removing highly sparse columns.
5. Handling missing categorical values.
6. Encoding categorical features.
7. Handling numerical missing values.
8. Splitting the dataset into training and testing sets using stratification.

### Dataset Split

- Training samples: 472,432
- Testing samples: 118,108
- Features used for modeling: 46

The fraud rate remained approximately 3.5% in both the training and testing datasets.

## Model

### Algorithm

The model uses **XGBoost Classifier** for binary fraud classification.

XGBoost was selected for its effectiveness on structured and tabular datasets and its ability to model nonlinear relationships between features.

### Class Imbalance

Because fraudulent transactions are much less frequent than legitimate transactions, class imbalance was addressed using the `scale_pos_weight` parameter.

The calculated value was:
~~~text
27.58
~~~

## Model Evaluation

The model was evaluated using metrics suitable for an imbalanced binary classification problem.

- ROC-AUC: `0.9441`

- PR-AUC: `0.6563`

The ROC-AUC score indicates strong discrimination between fraudulent and non-fraudulent transactions. PR-AUC is particularly useful for evaluating the minority fraud class in this imbalanced dataset.

## Threshold Optimization

Instead of relying only on the default classification threshold of 0.50, the decision threshold was optimized using the Precision-Recall trade-off.

### Optimized Threshold

~~~text

0.8294

~~~

### Performance at Optimized Threshold

- Precision: `0.7039`

- Recall: `0.5621`

- F1 Score: `0.6251`

The optimized threshold provides a better balance between precision and recall for fraud detection.

## Confusion Matrix

At the optimized threshold of `0.8294`, the confusion matrix was:

~~~text

[[112998    977]

 [  1812   2321]]

~~~

Where:

- True Negatives: `112,998`

- False Positives: `977`

- False Negatives: `1,812`

- True Positives: `2,321`

The model correctly identified 2,321 fraudulent transactions while producing 977 false positives on the test dataset.

## Classification Report

At the optimized threshold:

~~~text

              precision    recall  f1-score   support

Not Fraud       0.98      0.99      0.99    113975

Fraud           0.70      0.56      0.62      4133

accuracy                           0.98    118108

macro avg       0.84      0.78      0.81    118108

weighted avg    0.97      0.98      0.98    118108

~~~

The fraud class achieved a precision of approximately 0.70, recall of approximately 0.56, and F1 score of approximately 0.62.

## Feature Importance

XGBoost feature importance was used to identify features that contributed most strongly to the model's predictions.

The top features included:

~~~text

C8

R_emaildomain

C14

C5

C4

C12

card6

C1

C2

id_17

id_12

C6

C9

ProductCD

C13

~~~

The feature importance analysis provides insight into which variables were most influential for the trained model.

## ROC and Precision-Recall Analysis

The model evaluation included both ROC and Precision-Recall curve analysis.

The resulting scores were:

~~~text

ROC-AUC: 0.9441

PR-AUC:  0.6563

~~~

The Precision-Recall curve is especially relevant because the dataset contains a relatively small percentage of fraudulent transactions.

## Model Artifacts

The trained pipeline generated the following model-related files:

~~~text

features.pkl

encoder.pkl

fraud_model.pkl

~~~

These files contain the feature information, categorical encoder, and trained XGBoost model.

The model artifacts are not included in this repository because they are generated files and are excluded through `.gitignore`.

## Limitations

- The model was trained and evaluated using the IEEE-CIS Fraud Detection dataset.

- The dataset is highly imbalanced.

- Performance on real-world transaction data may differ from the reported test results.

- Fraud patterns can change over time.

- The classification threshold may need to be adjusted depending on business requirements.

- Feature importance indicates model contribution and does not establish causal relationships.

## Reproducibility

To reproduce the project:

1. Clone this repository.

2. Install the dependencies listed in `requirements.txt`.

3. Obtain the IEEE-CIS Fraud Detection dataset.

4. Place the required dataset files in the appropriate data directory.

5. Run the Jupyter notebook from beginning to end.

The notebook contains the complete workflow for data preprocessing, model training, evaluation, threshold optimization, and feature importance analysis.

## Technologies Used

- Python

- Pandas

- NumPy

- Scikit-learn

- XGBoost

- Matplotlib

- Seaborn

- Jupyter Notebook

## Project Objective

The objective of this project is to build a machine learning solution capable of detecting potentially fraudulent online transactions while addressing the challenges of highly imbalanced classification.

The project demonstrates an end-to-end fraud detection workflow, from raw transaction data preprocessing to model evaluation and threshold optimization.
