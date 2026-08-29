# IEEE-CIS Fraud Detection

An end-to-end machine learning project for detecting fraudulent online transactions using the **IEEE-CIS Fraud Detection dataset** and **XGBoost**.

The project focuses on handling highly imbalanced transaction data, building an effective fraud classification model, evaluating it using appropriate metrics, and optimizing the classification threshold to improve practical fraud detection performance.

---

## 🚀 Live Demo

👉 [Try the Live Streamlit App](https://ieee-cis-frauddetection.streamlit.app)

The application provides an interactive interface for predicting whether an online transaction is potentially fraudulent using the trained XGBoost model.

## 📌 Project Overview

Online transaction fraud is a challenging machine learning problem because fraudulent transactions represent only a very small percentage of total transactions.

In this project, transaction and identity information from the IEEE-CIS dataset are processed and used to train an XGBoost classifier capable of identifying potentially fraudulent transactions.

The complete workflow includes:

- Data loading and integration
- Transaction and identity data merging
- Missing-value analysis
- Feature selection
- Memory-efficient preprocessing
- Categorical feature encoding
- Stratified train-test splitting
- Class imbalance handling
- XGBoost model training
- Model evaluation
- Classification threshold optimization
- Feature importance analysis
- ROC and Precision-Recall analysis

---

## 📊 Dataset

The project uses the **IEEE-CIS Fraud Detection dataset**.

Transaction and identity information are provided separately and are connected using the common `TransactionID` field.

### Dataset Statistics

| Description | Value |
|---|---:|
| Total Transactions | 590,540 |
| Final Modeling Features | 46 |
| Non-Fraudulent Transactions | 569,877 |
| Fraudulent Transactions | 20,663 |
| Fraud Rate | 3.5% |

Only around **3.5% of all transactions are fraudulent**, making this a highly imbalanced classification problem.

Because of this imbalance, accuracy alone is not a reliable measure of model performance. Metrics such as **Precision, Recall, F1 Score, ROC-AUC, and PR-AUC** are used for evaluation.

---

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

1. Loading transaction and identity datasets.
2. Selecting relevant transaction and identity features.
3. Merging both datasets using `TransactionID`.
4. Removing unnecessary and highly sparse features.
5. Reducing memory usage for efficient processing.
6. Separating the target variable `isFraud`.
7. Identifying numerical and categorical features.
8. Handling missing categorical values.
9. Encoding categorical variables using `OrdinalEncoder`.
10. Performing a stratified train-test split to preserve the fraud ratio.
11. Handling class imbalance using positive-class weighting in XGBoost.

### Train-Test Split

| Dataset | Samples |
|---|---:|
| Training | 472,432 |
| Testing | 118,108 |

The fraud rate was maintained at approximately **3.5%** in both training and testing data.

---

## 🤖 Machine Learning Model

The project uses an **XGBoost Classifier**.

XGBoost was selected because of its strong performance on structured/tabular data and its ability to model complex nonlinear relationships between transaction characteristics.

Since fraudulent transactions represent the minority class, `scale_pos_weight` was used to give additional importance to fraudulent observations during training.

The calculated positive-class weight was approximately:

```text
27.58
```

---

## 📈 Model Performance

The initial model was evaluated using a default classification threshold of `0.50`.

### Initial Results

| Metric | Score |
|---|---:|
| ROC-AUC | **0.9441** |
| PR-AUC | **0.6564** |
| Precision | **0.2554** |
| Recall | **0.8195** |
| F1 Score | **0.3895** |

The model achieved high recall, detecting approximately **82% of fraudulent transactions**, but generated a relatively large number of false positives.

### Initial Confusion Matrix

```text
[[104102   9873]
 [   746   3387]]
```

This showed that the default threshold successfully captured a large proportion of fraudulent transactions but was too aggressive in flagging legitimate transactions.

---

## 🎯 Threshold Optimization

Using a fixed probability threshold of `0.50` is not always optimal for fraud detection.

A lower threshold generally detects more fraudulent transactions but produces more false alarms, while a higher threshold improves precision but may miss more fraudulent transactions.

The Precision-Recall curve was therefore analyzed to identify the threshold that maximized the F1 Score.

### Optimal Threshold

```text
0.8294
```

### Performance After Threshold Optimization

| Metric | Score |
|---|---:|
| Precision | **0.7039** |
| Recall | **0.5621** |
| F1 Score | **0.6251** |

Threshold optimization increased precision from approximately **25.5% to 70.4%** while producing a significantly better balance between precision and recall.

### Optimized Confusion Matrix

```text
[[112998    977]
 [  1812   2321]]
```

Where:

- **True Negatives:** 112,998
- **False Positives:** 977
- **False Negatives:** 1,812
- **True Positives:** 2,321

At the optimized threshold, the model correctly identified **2,321 fraudulent transactions** while producing only **977 false-positive alerts** on the test dataset.

---

## 🔍 Feature Importance

XGBoost feature importance was used to understand which variables contributed most strongly to fraud predictions.

The top features included:

1. `C8`
2. `R_emaildomain`
3. `C14`
4. `C5`
5. `C4`
6. `C12`
7. `card6`
8. `C1`
9. `C2`
10. `id_17`
11. `id_12`
12. `C6`
13. `C9`
14. `ProductCD`
15. `C13`

The most influential feature was `C8`, followed by email-domain, transaction-count, card-related, and identity-related features.

---

## 📉 Model Evaluation

Multiple evaluation techniques were used because fraud detection involves highly imbalanced data.

The model was evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

### ROC-AUC

```text
0.9441
```

The ROC-AUC indicates that the model has strong ability to distinguish between fraudulent and legitimate transactions.

### PR-AUC

```text
0.6563
```

PR-AUC is particularly important for this project because fraudulent transactions represent only **3.5%** of the dataset.

---

## 💡 Key Results

| Result | Value |
|---|---:|
| Transactions Analyzed | **590,540** |
| Features Used | **46** |
| Fraud Rate | **3.5%** |
| ROC-AUC | **0.9441** |
| PR-AUC | **0.6563** |
| Optimal Threshold | **0.8294** |
| Optimized Precision | **70.39%** |
| Optimized Recall | **56.21%** |
| Optimized F1 Score | **0.6251** |
| Correctly Detected Fraud Cases | **2,321** |
| False Positive Alerts | **977** |

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **Google Colab**
- **Git**
- **GitHub**

---

## 📁 Project Structure

```text
IEEE-CIS-Fraud-Detection/
│
├── 01_Fraud_Detection_EDA_and_Model.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The original IEEE-CIS dataset is not stored in this repository because of its large file size.

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Unnati22p/IEEE-CIS-Fraud-Detection.git
```

Move into the project directory:

```bash
cd IEEE-CIS-Fraud-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the **IEEE-CIS Fraud Detection** dataset from Kaggle.

Required training files:

```text
train_transaction.csv
train_identity.csv
```

### 4. Open the Notebook

Open:

```text
01_Fraud_Detection_EDA_and_Model.ipynb
```

The notebook can be executed locally using Jupyter Notebook or through Google Colab.

### 5. Run the Pipeline

Run the notebook cells sequentially to perform:

```text
Data Loading
     ↓
Feature Selection
     ↓
Transaction + Identity Merge
     ↓
Data Preprocessing
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
XGBoost Training
     ↓
Fraud Probability Prediction
     ↓
Threshold Optimization
     ↓
Model Evaluation
```

---

## 🧠 Key Learnings

This project demonstrates several important concepts in real-world machine learning:

- Working with a dataset containing more than **590K transactions**
- Handling highly imbalanced classification problems
- Memory-efficient preprocessing of large datasets
- Combining transaction and identity information
- Encoding categorical features
- Training gradient-boosted decision trees
- Handling class imbalance using class weighting
- Evaluating models beyond simple accuracy
- Understanding the precision-recall trade-off
- Optimizing classification thresholds
- Analyzing model feature importance

One of the main findings was that the default classification threshold was not optimal for the business problem.

Increasing the threshold from `0.50` to approximately `0.8294` improved fraud precision from **25.54% to 70.39%** and increased the F1 Score from **0.3895 to 0.6251**.

---

## 🚀 Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Cross-validation
- Advanced behavioral feature engineering
- SHAP-based model explainability
- Probability calibration
- Cost-sensitive threshold optimization
- FastAPI model-serving endpoint
- Interactive fraud-monitoring dashboard
- Real-time transaction processing
- Docker containerization
- Cloud deployment

---

## 👤 Author

**Unnati Patil**

GitHub: `Unnati22p`

---

⭐ If you found this project useful, feel free to star the repository.
