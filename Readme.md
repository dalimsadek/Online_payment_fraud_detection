# Fraud Detection System

This project focuses on detecting fraudulent transactions for **TRANSFER** and **CASH_OUT** types. The system implements data preprocessing, feature engineering, and model training using machine learning algorithms like **Logistic Regression**, **Random Forest**, **SVM**, and **XGBoost**, with an ensemble voting mechanism for the CASH_OUT transactions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The project is divided into two main parts:
1. **TRANSFER Transactions**: Uses PCA for dimensionality reduction and trains a Logistic Regression model.
2. **CASH_OUT Transactions**: Trains Random Forest, SVM, and XGBoost models, and combines them using an ensemble voting strategy for better accuracy.

---

## Dataset

- The dataset should be placed in the `data/` directory as `data.csv`.
- Ensure that the dataset contains relevant features for fraud detection, such as transaction type, amount, and balance information.

---

## Features

### Preprocessing Steps:
1. **Feature Engineering**:
   - Calculating balance differences.
   - Flagging high transaction receivers.
   - Flagging large transactions.
2. **Data Filtering**:
   - Separate data into **TRANSFER** and **CASH_OUT** types.

### Models:
- **TRANSFER**: PCA + Logistic Regression.
- **CASH_OUT**:
  - Random Forest
  - SVM
  - XGBoost
  - Ensemble Voting

### Evaluation:
- Confusion Matrix
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score

---

## Installation

### Prerequisites
- Python 3.8 or later
- Required libraries:
  ```bash
  pip install -r requirements.txt
