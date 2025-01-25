# main.py

from src import (
    filter_transfer_data,
    filter_cash_out_data,
    preprocess_data,
    calculate_balance_difference,
    flag_high_transaction_receivers,
    flag_large_amounts,
    train_transfer_model,
    train_cash_out_models
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def main():
    # Load the dataset (update the path to your actual dataset)
    df = pd.read_csv("data/data.csv")

    # ---- TRANSFER Transactions ----
    print("\n=== Processing TRANSFER Transactions ===")
    df_transfer = filter_transfer_data(df)  # Filter data for TRANSFER
    df_transfer = calculate_balance_difference(df_transfer)
    df_transfer = flag_high_transaction_receivers(df_transfer)
    df_transfer = flag_large_amounts(df_transfer)
    X_transfer, y_transfer = preprocess_data(df_transfer)  # Preprocess data
    X_transfer_train, X_transfer_test, y_transfer_train, y_transfer_test = train_test_split(
        X_transfer, y_transfer, test_size=0.2, random_state=42
    )
    train_transfer_model(X_transfer_train, y_transfer_train, X_transfer_test, y_transfer_test)  # Train and evaluate PCA + Logistic Regression

    # ---- CASH OUT Transactions ----
    print("\n=== Processing CASH OUT Transactions ===")
    df_cash_out = filter_cash_out_data(df)  # Filter data for CASH_OUT
    df_cash_out = calculate_balance_difference(df_cash_out)  # Add balance difference feature
    df_cash_out = flag_high_transaction_receivers(df_cash_out)  # Add high transaction receiver flag
    df_cash_out = flag_large_amounts(df_cash_out)  # Add large amount flag
    X_cash_out, y_cash_out = preprocess_data(df_cash_out)  # Preprocess data
    X_cash_out_train, X_cash_out_test, y_cash_out_train, y_cash_out_test = train_test_split(
        X_cash_out, y_cash_out, test_size=0.2, random_state=42
    )
    train_cash_out_models(X_cash_out_train, y_cash_out_train, X_cash_out_test, y_cash_out_test)  # Train and evaluate Random Forest, XGBoost, SVM, and Voting Ensemble


if __name__ == "__main__":
    main()
