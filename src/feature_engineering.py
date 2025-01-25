def calculate_balance_difference(df):
    """Calculate balance difference feature."""
    df['balance_difference'] = (df['oldbalanceOrg'] - df['newbalanceOrig']) - (df['newbalanceDest'] - df['oldbalanceDest'])
    df['balance_flag'] = (df['balance_difference'] != 0).astype(int)
    return df

def flag_high_transaction_receivers(df):
    """Flag receivers with more than a given number of transactions."""
    receiver_transaction_counts = df['nameDest'].value_counts()
    frequent_receivers = receiver_transaction_counts[(receiver_transaction_counts > 39) & (receiver_transaction_counts < 51)].index
    df['frequency_flag'] = df['nameDest'].isin(frequent_receivers).astype(int)
    return df

def flag_large_amounts(df):
    fraud_transactions = df[df['isFraud'] == 1]
    threshold = fraud_transactions['amount'].quantile(0.95)  # 95th percentile
    df['amount_flag'] = (df['amount'] <= threshold).astype(int)
    return df