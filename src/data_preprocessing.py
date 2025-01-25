from sklearn.preprocessing import StandardScaler


def filter_cash_out_data(df):
    return df[df['type'] == 'CASH_OUT']

def filter_transfer_data(df):
    return df[df['type'] == 'CASH_OUT']

def preprocess_data(df):
    X = df.drop(['isFraud', 'isFlaggedFraud', 'type', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
    y = df['isFraud']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def balance_data(X, y):
    """Balance data using SMOTE."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y