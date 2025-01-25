from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

def train_transfer_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Logistic Regression model using PCA for TRANSFER type.
    """
    # Apply PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_train_pca, y_train)
    logreg_pred = logreg.predict(X_test_pca)

    # Evaluation
    print("TRANSFER Type Logistic Regression with PCA Results:")
    print(confusion_matrix(y_test, logreg_pred))
    print(classification_report(y_test, logreg_pred))


def train_cash_out_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate models for CASH_OUT type and apply ensemble voting.
    """

    # Random Forest
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("CASH_OUT Random Forest Results:")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    # XGBoost
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print("CASH_OUT XGBoost Results:")
    print(confusion_matrix(y_test, xgb_pred))
    print(classification_report(y_test, xgb_pred))

    # LightGBM
    lgbm = lgb.LGBMClassifier(random_state=42, is_unbalance=True)
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    print("CASH_OUT LightGBM Results:")
    print(confusion_matrix(y_test, lgbm_pred))
    print(classification_report(y_test, lgbm_pred))

    # Ensemble Voting
    predictions = np.array([rf_pred, xgb_pred, lgbm_pred])
    ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    # Evaluate Ensemble Model
    print("CASH_OUT Ensemble Voting Results:")
    print(confusion_matrix(y_test, ensemble_pred))
    print(classification_report(y_test, ensemble_pred))
