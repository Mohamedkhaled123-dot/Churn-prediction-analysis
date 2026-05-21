import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from .feature_engineering import FeatureEngineer


def prepare_features(df: pd.DataFrame, drop_target: bool = True, reference_columns: list[str] | None = None):
    df = df.copy()
    if drop_target and 'churn' in df.columns:
        df = df.drop(columns=['churn'])
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    feature_engineer = FeatureEngineer()
    engineered = feature_engineer.transform(df)
    if 'churn_flag' in engineered.columns and drop_target:
        engineered = engineered.drop(columns=['churn_flag'])
    if reference_columns is not None:
        engineered = engineered.reindex(columns=reference_columns, fill_value=0)
    return engineered


def get_training_data(df: pd.DataFrame):
    if 'churn_flag' not in df.columns and 'churn' in df.columns:
        df = df.copy()
        df['churn_flag'] = df['churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    X = prepare_features(df)
    y = df['churn_flag']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_model_candidates():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    }
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5, random_state=42)
    if LGBMClassifier is not None:
        models['LightGBM'] = LGBMClassifier(n_estimators=120, learning_rate=0.08, random_state=42)
    if CatBoostClassifier is not None:
        models['CatBoost'] = CatBoostClassifier(verbose=0, iterations=120, random_state=42)
    return models


def train_models(df: pd.DataFrame):
    x_train, x_test, y_train, y_test = get_training_data(df)
    models = build_model_candidates()
    training_results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_test)
        training_results.append({
            'name': name,
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importances': getattr(model, 'feature_importances_', None)
        })
    return training_results


def get_best_model(results):
    if not results:
        return None
    return max(results, key=lambda item: item['f1'])


def predict_with_model(model, input_data: pd.DataFrame):
    engineered_input = prepare_features(input_data, drop_target=False)
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is not None:
        engineered_input = engineered_input.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(engineered_input)[0]
    probability = model.predict_proba(engineered_input)[0][1] if hasattr(model, 'predict_proba') else 0.0
    return prediction, probability

