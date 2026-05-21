import pandas as pd
import streamlit as st
from .config import DATA_PATH


def ensure_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'churn_flag' not in df.columns and 'churn' in df.columns:
        df['churn_flag'] = df['churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    elif 'churn_flag' not in df.columns:
        df['churn_flag'] = 0
    return df


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_churn_flag(df)


def get_active_dataset() -> pd.DataFrame | None:
    active = st.session_state.get('cleaned_df') if st.session_state.get('cleaned_df') is not None else st.session_state.get('uploaded_df')
    return ensure_churn_flag(active) if active is not None else None


def summarize_data(df: pd.DataFrame) -> dict:
    missing = df.isna().sum().sort_values(ascending=False)
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    shape = df.shape
    duplicate_count = df.duplicated().sum()
    health_score = max(0, 100 - int(missing.sum() * 2 + duplicate_count * 5))
    return {
        'shape': shape,
        'missing': missing,
        'dtypes': dtypes,
        'duplicates': duplicate_count,
        'health_score': health_score,
        'missing_summary': (missing / len(df) * 100).round(2)
    }


def clean_dataset(df: pd.DataFrame, fill_strategy: str = 'median') -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    numeric_cols = cleaned.select_dtypes(include=['int64', 'float64']).columns
    if fill_strategy == 'median':
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
    else:
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].mean())

    category_defaults = {
        'internet_service': 'Fiber optic',
        'contract': 'Month-to-month',
        'payment_method': 'Electronic check',
        'tech_support': 'No',
        'online_security': 'No',
    }

    for col in cleaned.select_dtypes(include=['object']).columns:
        if col in category_defaults:
            cleaned[col] = cleaned[col].fillna(category_defaults[col])
        else:
            fill_value = cleaned[col].mode(dropna=True)
            fill_value = fill_value.iloc[0] if not fill_value.empty else 'Unknown'
            cleaned[col] = cleaned[col].fillna(fill_value)

    return ensure_churn_flag(cleaned)
