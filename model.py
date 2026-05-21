import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report

from utils.feature_engineering import FeatureEngineer


# =========================
# Load Data
# =========================
df = pd.read_csv(r"C:\Users\Mohamed\Desktop\churn model\customer_churn_dataset.csv")

df['internet_service'] = df['internet_service'].fillna('Fiber')

X = df.drop(columns=['churn'])
y = df['churn']


# =========================
# Train Test Split
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================
# Feature Engineering
# =========================
feature_engineer = FeatureEngineer()

x_train = feature_engineer.fit_transform(x_train)
x_test = feature_engineer.transform(x_test)


# =========================
# Columns
# =========================
cat_cols = [
    'contract',
    'payment_method',
    'internet_service',
    'tech_support',
    'online_security',
    'support_segment'
]

# keep only existing columns (fix error)
cat_cols = [col for col in cat_cols if col in x_train.columns]

# make explicit lists for ColumnTransformer
num_cols = list(x_train.select_dtypes(include=['int64', 'float64']).columns)


# =========================
# Preprocessing
# =========================
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])


# =========================
# Models
# =========================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
if pos == 0:
    scale_pos_weight = 1.0
else:
    scale_pos_weight = neg / pos


cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    verbose=0
)

lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)


base_models = [
    ('cat', cat_model),
    ('lgbm', lgbm_model),
    ('xgb', xgb_model)
]


stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    stack_method='predict_proba',
    cv=5,
    n_jobs=-1
)


# =========================
# FULL PIPELINE
# =========================
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", stack_model)
])


# =========================
# Train
# =========================
pipeline.fit(x_train, y_train)


# =========================
# Predict
# =========================
preds = pipeline.predict(x_test)

print(classification_report(y_test, preds))

# =========================
# Save Model
# =========================
joblib.dump(pipeline, "model_pipeline.pkl")

print("Pipeline Saved Successfully 🚀")