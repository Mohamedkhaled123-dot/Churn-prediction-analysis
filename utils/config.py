import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(ROOT_DIR), "customer_churn_dataset.csv")

APP_TITLE = "ChurnAI Studio"
APP_ICON = "🤖"
APP_DESCRIPTION = "A premium customer churn intelligence platform for modern growth teams."

PAGES = {
    "Landing": "Landing page with product overview and key KPIs.",
    "Upload": "Upload, preview, and inspect churn data.",
    "Cleaning": "Data cleaning and transformation workflow.",
    "EDA": "Interactive exploratory data analysis.",
    "Feature Engineering": "Create and visualize engineered features.",
    "Model Training": "Train and compare churn prediction models.",
    "Explainable AI": "Global and local model explainability.",
    "Prediction": "Real-time prediction and deployment.",
    "Business Insights": "Executive insights and ROI guidance.",
    "Monitoring": "Model health, drift, and usage analytics.",
    "AI Assistant": "OpenAI chat assistant for churn strategy and insights."
}

CATEGORICAL_FEATURES = [
    "contract",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security",
    "support_segment"
]

PAYMENT_METHODS = [
    "Credit",
    "Debit",
    "Electronic check",
    "Mailed check",
    "Bank transfer"
]

INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
BINARY_OPTIONS = ["Yes", "No"]

LOTTIE_HERO = "https://assets7.lottiefiles.com/packages/lf20_z02a1pkl.json"
LOTTIE_ANALYTICS = "https://assets7.lottiefiles.com/packages/lf20_h5vhzjrp.json"
LOTTIE_DEPLOY = "https://assets7.lottiefiles.com/packages/lf20_2qx1atkl.json"
