# ChurnAI Studio
## 📊 *ChurnAI Studio: Executive Summary*

*What is it?* 
ChurnAI Studio is a web-based dashboard application that helps companies predict which customers are likely to leave (churn) and understand why. It combines machine learning with visual analytics to turn raw customer data into actionable retention strategies.

---

## 🎯 *For Non-Technical Audiences (Business Leaders)*

### *The Problem It Solves*
Losing customers is expensive. ChurnAI Studio identifies at-risk customers before they leave, so your team can take action proactively through retention campaigns, personalized offers, or service improvements.

### *What You See & Do*

1. *Upload Your Data*
   - Drop a CSV or Excel file with your customer information (billing history, contract details, support interactions, etc.)
   - The system automatically cleans and prepares the data

2. *Executive Dashboard (Landing Page)*
   - *Key Metrics at a glance:*
     - Total active customers
     - Current churn rate (% leaving)
     - Annual revenue at risk
     - Overall retention rate
   - Shows you the "pulse" of customer health in 4 numbers

3. *Data Explorer (EDA Page)*
   - Build interactive charts to explore patterns:
     - Which customer segments churn most?
     - How does contract length affect retention?
     - What's the relationship between billing and churn?
   - 8+ visualization types (scatter plots, histograms, bar charts, etc.)

4. *AI Explainability (Explainable AI Page)*
   - *Plain English:* See the top factors driving churn
     - E.g., "Customers with month-to-month contracts are 3x more likely to churn"
     - "High support calls correlate with churn risk"
   - Global view: What matters across all customers
   - Individual view: Why this specific customer is at risk

5. *Predict & Act (Prediction Page)*
   - Enter a customer's details (contract type, billing, tenure, etc.)
   - Get a *churn risk score* (0-100%)
   - Receive automated recommendations:
     - *High risk (>65%):* Offer retention package immediately
     - *Medium risk (35-65%):* Monitor and engage
     - *Low risk (<35%):* Maintain service quality

### *Business Impact*
- Identify 89.4% of at-risk customers accurately
- Make data-driven retention decisions in seconds
- Allocate retention budgets to highest-impact customers

---

## 🔬 *For Data Scientists & Technical Teams*

### *Architecture & Tech Stack*

*Frontend:* Streamlit (Python-based UI framework)
- Rapid iteration for analytics dashboards
- Native integration with ML models
- Custom CSS for professional styling (91.4% Python, 8.6% CSS)

*Core ML Pipeline:*
- *Feature Engineering:* Automated preprocessing for categorical and numerical features
- *Model Stack:* Ensemble of three base learners:
  - *CatBoost* (handles categorical features natively)
  - *LightGBM* (gradient boosting, fast inference)
  - *XGBoost* (robust gradient boosting)
  - *Meta-learner:* Logistic Regression (combines predictions)

*Explainability:* SHAP (SHapley Additive exPlanations)
- Global feature importance: Bar plots showing average impact
- Local explanations: Waterfall plots for individual predictions

### *Model Training & Deployment Flow*

python
1. Data Loading & Cleaning
   ├─ Fill missing values (median for numeric, mode for categorical)
   ├─ Remove duplicates
   └─ Ensure 'churn_flag' target variable exists

2. Feature Engineering
   ├─ Temporal features (tenure-based segments)
   ├─ Derived metrics (charges/tenure ratios)
   └─ Support call categorization

3. Preprocessing Pipeline
   ├─ Numerical: Median imputation
   └─ Categorical: One-hot encoding (handle_unknown="ignore")

4. Model Training
   ├─ Train/test split (80/20, stratified by churn)
   ├─ Class weight balancing (account for imbalanced classes)
   └─ Ensemble stacking (5-fold cross-validation)

5. SHAP Analysis
   ├─ Initialize explainer (KernelExplainer or TreeExplainer)
   ├─ Compute SHAP values for interpretability
   └─ Generate waterfall & bar plots


### *Key Technical Features*

| Feature | Implementation |
|---------|-----------------|
| *Data Validation* | Automated schema checking, missing value detection |
| *Handling Imbalance* | Class weight adjustment in XGBoost (scale_pos_weight) |
| *Hyperparameter Tuning* | Pre-configured optimal parameters for each model |
| *Inference Speed* | <3 seconds end-to-end (model + analytics refresh) |
| *Explainability* | SHAP + feature importance + local explanations |
| *Reproducibility* | Fixed random_state=42 for deterministic results |

### *Model Selection Logic*

python
# Stacking Classifier Configuration
estimators = [CatBoost, LightGBM, XGBoost]
meta_learner = LogisticRegression()
cv_strategy = 5-fold stratified


- *Why stacking?* Combines strengths of tree-based models while reducing overfitting
- *Why these models?*
  - CatBoost: Native categorical feature support (no manual encoding)
  - LightGBM: Memory efficient, handles large datasets
  - XGBoost: Industry standard, proven churn prediction performance

### *Sample Data Schema*
The system expects columns like:
- customer_id, tenure, monthly_charges, total_charges
- contract (Month-to-month, One year, Two year)
- payment_method, internet_service
- tech_support, online_security (Yes/No)
- support_calls (numeric)
- churn or churn_flag (target variable)

### *Extensibility*
- Add new model architectures in model.py
- Implement custom feature engineering in utils/feature_engineering.py
- Build additional EDA visualizations with Plotly
- Deploy as standalone service or Streamlit Cloud

---

## 🚀 *Quick Demo Walkthrough*

### *Step 1: Launch*
bash
streamlit run app.py


### *Step 2: Authenticate & Navigate*
- Simple sign-in (auth system in place)
- 5-page navigation: Landing → Upload → EDA → Explainable AI → Prediction

### *Step 3: Upload Data*
- Click "Upload" or load the included customer_churn_dataset.csv
- Review data quality metrics

### *Step 4: Explore Patterns*
- EDA page: Create scatter plots of (tenure vs. monthly_charges, colored by churn)
- Spot correlations visually

### *Step 5: Understand Drivers*
- Explainable AI page: See top 5 features driving churn globally
- Select a customer: View their personalized explanation waterfall

### *Step 6: Predict & Act*
- Input customer: 12 months tenure, $65/month, month-to-month contract
- Get churn probability: *67%* → Recommendation: "Offer retention package immediately"

---

## 💼 *Use Cases*

| User | Action | Outcome |
|------|--------|---------|
| *CFO* | Reviews landing KPIs | Quantifies revenue at risk from churn |
| *Retention Manager* | Uses Prediction page | Prioritizes 50 high-risk customers for outreach |
| *Product Manager* | Explores EDA | Discovers contract type is #1 churn driver |
| *Data Scientist* | Checks SHAP plots | Validates model decision boundaries |
| *Compliance Officer* | Reviews explainability | Understands why customers flagged as at-risk |

---

## 📈 *Key Advantages*

✅ *89.4% accuracy* – High-confidence predictions
✅ *3-second inference* – Real-time decisioning
✅ *Fully explainable* – SHAP + global/local insights
✅ *End-to-end pipeline* – Data upload to retention action
✅ *No coding needed* – Analyst-friendly UI
✅ *Production-ready* – Handles authentication, deployment

This is a *complete, enterprise-grade churn analytics platform* suitable for telecom, SaaS, subscription services, and any business with customer retention challenges.