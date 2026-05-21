# ChurnAI Studio

A polished customer churn prediction app built with Streamlit and explainable machine learning. ChurnAI Studio helps growth teams understand customer risk, visualize churn drivers, and act on retention opportunities with confidence.

## 🔍 Project Overview

This repository contains a Streamlit-based churn prediction solution that: 

- ingests customer churn datasets
- prepares and cleans tabular features
- trains and evaluates classification models
- computes customer churn risk scores
- displays explainability insights using SHAP-style analyses

## 🚀 Key Features

- **Dataset upload & preprocessing**: upload CSV/XLSX customer data and automatically prepare it for modeling
- **Churn prediction workflow**: train or infer churn likelihood for active customers
- **Explainable AI**: interpret model results and identify top churn drivers
- **Interactive dashboards**: view summary KPIs, risk metrics, and retention-focused recommendations
- **Production-ready layout**: Streamlit UI designed for modern analytics teams

## 📁 Repository Structure

- `app.py` — main Streamlit application entrypoint
- `model.py` — model training and pipeline orchestration
- `utils/` — preprocessing, feature engineering, modeling, and helper utilities
- `components/` — custom Streamlit UI components and charts
- `customer_churn_dataset.csv` — sample data file
- `environment/` — local Python virtual environment and dependencies

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Mohamedkhaled123-dot/churn-prediction-ai.git
cd "churn model"
```

2. Activate the Python environment:

```powershell
& "environment\Scripts\Activate.ps1"
```

3. Install dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the App

Start the Streamlit app locally:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal to explore the dashboard.

## 🎬 Demo

To see ChurnAI Studio in action, use the sample dataset included in the repository or upload your own customer churn CSV/XLSX file. The app will automatically prepare the data, train the model if needed, and display:

- an executive landing overview
- dataset pulse KPIs
- explainability insights for churn drivers
- model predictions and risk segmentation

If you want a quick preview, follow these steps:

1. Run the app locally.
2. Open the app in your browser.
3. Upload `customer_churn_dataset.csv` or your own dataset.
4. Browse the **Upload**, **EDA**, **Explainable AI**, and **Prediction** pages.

## 📌 Usage

- Navigate to the **Landing** page for app overview
- Upload your churn dataset on the **Upload** page
- Explore dataset health and preprocessing results
- Use the **EDA** and **Explainable AI** pages to inspect model behavior
- Generate customer churn predictions on the **Prediction** page

## 💡 Tips

- Ensure your dataset includes a `churn_flag` column or let the app generate one
- Keep the data schema consistent with `monthly_charges`, `contract_type`, and other customer fields used by the model

## 🤝 Contribution

Feel free to extend the app with additional explainability visualizations, model options, or retention playbook exports.

## 📄 License

This project is provided as-is. Add a license file if you want to share or publish the repository.
