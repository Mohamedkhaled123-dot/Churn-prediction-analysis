import pandas as pd


class FeatureEngineer:
    """Simple feature engineering utility for churn dataset preview."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = df.copy()

        support_calls = pd.Series(0, index=engineered.index)
        if "support_calls" in engineered.columns:
            support_calls = engineered["support_calls"].astype(float).fillna(0)

        tenure = pd.Series(0.0, index=engineered.index)
        if "tenure" in engineered.columns:
            tenure = engineered["tenure"].astype(float).fillna(0)

        monthly = pd.Series(0.0, index=engineered.index)
        if "monthly_charges" in engineered.columns:
            monthly = engineered["monthly_charges"].astype(float).fillna(0)

        total = pd.Series(0.0, index=engineered.index)
        if "total_charges" in engineered.columns:
            total = engineered["total_charges"].astype(float).fillna(0)

        engineered["support_call_rate"] = support_calls / (tenure.replace(0, 1) + 1)

        if monthly.nunique(dropna=True) >= 5:
            engineered["charge_segment"] = pd.qcut(monthly, q=5, labels=False, duplicates="drop")
        elif monthly.nunique(dropna=True) > 1:
            bins = min(monthly.nunique(dropna=True), 5)
            engineered["charge_segment"] = pd.cut(monthly, bins=bins, labels=False, include_lowest=True)
        else:
            engineered["charge_segment"] = 0

        engineered["charge_per_tenure"] = monthly / (tenure + 1)
        engineered["avg_total_per_month"] = total / (tenure + 1)
        engineered["value_gap"] = monthly - engineered["avg_total_per_month"]

        engineered["support_segment"] = pd.cut(
            support_calls,
            bins=[-1, 0, 2, 5, 20],
            labels=["No Calls", "Low", "Medium", "High"],
            include_lowest=True,
        )

        engineered["frustration_score"] = support_calls * monthly
        engineered["is_monthly_contract"] = engineered.get("contract", "Month-to-month") == "Month-to-month"
        engineered["high_risk_customer"] = (
            engineered["is_monthly_contract"]
            & (monthly > monthly.median())
            & (support_calls >= 2)
        )

        categorical_cols = engineered.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            engineered = pd.get_dummies(engineered, columns=categorical_cols, dummy_na=False)

        return engineered
