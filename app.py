import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

def load_css():
    with open("style.css", "r", encoding="utf-8") as file:
        css = file.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "churn_model_pipeline.joblib"
    )


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    with open(
        "churn_model_metadata.json",
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD CHURN DATA
# ============================================================

@st.cache_data
def load_churn_data():

    return pd.read_csv(
        "churn.csv"
    )


# ============================================================
# LOAD SCORED DATA
# ============================================================

@st.cache_data
def load_scored_data():

    try:

        return pd.read_csv(
            "customers_scored.csv"
        )

    except FileNotFoundError:

        return None


# ============================================================
# LOAD PROJECT FILES
# ============================================================

model = load_model()

metadata = load_metadata()

df = load_churn_data()

scored_df = load_scored_data()

threshold = metadata[
    "decision_threshold"
]


# ============================================================
# FIND PREPROCESSOR AND CLASSIFIER
# ============================================================

preprocessor = None

classifier = None


for step_name, step_object in model.named_steps.items():

    if (
        preprocessor is None
        and hasattr(
            step_object,
            "transform"
        )
        and hasattr(
            step_object,
            "get_feature_names_out"
        )
    ):

        preprocessor = step_object


    if (
        classifier is None
        and hasattr(
            step_object,
            "predict_proba"
        )
    ):

        classifier = step_object


if preprocessor is None:

    st.error(
        "Preprocessing component "
        "could not be found."
    )

    st.stop()


if classifier is None:

    st.error(
        "Classifier could not be found."
    )

    st.stop()


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(
    _classifier,
    background_data
):

    classifier_name = (
        _classifier
        .__class__
        .__name__
    )


    tree_models = [

        "RandomForestClassifier",

        "XGBClassifier",

        "LGBMClassifier",

        "DecisionTreeClassifier",

        "ExtraTreesClassifier",

        "GradientBoostingClassifier",

        "HistGradientBoostingClassifier"

    ]


    if classifier_name in tree_models:

        return shap.TreeExplainer(
            _classifier
        )


    if classifier_name == "LogisticRegression":

        return shap.LinearExplainer(
            _classifier,
            background_data
        )


    return shap.Explainer(
        _classifier,
        background_data
    )


# ============================================================
# SHAP BACKGROUND DATA
# ============================================================

drop_columns = [

    "customerID",

    "Churn",

    "ChurnFlag"

]


X_background = df.drop(
    columns=drop_columns,
    errors="ignore"
)


transformed_background = (
    preprocessor.transform(
        X_background
    )
)


if hasattr(
    transformed_background,
    "toarray"
):

    transformed_background = (
        transformed_background.toarray()
    )


explainer = create_shap_explainer(
    classifier,
    transformed_background
)


# ============================================================
# RETENTION RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    customer_data,
    prediction
):

    recommendations = []


    if prediction == 1:

        recommendations.append(
            "Contact the customer proactively "
            "to understand possible service concerns."
        )


    contract = customer_data[
        "Contract"
    ].iloc[0]


    if contract == "Month-to-month":

        recommendations.append(
            "Offer a longer-term contract option "
            "with an appropriate retention incentive."
        )


    tenure = customer_data[
        "tenure"
    ].iloc[0]


    if tenure <= 12:

        recommendations.append(
            "Provide additional onboarding and "
            "early-stage customer support."
        )


    monthly_charges = customer_data[
        "MonthlyCharges"
    ].iloc[0]


    if monthly_charges >= 70:

        recommendations.append(
            "Review the customer's current plan "
            "and pricing for a suitable alternative."
        )


    payment_method = customer_data[
        "PaymentMethod"
    ].iloc[0]


    if payment_method == "Electronic check":

        recommendations.append(
            "Consider offering convenient "
            "automatic payment options."
        )


    internet_service = customer_data[
        "InternetService"
    ].iloc[0]


    if internet_service == "Fiber optic":

        recommendations.append(
            "Review the customer's internet plan "
            "and address service or pricing concerns."
        )


    addon_count = customer_data[
        "NumAddonServices"
    ].iloc[0]


    if addon_count <= 1:

        recommendations.append(
            "Consider relevant service bundles "
            "or additional services."
        )


    if len(recommendations) == 0:

        recommendations.append(
            "Continue normal customer engagement "
            "and monitor churn risk."
        )


    return recommendations



# ============================================================
# AI INSIGHT ENGINE
# ============================================================

def generate_dashboard_ai_insight(
    data,
    risk_data,
    probability_column,
    decision_threshold
):
    """
    Generate business-friendly insights from the existing
    customer data and ML churn scores.
    """

    insights = []
    actions = []

    if data is None or len(data) == 0:
        return (
            "There is not enough customer data for a detailed AI summary.",
            ["Adjust the dashboard filters to include customers."]
        )

    if "ChurnFlag" in data.columns:
        churn_rate_value = data["ChurnFlag"].mean() * 100
        insights.append(
            f"The selected customer segment has an observed churn rate "
            f"of {churn_rate_value:.1f}%."
        )

    if "Contract" in data.columns and "ChurnFlag" in data.columns:
        contract_rates = (
            data.groupby("Contract")["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )

        if len(contract_rates) > 0:
            top_contract = contract_rates.index[0]
            top_contract_rate = contract_rates.iloc[0] * 100

            insights.append(
                f"{top_contract} customers have the highest observed "
                f"churn rate among the selected contract groups "
                f"({top_contract_rate:.1f}%)."
            )

            if "Month-to-month" in contract_rates.index:
                if contract_rates["Month-to-month"] >= contract_rates.mean():
                    actions.append(
                        "Prioritize month-to-month customers for retention "
                        "offers or contract-upgrade campaigns."
                    )

    if "InternetService" in data.columns and "ChurnFlag" in data.columns:
        internet_rates = (
            data.groupby("InternetService")["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )

        if len(internet_rates) > 0:
            service = internet_rates.index[0]
            service_rate = internet_rates.iloc[0] * 100

            insights.append(
                f"{service} customers show the highest observed churn "
                f"rate among the selected internet-service groups "
                f"({service_rate:.1f}%)."
            )

    if "TenureBucket" in data.columns and "ChurnFlag" in data.columns:
        tenure_rates = (
            data.groupby("TenureBucket")["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )

        if len(tenure_rates) > 0:
            tenure_group = tenure_rates.index[0]
            tenure_rate = tenure_rates.iloc[0] * 100

            insights.append(
                f"The {tenure_group} tenure group has the highest "
                f"observed churn rate in the selected segment "
                f"({tenure_rate:.1f}%)."
            )

    if "PaymentMethod" in data.columns and "ChurnFlag" in data.columns:
        payment_rates = (
            data.groupby("PaymentMethod")["ChurnFlag"]
            .mean()
            .sort_values(ascending=False)
        )

        if len(payment_rates) > 0:
            payment = payment_rates.index[0]
            payment_rate = payment_rates.iloc[0] * 100

            insights.append(
                f"{payment} customers have the highest observed churn "
                f"rate among the selected payment methods "
                f"({payment_rate:.1f}%)."
            )

            if str(payment).lower() == "electronic check":
                actions.append(
                    "Review electronic-check customers for billing "
                    "friction and alternative payment options."
                )

    if (
        risk_data is not None
        and probability_column is not None
        and len(risk_data) > 0
    ):
        risk_count = len(risk_data)
        average_risk = risk_data[probability_column].mean() * 100

        insights.append(
            f"The filtered segment contains {risk_count:,} customers "
            f"above the model decision threshold, with an average "
            f"predicted churn probability of {average_risk:.1f}%."
        )

        actions.append(
            "Prioritize the highest-probability customers for proactive "
            "retention outreach."
        )
    else:
        actions.append(
            "Use the Customer Prediction page to investigate individual "
            "customers and their strongest churn drivers."
        )

    if not insights:
        insights.append(
            "The available fields do not provide enough information "
            "for a detailed churn summary."
        )

    if not actions:
        actions.append(
            "Continue monitoring churn patterns and investigate "
            "high-risk customers."
        )

    return " ".join(insights), actions


def generate_customer_ai_insight(
    probability,
    prediction,
    top_features,
    customer_data
):
    """
    Generate a customer-level explanation using the ML probability,
    SHAP drivers and customer attributes.
    """

    risk_label = get_risk_level(probability)

    explanation = (
        f"This customer is classified as {risk_label} risk "
        f"with a predicted churn probability of {probability:.1%}."
    )

    drivers = []

    if top_features is not None and len(top_features) > 0:
        for _, row in top_features.head(3).iterrows():
            feature = str(row["Feature"])
            value = float(row["SHAP Value"])

            direction = "increases" if value > 0 else "reduces"

            drivers.append(
                f"{feature} {direction} the model's churn score"
            )

    if drivers:
        explanation += (
            " The strongest model drivers are "
            + "; ".join(drivers)
            + "."
        )

    actions = []

    if prediction == 1:

        if (
            "Contract" in customer_data.columns
            and str(customer_data["Contract"].iloc[0])
            == "Month-to-month"
        ):
            actions.append(
                "Consider a targeted incentive for moving to a "
                "longer-term contract."
            )

        if (
            "tenure" in customer_data.columns
            and float(customer_data["tenure"].iloc[0]) <= 12
        ):
            actions.append(
                "Consider an early-tenure engagement or onboarding campaign."
            )

        if (
            "MonthlyCharges" in customer_data.columns
            and float(customer_data["MonthlyCharges"].iloc[0]) >= 70
        ):
            actions.append(
                "Review the customer's current package and perceived value."
            )

        if not actions:
            actions.append(
                "Prioritize proactive retention outreach and review "
                "the customer's strongest SHAP drivers."
            )

    else:
        actions.append(
            "Continue normal engagement while monitoring future "
            "churn risk."
        )

    return explanation, actions


# ============================================================
# RISK SEGMENTATION
# ============================================================

# These thresholds are business rules used to group predicted
# churn probabilities into actionable risk levels.
RISK_THRESHOLDS = {
    "Critical": 0.80,
    "High": 0.60,
    "Medium": 0.40,
    "Low": 0.00
}


def get_risk_level(probability):
    """Convert a churn probability into a business risk segment."""

    probability = float(probability)

    if probability >= RISK_THRESHOLDS["Critical"]:
        return "Critical"

    if probability >= RISK_THRESHOLDS["High"]:
        return "High"

    if probability >= RISK_THRESHOLDS["Medium"]:
        return "Medium"

    return "Low"


def add_risk_level(data, probability_column):
    """Return a copy of scored data with a RiskLevel column."""

    result = data.copy()

    if probability_column is not None and probability_column in result.columns:
        result["RiskLevel"] = result[probability_column].apply(
            get_risk_level
        )

    return result


def add_revenue_at_risk(data, probability_column):
    """Add an estimated monthly revenue-at-risk proxy.

    Formula: MonthlyCharges × predicted churn probability.
    This is an estimate, not actual future lost revenue.
    """

    result = data.copy()

    if (
        probability_column is not None
        and probability_column in result.columns
        and "MonthlyCharges" in result.columns
    ):
        result["RevenueAtRisk"] = (
            pd.to_numeric(result["MonthlyCharges"], errors="coerce")
            * pd.to_numeric(result[probability_column], errors="coerce")
        ).fillna(0)

    return result


def get_retention_action(row, probability_column):
    """Generate an explainable retention action from customer attributes."""

    probability = float(row.get(probability_column, 0))
    risk = str(row.get("RiskLevel", "Low"))
    actions = []

    if risk in ["Critical", "High"]:
        if str(row.get("Contract", "")) == "Month-to-month":
            actions.append("Discuss a longer-term contract option")

        if pd.to_numeric(row.get("tenure", 0), errors="coerce") <= 12:
            actions.append("Run an early-tenure engagement campaign")

        if pd.to_numeric(row.get("MonthlyCharges", 0), errors="coerce") >= 70:
            actions.append("Review plan value and pricing")

        if str(row.get("PaymentMethod", "")) == "Electronic check":
            actions.append("Review billing/payment experience")

        if not actions:
            actions.append("Proactive retention outreach")
    elif risk == "Medium":
        actions.append("Monitor risk and consider proactive engagement")
    else:
        actions.append("Continue normal engagement and monitor risk")

    return " • ".join(actions[:2])


def build_retention_priority(data, probability_column):
    """Create a transparent retention-priority queue."""

    result = add_risk_level(data, probability_column)
    result = add_revenue_at_risk(result, probability_column)

    if probability_column not in result.columns:
        return result

    # Priority score emphasizes both likelihood of churn and monthly revenue exposure.
    result["RetentionPriorityScore"] = (
        pd.to_numeric(result[probability_column], errors="coerce").fillna(0)
        * pd.to_numeric(result.get("RevenueAtRisk", 0), errors="coerce").fillna(0)
    )

    result["RecommendedAction"] = result.apply(
        lambda row: get_retention_action(row, probability_column),
        axis=1
    )

    return result.sort_values(
        "RetentionPriorityScore",
        ascending=False
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("Customer analytics & retention insights")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Customer Prediction"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.html(
        """
        <div class="dashboard-header">
            <div class="dashboard-header-title">CUSTOMER CHURN INTELLIGENCE</div>
            <div class="dashboard-header-subtitle">
                Customer retention, churn risk and machine learning insights
            </div>
        </div>
        """
)

    # --------------------------------------------------------
    # DASHBOARD DATA
    # --------------------------------------------------------

    total_customers = len(df)

    if "ChurnFlag" in df.columns:
        churned_customers = int(df["ChurnFlag"].sum())
        churn_rate = (churned_customers / total_customers * 100) if total_customers else 0
    else:
        churned_customers = 0
        churn_rate = 0

    probability_column = None

    if scored_df is not None:
        for column in [
            "ChurnProbability",
            "churn_probability",
            "churn_prob",
            "probability"
        ]:
            if column in scored_df.columns:
                probability_column = column
                break

    high_risk_customers = 0
    if probability_column is not None:
        high_risk_customers = int(
            (scored_df[probability_column] >= threshold).sum()
        )

    estimated_revenue_at_risk = 0.0

    if (
        scored_df is not None
        and probability_column is not None
        and "MonthlyCharges" in scored_df.columns
    ):
        revenue_df = add_revenue_at_risk(
            scored_df,
            probability_column
        )
        if "RevenueAtRisk" in revenue_df.columns:
            estimated_revenue_at_risk = float(
                revenue_df["RevenueAtRisk"].sum()
            )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.html(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-title">TOTAL CUSTOMERS</div>
                <div class="kpi-value">{total_customers:,}</div>
                <div class="kpi-description">Total customer base</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-title">CHURNED CUSTOMERS</div>
                <div class="kpi-value">{churned_customers:,}</div>
                <div class="kpi-description">Customers who churned</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-icon">🚨</div>
                <div class="kpi-title">HIGH-RISK CUSTOMERS</div>
                <div class="kpi-value">{high_risk_customers:,}</div>
                <div class="kpi-description">Probability ≥ decision threshold</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-icon">📉</div>
                <div class="kpi-title">CHURN RATE</div>
                <div class="kpi-value">{churn_rate:.1f}%</div>
                <div class="kpi-description">Overall observed churn</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-title">REVENUE AT RISK</div>
                <div class="kpi-value">${estimated_revenue_at_risk:,.0f}</div>
                <div class="kpi-description">Estimated monthly revenue exposure</div>
            </div>
        </div>
        """
)

    st.markdown('<div class="dashboard-spacer"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.html(
        """
        <div class="filter-heading">
            🎛️ CUSTOMER FILTERS
        </div>
        """
)

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        contract_filter = st.multiselect(
            "Contract",
            sorted(df["Contract"].dropna().unique().tolist()) if "Contract" in df.columns else [],
            placeholder="All contracts"
        )

    with filter_col2:
        internet_filter = st.multiselect(
            "Internet Service",
            sorted(df["InternetService"].dropna().unique().tolist()) if "InternetService" in df.columns else [],
            placeholder="All services"
        )

    with filter_col3:
        tenure_filter = st.multiselect(
            "Tenure Bucket",
            sorted(df["TenureBucket"].dropna().unique().tolist()) if "TenureBucket" in df.columns else [],
            placeholder="All tenure groups"
        )

    with filter_col4:
        customer_search = st.text_input(
            "Customer ID",
            placeholder="Search customer ID..."
        )

    filtered_df = df.copy()

    if contract_filter:
        filtered_df = filtered_df[filtered_df["Contract"].isin(contract_filter)]

    if internet_filter:
        filtered_df = filtered_df[filtered_df["InternetService"].isin(internet_filter)]

    if tenure_filter:
        filtered_df = filtered_df[filtered_df["TenureBucket"].isin(tenure_filter)]

    if customer_search.strip() and "customerID" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["customerID"].astype(str).str.contains(
                customer_search.strip(),
                case=False,
                na=False
            )
        ]

    st.caption(f"Showing {len(filtered_df):,} customer(s) based on the selected filters.")


    # --------------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------------

    # Apply the same dashboard filters to the scored customer table.
    filtered_scored_df = (
        scored_df.copy()
        if scored_df is not None
        else None
    )

    if filtered_scored_df is not None:

        if contract_filter and "Contract" in filtered_scored_df.columns:
            filtered_scored_df = filtered_scored_df[
                filtered_scored_df["Contract"].isin(contract_filter)
            ]

        if internet_filter and "InternetService" in filtered_scored_df.columns:
            filtered_scored_df = filtered_scored_df[
                filtered_scored_df["InternetService"].isin(internet_filter)
            ]

        if tenure_filter and "TenureBucket" in filtered_scored_df.columns:
            filtered_scored_df = filtered_scored_df[
                filtered_scored_df["TenureBucket"].isin(tenure_filter)
            ]

        if (
            customer_search.strip()
            and "customerID" in filtered_scored_df.columns
        ):
            filtered_scored_df = filtered_scored_df[
                filtered_scored_df["customerID"].astype(str).str.contains(
                    customer_search.strip(),
                    case=False,
                    na=False
                )
            ]


    summary_customers = len(filtered_df)

    if "ChurnFlag" in filtered_df.columns and summary_customers > 0:
        summary_churned = int(
            filtered_df["ChurnFlag"].sum()
        )
        summary_churn_rate = (
            summary_churned / summary_customers * 100
        )
    else:
        summary_churned = 0
        summary_churn_rate = 0.0


    summary_high_risk = 0
    summary_revenue_at_risk = 0.0

    if (
        filtered_scored_df is not None
        and probability_column is not None
    ):

        summary_high_risk = int(
            (
                filtered_scored_df[probability_column]
                >= threshold
            ).sum()
        )

        if "MonthlyCharges" in filtered_scored_df.columns:

            summary_revenue_df = add_revenue_at_risk(
                filtered_scored_df,
                probability_column
            )

            summary_revenue_at_risk = float(
                summary_revenue_df["RevenueAtRisk"].sum()
            )


    st.html(
        """
        <div class="section-header">
            📋 EXECUTIVE SUMMARY
        </div>
        <div class="section-description">
            A business-level snapshot of the currently selected customer segment.
        </div>
        """
    )


    st.html(
        f"""
        <div class="executive-grid">

            <div class="executive-card">
                <div class="executive-label">CUSTOMERS IN VIEW</div>
                <div class="executive-value">
                    {summary_customers:,}
                </div>
                <div class="executive-note">
                    Current filtered customer segment
                </div>
            </div>


            <div class="executive-card">
                <div class="executive-label">OBSERVED CHURN RATE</div>
                <div class="executive-value">
                    {summary_churn_rate:.1f}%
                </div>
                <div class="executive-note">
                    {summary_churned:,} observed churned customers
                </div>
            </div>


            <div class="executive-card">
                <div class="executive-label">HIGH-RISK CUSTOMERS</div>
                <div class="executive-value">
                    {summary_high_risk:,}
                </div>
                <div class="executive-note">
                    Probability ≥ {float(threshold):.3f}
                </div>
            </div>


            <div class="executive-card">
                <div class="executive-label">EST. REVENUE AT RISK</div>
                <div class="executive-value">
                    ${summary_revenue_at_risk:,.0f}
                </div>
                <div class="executive-note">
                    Estimated monthly exposure
                </div>
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # RISK SEGMENTATION
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            🎯 CHURN RISK SEGMENTATION
        </div>
        <div class="section-description">
            Predicted churn probability grouped into configurable business risk levels.
        </div>
        """
    )

    if scored_df is not None and probability_column is not None:

        segmented_df = scored_df.copy()

        if contract_filter and "Contract" in segmented_df.columns:
            segmented_df = segmented_df[
                segmented_df["Contract"].isin(contract_filter)
            ]

        if internet_filter and "InternetService" in segmented_df.columns:
            segmented_df = segmented_df[
                segmented_df["InternetService"].isin(internet_filter)
            ]

        if tenure_filter and "TenureBucket" in segmented_df.columns:
            segmented_df = segmented_df[
                segmented_df["TenureBucket"].isin(tenure_filter)
            ]

        if customer_search.strip() and "customerID" in segmented_df.columns:
            segmented_df = segmented_df[
                segmented_df["customerID"].astype(str).str.contains(
                    customer_search.strip(),
                    case=False,
                    na=False
                )
            ]

        segmented_df = add_risk_level(
            segmented_df,
            probability_column
        )

        risk_counts = segmented_df["RiskLevel"].value_counts()

        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

        with risk_col1:
            st.metric(
                "🔴 Critical",
                f"{int(risk_counts.get('Critical', 0)):,}",
                help="Predicted churn probability ≥ 80%."
            )

        with risk_col2:
            st.metric(
                "🟠 High",
                f"{int(risk_counts.get('High', 0)):,}",
                help="Predicted churn probability from 60% to below 80%."
            )

        with risk_col3:
            st.metric(
                "🟡 Medium",
                f"{int(risk_counts.get('Medium', 0)):,}",
                help="Predicted churn probability from 40% to below 60%."
            )

        with risk_col4:
            st.metric(
                "🟢 Low",
                f"{int(risk_counts.get('Low', 0)):,}",
                help="Predicted churn probability below 40%."
            )

        st.caption(
            "Risk thresholds: Critical ≥ 80% | High 60–79% | "
            "Medium 40–59% | Low < 40%. These are configurable business rules."
        )

    else:
        st.info(
            "Risk segmentation requires customers_scored.csv with a churn probability column."
        )


    # --------------------------------------------------------
    # MODEL PERFORMANCE & MONITORING
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            📈 MODEL PERFORMANCE & MONITORING
        </div>
        <div class="section-description">
            Evaluation metrics and scoring information recorded with the trained churn model.
        </div>
        """
    )

    # Read evaluation values from churn_model_metadata.json.
    # No model-performance values are hard-coded here.
    model_name = metadata.get(
        "model_name",
        classifier.__class__.__name__
    )

    roc_auc = metadata.get("test_roc_auc")
    pr_auc = metadata.get("test_pr_auc")
    decision_threshold = metadata.get(
        "decision_threshold",
        threshold
    )

    customers_scored = (
        len(scored_df)
        if scored_df is not None
        else 0
    )

    performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)

    with performance_col1:
        if roc_auc is not None:
            st.metric(
                "ROC-AUC",
                f"{float(roc_auc):.3f}"
            )
        else:
            st.metric(
                "ROC-AUC",
                "N/A"
            )

    with performance_col2:
        if pr_auc is not None:
            st.metric(
                "PR-AUC",
                f"{float(pr_auc):.3f}"
            )
        else:
            st.metric(
                "PR-AUC",
                "N/A"
            )

    with performance_col3:
        st.metric(
            "Decision Threshold",
            f"{float(decision_threshold):.3f}"
        )

    with performance_col4:
        st.metric(
            "Customers Scored",
            f"{customers_scored:,}"
        )

    # --------------------------------------------------------
    # RISK DISTRIBUTION + MODEL INFORMATION
    # --------------------------------------------------------

    performance_col1, performance_col2 = st.columns(
        [1.15, 1],
        gap="large"
    )

    if scored_df is not None and probability_column is not None:

        performance_risk_df = add_risk_level(
            scored_df,
            probability_column
        )

        risk_order = [
            "Low",
            "Medium",
            "High",
            "Critical"
        ]

        risk_distribution = (
            performance_risk_df["RiskLevel"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
        )

        total_risk_customers = int(
            risk_distribution.sum()
        )

    else:

        risk_distribution = pd.Series(
            [0, 0, 0, 0],
            index=[
                "Low",
                "Medium",
                "High",
                "Critical"
            ]
        )

        total_risk_customers = 0


    # --------------------------------------------------------
    # RISK DISTRIBUTION CARD
    # --------------------------------------------------------

    with performance_col1:

        risk_rows = ""

        risk_styles = {
            "Low": "risk-low",
            "Medium": "risk-medium",
            "High": "risk-high",
            "Critical": "risk-critical"
        }

        for risk_name in [
            "Low",
            "Medium",
            "High",
            "Critical"
        ]:

            count = int(
                risk_distribution.get(
                    risk_name,
                    0
                )
            )

            percentage = (
                count / total_risk_customers * 100
                if total_risk_customers > 0
                else 0
            )

            risk_rows += f"""
                <div class="risk-row">
                    <div class="risk-row-top">
                        <div class="risk-name">
                            <span class="risk-dot {risk_styles[risk_name]}"></span>
                            {risk_name}
                        </div>

                        <div class="risk-count">
                            {count:,}
                            <span>{percentage:.1f}%</span>
                        </div>
                    </div>

                    <div class="risk-track">
                        <div
                            class="risk-fill {risk_styles[risk_name]}"
                            style="width:{percentage:.1f}%"
                        ></div>
                    </div>
                </div>
            """

        st.html(
            f"""
            <div class="phase4-card risk-distribution-card">

                <div class="phase4-card-header">
                    <div>
                        <div class="phase4-card-title">
                            🎯 Risk Distribution
                        </div>

                        <div class="phase4-card-subtitle">
                            Customers grouped by predicted churn risk
                        </div>
                    </div>

                    <div class="phase4-total">
                        <strong>{total_risk_customers:,}</strong>
                        <span>customers</span>
                    </div>
                </div>

                <div class="risk-list">
                    {risk_rows}
                </div>

                <div class="risk-threshold-note">
                    Critical ≥ 80% &nbsp;•&nbsp;
                    High 60–79% &nbsp;•&nbsp;
                    Medium 40–59% &nbsp;•&nbsp;
                    Low &lt; 40%
                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # MODEL INFORMATION CARD
    # --------------------------------------------------------

    with performance_col2:

        st.html(
            f"""
            <div class="phase4-card model-information-card">

                <div class="phase4-card-header">
                    <div>
                        <div class="phase4-card-title">
                            🧾 Model Information
                        </div>

                        <div class="phase4-card-subtitle">
                            Configuration used by the application
                        </div>
                    </div>

                    <div class="model-badge">
                        ML
                    </div>
                </div>


                <div class="model-detail-grid">

                    <div class="model-detail">
                        <div class="model-detail-label">
                            MODEL
                        </div>

                        <div class="model-detail-value">
                            {model_name}
                        </div>
                    </div>


                    <div class="model-detail">
                        <div class="model-detail-label">
                            DECISION THRESHOLD
                        </div>

                        <div class="model-detail-value">
                            {float(decision_threshold):.3f}
                        </div>
                    </div>


                    <div class="model-detail">
                        <div class="model-detail-label">
                            CUSTOMERS SCORED
                        </div>

                        <div class="model-detail-value">
                            {customers_scored:,}
                        </div>
                    </div>

                </div>


                <div class="model-explanation">

                    <div class="model-explanation-title">
                        How the decision works
                    </div>

                    <div class="model-explanation-text">
                        The model produces a churn probability for each
                        customer. The configured threshold of
                        <strong>{float(decision_threshold):.3f}</strong>
                        is then used to convert that probability into the
                        model's churn-risk decision.
                    </div>

                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # REVENUE AT RISK
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            💰 REVENUE AT RISK
        </div>
        <div class="section-description">
            Estimated monthly revenue exposure calculated as Monthly Charges × predicted churn probability.
            This is a risk proxy, not a forecast of actual lost revenue.
        </div>
        """
    )

    if (
        scored_df is not None
        and probability_column is not None
        and "MonthlyCharges" in scored_df.columns
    ):
        revenue_risk_df = scored_df.copy()

        if contract_filter and "Contract" in revenue_risk_df.columns:
            revenue_risk_df = revenue_risk_df[
                revenue_risk_df["Contract"].isin(contract_filter)
            ]

        if internet_filter and "InternetService" in revenue_risk_df.columns:
            revenue_risk_df = revenue_risk_df[
                revenue_risk_df["InternetService"].isin(internet_filter)
            ]

        if tenure_filter and "TenureBucket" in revenue_risk_df.columns:
            revenue_risk_df = revenue_risk_df[
                revenue_risk_df["TenureBucket"].isin(tenure_filter)
            ]

        if customer_search.strip() and "customerID" in revenue_risk_df.columns:
            revenue_risk_df = revenue_risk_df[
                revenue_risk_df["customerID"].astype(str).str.contains(
                    customer_search.strip(),
                    case=False,
                    na=False
                )
            ]

        revenue_risk_df = add_risk_level(
            revenue_risk_df,
            probability_column
        )
        revenue_risk_df = add_revenue_at_risk(
            revenue_risk_df,
            probability_column
        )

        if "RevenueAtRisk" in revenue_risk_df.columns:
            revenue_col1, revenue_col2 = st.columns(2)

            with revenue_col1:
                st.metric(
                    "💰 Estimated Monthly Revenue at Risk",
                    f"${revenue_risk_df['RevenueAtRisk'].sum():,.2f}"
                )

            with revenue_col2:
                customers_with_risk = int(
                    (revenue_risk_df[probability_column] >= threshold).sum()
                )
                st.metric(
                    "🚨 Customers Above Model Threshold",
                    f"{customers_with_risk:,}"
                )

            revenue_display_columns = [
                "customerID",
                "RiskLevel",
                probability_column,
                "MonthlyCharges",
                "RevenueAtRisk"
            ]

            revenue_display_columns = [
                column for column in revenue_display_columns
                if column in revenue_risk_df.columns
            ]

            revenue_display = (
                revenue_risk_df[revenue_display_columns]
                .sort_values("RevenueAtRisk", ascending=False)
                .head(20)
                .copy()
            )

            if probability_column in revenue_display.columns:
                revenue_display[probability_column] = (
                    revenue_display[probability_column] * 100
                ).round(1).astype(str) + "%"

            if "MonthlyCharges" in revenue_display.columns:
                revenue_display["MonthlyCharges"] = (
                    revenue_display["MonthlyCharges"]
                    .round(2)
                    .map(lambda value: f"${value:,.2f}")
                )

            if "RevenueAtRisk" in revenue_display.columns:
                revenue_display["RevenueAtRisk"] = (
                    revenue_display["RevenueAtRisk"]
                    .round(2)
                    .map(lambda value: f"${value:,.2f}")
                )

            st.dataframe(
                revenue_display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Revenue-at-Risk Report",
                data=revenue_risk_df
                .sort_values("RevenueAtRisk", ascending=False)
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="revenue_at_risk_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.info(
            "Revenue-at-risk analysis requires customers_scored.csv with "
            "ChurnProbability and MonthlyCharges columns."
        )

    # --------------------------------------------------------
    # RETENTION PRIORITY QUEUE
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            🎯 RETENTION PRIORITY QUEUE
        </div>
        <div class="section-description">
            Customers are ordered using a transparent priority score based on predicted churn probability and estimated monthly revenue at risk.
        </div>
        """
    )

    if scored_df is not None and probability_column is not None:

        priority_df = scored_df.copy()

        if contract_filter and "Contract" in priority_df.columns:
            priority_df = priority_df[priority_df["Contract"].isin(contract_filter)]

        if internet_filter and "InternetService" in priority_df.columns:
            priority_df = priority_df[priority_df["InternetService"].isin(internet_filter)]

        if tenure_filter and "TenureBucket" in priority_df.columns:
            priority_df = priority_df[priority_df["TenureBucket"].isin(tenure_filter)]

        if customer_search.strip() and "customerID" in priority_df.columns:
            priority_df = priority_df[
                priority_df["customerID"].astype(str).str.contains(
                    customer_search.strip(), case=False, na=False
                )
            ]

        priority_df = build_retention_priority(
            priority_df, probability_column
        )

        priority_df = priority_df[
            priority_df[probability_column] >= RISK_THRESHOLDS["Medium"]
        ].copy()

        if len(priority_df) > 0:
            priority_display_columns = [
                "customerID",
                "RiskLevel",
                probability_column,
                "MonthlyCharges",
                "RevenueAtRisk",
                "RecommendedAction"
            ]

            priority_display_columns = [
                column for column in priority_display_columns
                if column in priority_df.columns
            ]

            priority_display = priority_df[priority_display_columns].head(20).copy()

            if probability_column in priority_display.columns:
                priority_display[probability_column] = (
                    priority_display[probability_column] * 100
                ).round(1).astype(str) + "%"

            if "MonthlyCharges" in priority_display.columns:
                priority_display["MonthlyCharges"] = (
                    priority_display["MonthlyCharges"]
                    .round(2)
                    .map(lambda value: f"${value:,.2f}")
                )

            if "RevenueAtRisk" in priority_display.columns:
                priority_display["RevenueAtRisk"] = (
                    priority_display["RevenueAtRisk"]
                    .round(2)
                    .map(lambda value: f"${value:,.2f}")
                )

            st.dataframe(
                priority_display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Retention Priority Report",
                data=priority_df.to_csv(index=False).encode("utf-8"),
                file_name="retention_priority_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No Medium-or-higher risk customers match the selected filters.")

    else:
        st.info(
            "Retention priority analysis requires customers_scored.csv with a churn probability column."
        )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.html(
            """
            <div class="chart-title">📄 CHURN BY CONTRACT</div>
            <div class="chart-subtitle">Observed churn rate by contract type</div>
            """
)

        if "Contract" in filtered_df.columns and "ChurnFlag" in filtered_df.columns and len(filtered_df) > 0:
            contract_churn = (
                filtered_df.groupby("Contract")["ChurnFlag"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            st.bar_chart(contract_churn, height=260)
        else:
            st.info("No data available for the selected filters.")

    with chart_col2:
        st.html(
            """
            <div class="chart-title">🌐 CHURN BY INTERNET SERVICE</div>
            <div class="chart-subtitle">Observed churn rate by internet service</div>
            """
)

        if "InternetService" in filtered_df.columns and "ChurnFlag" in filtered_df.columns and len(filtered_df) > 0:
            internet_churn = (
                filtered_df.groupby("InternetService")["ChurnFlag"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            st.bar_chart(internet_churn, height=260)
        else:
            st.info("No data available for the selected filters.")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.html(
            """
            <div class="chart-title">💳 CHURN BY PAYMENT METHOD</div>
            <div class="chart-subtitle">Observed churn rate by payment method</div>
            """
)

        if "PaymentMethod" in filtered_df.columns and "ChurnFlag" in filtered_df.columns and len(filtered_df) > 0:
            payment_churn = (
                filtered_df.groupby("PaymentMethod")["ChurnFlag"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )
            st.bar_chart(payment_churn, height=260)
        else:
            st.info("No data available for the selected filters.")

    with chart_col4:
        st.html(
            """
            <div class="chart-title">⏳ CHURN BY TENURE</div>
            <div class="chart-subtitle">Observed churn rate by customer tenure</div>
            """
)

        if "TenureBucket" in filtered_df.columns and "ChurnFlag" in filtered_df.columns and len(filtered_df) > 0:
            tenure_churn = (
                filtered_df.groupby("TenureBucket")["ChurnFlag"]
                .mean()
                .mul(100)
            )
            st.bar_chart(tenure_churn, height=260)
        else:
            st.info("No data available for the selected filters.")

    # --------------------------------------------------------
    # CHURN DRIVER ANALYSIS
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            🧠 CHURN DRIVER ANALYSIS
        </div>
        <div class="section-description">
            Observed churn patterns across key customer characteristics.
            These are associations in the available data, not causal effects.
        </div>
        """
    )


    def build_churn_pattern_rows(
        data,
        group_column,
        display_limit=None
    ):

        if (
            group_column not in data.columns
            or "ChurnFlag" not in data.columns
            or len(data) == 0
        ):
            return "", 0

        grouped = (
            data.groupby(
                group_column,
                dropna=False
            )["ChurnFlag"]
            .agg(
                ["mean", "count"]
            )
            .reset_index()
        )

        grouped["ChurnRate"] = (
            grouped["mean"] * 100
        )

        grouped = grouped.drop(
            columns=["mean"]
        )

        grouped[group_column] = (
            grouped[group_column]
            .fillna("Unknown")
            .astype(str)
        )

        if display_limit is not None:
            grouped = grouped.head(display_limit)

        if len(grouped) == 0:
            return "", 0

        max_rate = max(
            float(grouped["ChurnRate"].max()),
            1.0
        )

        rows = ""

        for _, row in grouped.iterrows():

            label = str(
                row[group_column]
            )

            rate = float(
                row["ChurnRate"]
            )

            count = int(
                row["count"]
            )

            bar_width = (
                rate / max_rate * 100
            )

            rows += f"""
                <div class="driver-row">

                    <div class="driver-row-top">

                        <div class="driver-label">
                            {label}
                        </div>

                        <div class="driver-value">
                            {rate:.1f}%
                            <span>
                                ({count:,})
                            </span>
                        </div>

                    </div>

                    <div class="driver-track">
                        <div
                            class="driver-fill"
                            style="width:{bar_width:.1f}%"
                        ></div>
                    </div>

                </div>
            """

        return rows, len(grouped)


    # Contract
    contract_rows, contract_count = (
        build_churn_pattern_rows(
            filtered_df,
            "Contract"
        )
    )


    # Internet service
    internet_rows, internet_count = (
        build_churn_pattern_rows(
            filtered_df,
            "InternetService"
        )
    )


    # Tenure
    tenure_driver_df = filtered_df.copy()

    if "TenureBucket" in tenure_driver_df.columns:

        tenure_order = (
            tenure_driver_df["TenureBucket"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        # Keep the dataset's existing groups, while putting
        # numeric range labels in a natural order when possible.
        import re

        def tenure_sort_key(value):

            match = re.search(
                r"\d+",
                str(value)
            )

            if match:
                return (
                    0,
                    int(match.group())
                )

            return (
                1,
                str(value)
            )

        tenure_order = sorted(
            tenure_order,
            key=tenure_sort_key
        )

        tenure_driver_df["TenureBucket"] = (
            tenure_driver_df["TenureBucket"]
            .astype(str)
        )

        tenure_grouped = (
            tenure_driver_df.groupby(
                "TenureBucket"
            )["ChurnFlag"]
            .agg(["mean", "count"])
            .reindex(tenure_order)
            .dropna(
                subset=["mean"]
            )
            .reset_index()
        )

        tenure_grouped["ChurnRate"] = (
            tenure_grouped["mean"] * 100
        )

        tenure_max = max(
            float(
                tenure_grouped["ChurnRate"].max()
            )
            if len(tenure_grouped) > 0
            else 0,
            1.0
        )

        tenure_rows = ""

        for _, row in tenure_grouped.iterrows():

            rate = float(
                row["ChurnRate"]
            )

            count = int(
                row["count"]
            )

            bar_width = (
                rate / tenure_max * 100
            )

            tenure_rows += f"""
                <div class="driver-row">

                    <div class="driver-row-top">

                        <div class="driver-label">
                            {str(row["TenureBucket"])}
                        </div>

                        <div class="driver-value">
                            {rate:.1f}%
                            <span>
                                ({count:,})
                            </span>
                        </div>

                    </div>

                    <div class="driver-track">
                        <div
                            class="driver-fill"
                            style="width:{bar_width:.1f}%"
                        ></div>
                    </div>

                </div>
            """

    else:
        tenure_rows = ""


    st.html(
        f"""
        <div class="driver-grid">

            <div class="driver-card">

                <div class="driver-card-title">
                    📄 Contract
                </div>

                <div class="driver-card-subtitle">
                    Observed churn rate by contract type
                </div>

                <div class="driver-list">
                    {contract_rows}
                </div>

            </div>


            <div class="driver-card">

                <div class="driver-card-title">
                    🌐 Internet Service
                </div>

                <div class="driver-card-subtitle">
                    Observed churn rate by service type
                </div>

                <div class="driver-list">
                    {internet_rows}
                </div>

            </div>


            <div class="driver-card">

                <div class="driver-card-title">
                    ⏳ Tenure
                </div>

                <div class="driver-card-subtitle">
                    Observed churn rate across tenure groups
                </div>

                <div class="driver-list">
                    {tenure_rows}
                </div>

            </div>

        </div>
        """
    )


    # Monthly charges is handled separately because it is numeric.
    if (
        "MonthlyCharges" in filtered_df.columns
        and "ChurnFlag" in filtered_df.columns
        and len(filtered_df) > 0
    ):

        charge_data = filtered_df.copy()

        charge_data["ChargeBand"] = pd.cut(
            pd.to_numeric(
                charge_data["MonthlyCharges"],
                errors="coerce"
            ),
            bins=[
                -np.inf,
                30,
                50,
                70,
                90,
                np.inf
            ],
            labels=[
                "≤ $30",
                "$30–50",
                "$50–70",
                "$70–90",
                "> $90"
            ]
        )

        charge_grouped = (
            charge_data.groupby(
                "ChargeBand",
                observed=False
            )["ChurnFlag"]
            .agg(["mean", "count"])
            .reset_index()
        )

        charge_grouped["ChurnRate"] = (
            charge_grouped["mean"] * 100
        )

        charge_grouped = charge_grouped[
            charge_grouped["count"] > 0
        ]

        charge_max = max(
            float(
                charge_grouped["ChurnRate"].max()
            )
            if len(charge_grouped) > 0
            else 0,
            1.0
        )

        charge_rows = ""

        for _, row in charge_grouped.iterrows():

            rate = float(
                row["ChurnRate"]
            )

            count = int(
                row["count"]
            )

            bar_width = (
                rate / charge_max * 100
            )

            charge_rows += f"""
                <div class="driver-row">

                    <div class="driver-row-top">

                        <div class="driver-label">
                            {str(row["ChargeBand"])}
                        </div>

                        <div class="driver-value">
                            {rate:.1f}%
                            <span>
                                ({count:,})
                            </span>
                        </div>

                    </div>

                    <div class="driver-track">
                        <div
                            class="driver-fill"
                            style="width:{bar_width:.1f}%"
                        ></div>
                    </div>

                </div>
            """

        st.html(
            f"""
            <div class="driver-card driver-card-wide">

                <div class="driver-card-title">
                    💰 Monthly Charges
                </div>

                <div class="driver-card-subtitle">
                    Observed churn rate across monthly charge bands
                </div>

                <div class="driver-list charge-list">
                    {charge_rows}
                </div>

            </div>
            """
        )


    st.html(
        """
        <div class="driver-footnote">
            <strong>Interpretation:</strong>
            A higher observed churn rate means a larger share of customers
            in that group churned in the historical dataset. It does not
            by itself establish that the characteristic caused churn.
        </div>
        """
    )


    # --------------------------------------------------------
    # PHASE 5.3 - RETENTION ACTION CENTER
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            🎯 RETENTION ACTION CENTER
        </div>
        <div class="section-description">
            A practical action queue connecting customer risk, revenue exposure,
            the main business reason and a suggested retention action.
        </div>
        """
    )

    if scored_df is not None and probability_column is not None:

        action_df = scored_df.copy()

        if contract_filter and "Contract" in action_df.columns:
            action_df = action_df[action_df["Contract"].isin(contract_filter)]

        if internet_filter and "InternetService" in action_df.columns:
            action_df = action_df[action_df["InternetService"].isin(internet_filter)]

        if tenure_filter and "TenureBucket" in action_df.columns:
            action_df = action_df[action_df["TenureBucket"].isin(tenure_filter)]

        if customer_search.strip() and "customerID" in action_df.columns:
            action_df = action_df[
                action_df["customerID"].astype(str).str.contains(
                    customer_search.strip(), case=False, na=False
                )
            ]

        action_df = add_risk_level(action_df, probability_column)
        action_df = add_revenue_at_risk(action_df, probability_column)

        def get_primary_reason(row):
            risk = str(row.get("RiskLevel", "Low"))
            if risk in ["Critical", "High"] and str(row.get("Contract", "")) == "Month-to-month":
                return "Month-to-month contract"
            if risk in ["Critical", "High"] and pd.to_numeric(row.get("tenure", 0), errors="coerce") <= 12:
                return "Short customer tenure"
            if risk in ["Critical", "High"] and pd.to_numeric(row.get("MonthlyCharges", 0), errors="coerce") >= 70:
                return "High monthly charges"
            if risk in ["Critical", "High"] and str(row.get("PaymentMethod", "")) == "Electronic check":
                return "Electronic-check payment method"
            if risk == "Medium":
                return "Moderate predicted churn risk"
            return "Lower predicted churn risk"

        action_df["PrimaryReason"] = action_df.apply(get_primary_reason, axis=1)
        action_df["SuggestedAction"] = action_df.apply(
            lambda row: get_retention_action(row, probability_column), axis=1
        )

        action_df = action_df[
            action_df[probability_column] >= RISK_THRESHOLDS["Medium"]
        ].copy()

        action_df = action_df.sort_values(
            "RetentionPriorityScore" if "RetentionPriorityScore" in action_df.columns else probability_column,
            ascending=False
        )

        if "RetentionPriorityScore" not in action_df.columns:
            action_df["RetentionPriorityScore"] = (
                pd.to_numeric(action_df[probability_column], errors="coerce").fillna(0)
                * pd.to_numeric(action_df.get("RevenueAtRisk", 0), errors="coerce").fillna(0)
            )
            action_df = action_df.sort_values("RetentionPriorityScore", ascending=False)

        if len(action_df) > 0:
            action_display = action_df[
                [
                    "customerID", "RiskLevel", probability_column,
                    "PrimaryReason", "RevenueAtRisk", "SuggestedAction"
                ]
            ].head(25).copy()

            action_display[probability_column] = (
                action_display[probability_column] * 100
            ).round(1).astype(str) + "%"
            action_display["RevenueAtRisk"] = action_display["RevenueAtRisk"].round(2).map(
                lambda value: f"${value:,.2f}"
            )

            st.dataframe(
                action_display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Retention Action Center",
                data=action_df.to_csv(index=False).encode("utf-8"),
                file_name="retention_action_center.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success("No Medium-or-higher risk customers match the selected filters.")
    else:
        st.info("Retention Action Center requires customers_scored.csv with a churn probability column.")


    # --------------------------------------------------------
    # PHASE 5.4 - CUSTOMER 360 VIEW
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            👤 CUSTOMER 360 VIEW
        </div>
        <div class="section-description">
            A single-customer profile combining churn risk, financial exposure,
            contract, services and recommended retention action.
        </div>
        """
    )

    if scored_df is not None and probability_column is not None and "customerID" in scored_df.columns:

        customer_options = sorted(
            scored_df["customerID"].dropna().astype(str).unique().tolist()
        )

        c360_customer = st.selectbox(
            "Select Customer",
            customer_options,
            key="customer_360_selector"
        )

        c360 = scored_df[
            scored_df["customerID"].astype(str) == str(c360_customer)
        ].copy()

        if len(c360) > 0:
            c360 = add_risk_level(c360, probability_column)
            c360 = add_revenue_at_risk(c360, probability_column)
            c360_row = c360.iloc[0]

            c360_probability = float(c360_row[probability_column])
            c360_risk = str(c360_row.get("RiskLevel", get_risk_level(c360_probability)))
            c360_revenue = float(c360_row.get("RevenueAtRisk", 0))
            c360_action = get_retention_action(c360_row, probability_column)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Churn Probability", f"{c360_probability:.1%}")
            with c2:
                st.metric("Risk Level", c360_risk)
            with c3:
                st.metric("Monthly Charges", f"${float(c360_row.get('MonthlyCharges', 0)):,.2f}")
            with c4:
                st.metric("Revenue at Risk", f"${c360_revenue:,.2f}")

            profile_col1, profile_col2 = st.columns(2)

            with profile_col1:
                profile_data = {
                    "Customer ID": c360_customer,
                    "Contract": c360_row.get("Contract", "N/A"),
                    "Tenure": c360_row.get("tenure", "N/A"),
                    "Internet Service": c360_row.get("InternetService", "N/A"),
                    "Payment Method": c360_row.get("PaymentMethod", "N/A"),
                    "Tenure Bucket": c360_row.get("TenureBucket", "N/A")
                }
                st.dataframe(
                    pd.DataFrame(profile_data.items(), columns=["Profile", "Value"]),
                    use_container_width=True,
                    hide_index=True
                )

            with profile_col2:
                service_columns = [
                    "PhoneService", "MultipleLines", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"
                ]
                service_rows = []
                for column in service_columns:
                    if column in c360_row.index:
                        service_rows.append({
                            "Service": column,
                            "Status": c360_row[column]
                        })
                st.dataframe(
                    pd.DataFrame(service_rows),
                    use_container_width=True,
                    hide_index=True
                )

            st.html(
                f"""
                <div class="c360-action-card">
                    <div class="c360-action-title">💡 Recommended Retention Action</div>
                    <div class="c360-action-text">{c360_action}</div>
                </div>
                """
            )

            st.download_button(
                "📥 Download Customer 360 Profile",
                data=c360.to_csv(index=False).encode("utf-8"),
                file_name=f"customer_360_{c360_customer}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Customer 360 requires customers_scored.csv with customerID and churn probability columns.")


    # --------------------------------------------------------
    # PHASE 5.5 - PROFESSIONAL EXPORT & REPORTING
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            📊 PROFESSIONAL EXPORT & REPORTING
        </div>
        <div class="section-description">
            Export the current analysis as a compact reporting package for review,
            retention planning and presentation use.
        </div>
        """
    )

    import io
    import zipfile

    report_files = {}

    report_files["executive_summary.csv"] = pd.DataFrame({
        "Metric": [
            "Customers in View",
            "Observed Churn Rate",
            "High-Risk Customers",
            "Estimated Revenue at Risk"
        ],
        "Value": [
            summary_customers,
            f"{summary_churn_rate:.2f}%",
            summary_high_risk,
            f"${summary_revenue_at_risk:,.2f}"
        ]
    }).to_csv(index=False)

    report_files["filtered_customers.csv"] = filtered_df.to_csv(index=False)

    if scored_df is not None and probability_column is not None:
        report_scored = add_risk_level(scored_df.copy(), probability_column)
        report_scored = add_revenue_at_risk(report_scored, probability_column)
        report_scored["RecommendedAction"] = report_scored.apply(
            lambda row: get_retention_action(row, probability_column), axis=1
        )
        report_files["scored_customers.csv"] = report_scored.to_csv(index=False)

        report_priority = build_retention_priority(
            scored_df.copy(), probability_column
        )
        report_files["retention_priority.csv"] = report_priority.to_csv(index=False)

        report_action = report_priority.copy()
        report_action["PrimaryReason"] = report_action.apply(get_primary_reason, axis=1)
        report_action["SuggestedAction"] = report_action["RecommendedAction"]
        report_files["retention_action_center.csv"] = report_action.to_csv(index=False)

        report_high_risk = report_scored[
            report_scored[probability_column] >= threshold
        ].sort_values(probability_column, ascending=False)
        report_files["high_risk_customers.csv"] = report_high_risk.to_csv(index=False)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in report_files.items():
            zip_file.writestr(filename, content)

    zip_buffer.seek(0)

    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.metric("Report Files", len(report_files))

    with report_col2:
        st.metric("Customers Included", len(filtered_df))

    st.download_button(
        "📦 Download Complete Reporting Package",
        data=zip_buffer.getvalue(),
        file_name="customer_churn_reporting_package.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.caption(
        "The package contains CSV reports for executive summary, filtered customers, "
        "scored customers, risk/retention analysis and high-risk customers."
    )

    # --------------------------------------------------------
    # HIGH-RISK CUSTOMERS
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            🚨 HIGH-RISK CUSTOMERS
        </div>
        <div class="section-description">
            Customers whose predicted churn probability is at or above the model decision threshold.
        </div>
        """
)

    if scored_df is not None and probability_column is not None:

        risk_df = scored_df.copy()

        if contract_filter and "Contract" in risk_df.columns:
            risk_df = risk_df[risk_df["Contract"].isin(contract_filter)]

        if internet_filter and "InternetService" in risk_df.columns:
            risk_df = risk_df[risk_df["InternetService"].isin(internet_filter)]

        if tenure_filter and "TenureBucket" in risk_df.columns:
            risk_df = risk_df[risk_df["TenureBucket"].isin(tenure_filter)]

        if customer_search.strip() and "customerID" in risk_df.columns:
            risk_df = risk_df[
                risk_df["customerID"].astype(str).str.contains(
                    customer_search.strip(),
                    case=False,
                    na=False
                )
            ]

        risk_df = risk_df[
            risk_df[probability_column] >= threshold
        ].sort_values(
            probability_column,
            ascending=False
        )

        risk_df = add_risk_level(
            risk_df,
            probability_column
        )

        display_columns = [
            "customerID",
            "Contract",
            "tenure",
            "InternetService",
            "MonthlyCharges",
            probability_column,
            "RiskLevel"
        ]

        display_columns = [
            column for column in display_columns
            if column in risk_df.columns
        ]

        if len(risk_df) > 0:
            display_risk = risk_df[display_columns].head(20).copy()

            if probability_column in display_risk.columns:
                display_risk[probability_column] = (
                    display_risk[probability_column] * 100
                ).round(1).astype(str) + "%"

            st.dataframe(
                display_risk,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download High-Risk Customers",
                data=risk_df.to_csv(index=False).encode("utf-8"),
                file_name="high_risk_customers.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success("No high-risk customers match the selected filters.")

    elif scored_df is None:
        st.info("customers_scored.csv was not found.")
    else:
        st.warning("Churn probability column was not found in customers_scored.csv.")

    # --------------------------------------------------------
    # FILTERED CUSTOMER DATA
    # --------------------------------------------------------

    st.html(
        """
        <div class="section-header">
            👥 FILTERED CUSTOMER DATA
        </div>
        """
)

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Filtered Customers",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

    # --------------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------------

    filtered_risk_df = None

    if scored_df is not None and probability_column is not None:

        filtered_risk_df = scored_df.copy()

        if contract_filter and "Contract" in filtered_risk_df.columns:
            filtered_risk_df = filtered_risk_df[
                filtered_risk_df["Contract"].isin(contract_filter)
            ]

        if internet_filter and "InternetService" in filtered_risk_df.columns:
            filtered_risk_df = filtered_risk_df[
                filtered_risk_df["InternetService"].isin(internet_filter)
            ]

        if tenure_filter and "TenureBucket" in filtered_risk_df.columns:
            filtered_risk_df = filtered_risk_df[
                filtered_risk_df["TenureBucket"].isin(tenure_filter)
            ]

        if customer_search.strip() and "customerID" in filtered_risk_df.columns:
            filtered_risk_df = filtered_risk_df[
                filtered_risk_df["customerID"].astype(str).str.contains(
                    customer_search.strip(),
                    case=False,
                    na=False
                )
            ]

        filtered_risk_df = filtered_risk_df[
            filtered_risk_df[probability_column] >= threshold
        ]

    ai_summary, ai_actions = generate_dashboard_ai_insight(
        filtered_df,
        filtered_risk_df,
        probability_column,
        threshold
    )

    actions_html = "".join(
        f"<li>{action}</li>"
        for action in ai_actions
    )

    st.html(
        f"""
        <div class="ai-card">
            <div class="ai-title">🤖 AI CHURN INSIGHTS</div>

            <div class="ai-text">
                {ai_summary}
            </div>

            <div class="ai-action-title">
                💡 Suggested Business Actions
            </div>

            <ul class="ai-action-list">
                {actions_html}
            </ul>
        </div>
        """
    )



# ============================================================
# CUSTOMER PREDICTION PAGE
# ============================================================

elif page == "👤 Customer Prediction":


    st.title(
        "👤 Customer Churn Prediction"
    )


    st.write(
        "Enter customer information to predict "
        "churn probability."
    )


    st.divider()


    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

    st.markdown(
        "### Personal Information"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        gender = st.selectbox(
            "Gender",
            sorted(
                df["gender"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )


    with col3:

        partner = st.selectbox(
            "Partner",
            sorted(
                df["Partner"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col4:

        dependents = st.selectbox(
            "Dependents",
            sorted(
                df["Dependents"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    # ========================================================
    # TENURE
    # ========================================================

    st.markdown(
        "### Tenure & Phone Service"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )


    with col2:

        tenure_bucket = st.selectbox(
            "Tenure Bucket",
            sorted(
                df["TenureBucket"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            sorted(
                df["PhoneService"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    # ========================================================
    # INTERNET SERVICES
    # ========================================================

    st.markdown(
        "### Internet & Additional Services"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            sorted(
                df["MultipleLines"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            sorted(
                df["InternetService"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col3:

        online_security = st.selectbox(
            "Online Security",
            sorted(
                df["OnlineSecurity"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            sorted(
                df["OnlineBackup"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            sorted(
                df["DeviceProtection"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col3:

        tech_support = st.selectbox(
            "Tech Support",
            sorted(
                df["TechSupport"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    col1, col2 = st.columns(2)


    with col1:

        streaming_tv = st.selectbox(
            "Streaming TV",
            sorted(
                df["StreamingTV"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col2:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            sorted(
                df["StreamingMovies"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    # ========================================================
    # CONTRACT AND BILLING
    # ========================================================

    st.markdown(
        "### Contract & Billing"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        contract = st.selectbox(
            "Contract",
            sorted(
                df["Contract"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            sorted(
                df["PaperlessBilling"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            sorted(
                df["PaymentMethod"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    # ========================================================
    # FINANCIAL INFORMATION
    # ========================================================

    st.markdown(
        "### Financial Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )


    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )


    # ========================================================
    # ENGINEERED FEATURES
    # ========================================================

    is_month_to_month = int(
        contract == "Month-to-month"
    )


    is_electronic_check = int(
        payment_method
        == "Electronic check"
    )


    if tenure > 0:

        avg_monthly_spend = (
            total_charges / tenure
        )

    else:

        avg_monthly_spend = (
            monthly_charges
        )


    addon_services = [

        online_security,

        online_backup,

        device_protection,

        tech_support,

        streaming_tv,

        streaming_movies

    ]


    num_addon_services = sum(
        value == "Yes"
        for value in addon_services
    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()


    predict_button = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )


    if predict_button:


        # ====================================================
        # CUSTOMER DATA
        # ====================================================

        customer_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                senior_citizen
            ],

            "Partner": [
                partner
            ],

            "Dependents": [
                dependents
            ],

            "tenure": [
                tenure
            ],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless_billing
            ],

            "PaymentMethod": [
                payment_method
            ],

            "TenureBucket": [
                tenure_bucket
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ],

            "NumAddonServices": [
                num_addon_services
            ],

            "AvgMonthlySpend": [
                avg_monthly_spend
            ],

            "IsMonthToMonth": [
                is_month_to_month
            ],

            "IsElectronicCheck": [
                is_electronic_check
            ]

        })


        # ====================================================
        # PREDICTION
        # ====================================================

        probability = model.predict_proba(
            customer_data
        )[0, 1]


        prediction = int(
            probability >= threshold
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()


        st.subheader(
            "🎯 Prediction Result"
        )


        risk_level = get_risk_level(probability)

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        with col3:

            if prediction == 1:

                st.error(
                    "⚠️ MODEL DECISION: CHURN RISK"
                )

            else:

                st.success(
                    "✅ MODEL DECISION: LOWER RISK"
                )


        st.progress(
            float(probability)
        )


        # ====================================================
        # SHAP
        # ====================================================

        st.divider()


        st.subheader(
            "🧠 Why This Prediction?"
        )


        transformed_customer = (
            preprocessor.transform(
                customer_data
            )
        )


        if hasattr(
            transformed_customer,
            "toarray"
        ):

            transformed_customer = (
                transformed_customer.toarray()
            )


        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        shap_output = (
            explainer.shap_values(
                transformed_customer
            )
        )


        if isinstance(
            shap_output,
            list
        ):

            shap_values = (
                shap_output[1]
            )

        else:

            shap_values = (
                shap_output
            )


        if hasattr(
            shap_values,
            "values"
        ):

            shap_values = (
                shap_values.values
            )


        shap_values = np.asarray(
            shap_values
        )


        n_features = (
            transformed_customer.shape[1]
        )


        if shap_values.ndim == 3:

            if shap_values.shape[0] == 1:

                shap_values = (
                    shap_values[0]
                )


            if (
                shap_values.ndim == 2
                and shap_values.shape[1] == 2
            ):

                shap_values = (
                    shap_values[:, 1]
                )


        elif shap_values.ndim == 2:

            if (
                shap_values.shape[0] == 1
                and shap_values.shape[1]
                == n_features
            ):

                shap_values = (
                    shap_values[0]
                )


            elif (
                shap_values.shape[0]
                == n_features
                and shap_values.shape[1]
                == 2
            ):

                shap_values = (
                    shap_values[:, 1]
                )


        shap_values = np.asarray(
            shap_values
        ).reshape(-1)


        if len(shap_values) != len(
            feature_names
        ):

            st.error(
                "SHAP feature/value mismatch."
            )

            st.stop()


        explanation = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": shap_values,

            "Absolute SHAP":
                np.abs(shap_values)

        })


        explanation = (
            explanation
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
        )


        top_features = (
            explanation
            .head(8)
            .copy()
        )


        top_features[
            "Feature"
        ] = (
            top_features[
                "Feature"
            ]
            .str.replace(
                "num__",
                "",
                regex=False
            )
            .str.replace(
                "cat__",
                "",
                regex=False
            )
        )


        st.markdown(
            "### Top Factors Influencing This Prediction"
        )


        st.dataframe(
            top_features[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ],
            use_container_width=True
        )


        # ====================================================
        # AI CUSTOMER INSIGHT
        # ====================================================

        customer_ai_summary, customer_ai_actions = (
            generate_customer_ai_insight(
                probability,
                prediction,
                top_features,
                customer_data
            )
        )

        customer_actions_html = "".join(
            f"<li>{action}</li>"
            for action in customer_ai_actions
        )

        st.html(
            f"""
            <div class="ai-card">
                <div class="ai-title">
                    🤖 AI CUSTOMER INSIGHT
                </div>

                <div class="ai-text">
                    {customer_ai_summary}
                </div>

                <div class="ai-action-title">
                    💡 Recommended Retention Actions
                </div>

                <ul class="ai-action-list">
                    {customer_actions_html}
                </ul>
            </div>
            """
        )


        # ====================================================
        # SHAP CHART
        # ====================================================

        plot_data = (
            top_features
            .sort_values(
                "SHAP Value"
            )
        )


        fig, ax = plt.subplots(
            figsize=(10, 6)
        )


        ax.barh(
            plot_data["Feature"],
            plot_data["SHAP Value"]
        )


        ax.axvline(
            0,
            linewidth=1
        )


        ax.set_xlabel(
            "SHAP Value"
        )


        ax.set_title(
            "Factors Influencing Churn Prediction"
        )


        plt.tight_layout()


        st.pyplot(fig)


        plt.close(fig)


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.divider()


        st.subheader(
            "💡 Retention Recommendations"
        )


        recommendations = (
            generate_recommendations(
                customer_data,
                prediction
            )
        )


        for recommendation in recommendations:

            st.markdown(
                f"• {recommendation}"
            )


        # ====================================================
        # CUSTOMER INPUT
        # ====================================================

        with st.expander(
            "🔍 View Customer Input"
        ):

            st.dataframe(
                customer_data,
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Intelligence • "
    "Machine Learning + Explainable AI"
)