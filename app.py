import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_resource
def load_model_assets():
    model = joblib.load("models/random_forest_model.joblib")
    encoders = joblib.load("models/label_encoders.joblib")
    return model, encoders


model, encoders = load_model_assets()


data = pd.read_csv("data/credit_risk_dataset.csv")



# Streamlit Page Setup
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("Credit Risk Prediction System")
st.write(
    """This application predicts the likelihood of a loan applicant defaulting "
    "based on their personal, financial, and loan-related information. "
    "It also provides visual insights into key relationships within the credit risk dataset.
"""
)


# Tabs for navigation
tab1, tab2 = st.tabs(["Prediction", "Data Insights"])


# Tab 1: Prediction
with tab1:
    st.subheader("Enter Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
        person_income = st.number_input("Annual Income (USD)", min_value=1000, max_value=1000000, value=25000)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=3.0)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=2)

    with col2:
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Loan Amount (USD)", min_value=500, max_value=50000, value=10000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=10.99)
        loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.15)
        cb_person_default_on_file = st.selectbox("Previously Defaulted", ["Y", "N"])

    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    # Encode categorical features
    for col in encoders:
        if col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])

    if st.button("Predict Credit Risk"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Risk of Default ({proba[1]*100:.2f}% probability)")
        else:
            st.success(f"Low Risk of Default ({proba[0]*100:.2f}% probability)")

        st.caption("Model used: Random Forest | Data source: credit_risk_dataset.csv")


# Tab 2: Data Visualization
with tab2:
   
    overview_data = data.copy()
    display_data = overview_data.rename(columns={
        "person_age": "Age",
        "person_income": "Annual Income (USD)",
        "person_home_ownership": "Home Ownership",
        "person_emp_length": "Employment Length (Years)",
        "loan_intent": "Loan Intent",
        "loan_grade": "Loan Grade",
        "loan_amnt": "Loan Amount (USD)",
        "loan_int_rate": "Interest Rate (%)",
        "loan_percent_income": "Loan to Income Ratio",
        "cb_person_default_on_file": "Previous Default (Y/N)",
        "cb_person_cred_hist_length": "Credit History Length (Years)",
        "loan_status": "Loan Status (0 = Safe, 1 = Default)"
    })

    st.subheader("Dataset Overview")
    st.dataframe(display_data.head(10), width=True,)


    if "loan_status" not in data.columns:
        st.warning("The dataset does not contain a 'loan_status' column. Visualizations may be incomplete.")
    else:
        st.info("Note: In all charts below — 0 represents **No Default (Safe Borrower)**, "
                "and 1 represents **Default (High Risk Borrower)**.")

        # 1 Loan Intent Distribution
        st.markdown("#### Loan Intent Distribution")
        
        data["loan_intent"] = data["loan_intent"].replace({
            "DEBTCONSOLIDATION": "DEBT CONSOLIDATION",
            "HOMEIMPROVEMENT": "HOME IMPROVEMENT"
        })

        intent_counts = data["loan_intent"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            intent_counts.values,
            labels=intent_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel"),
            
        )
        ax.axis("equal")
        st.pyplot(fig,width=600)

        # 2 Age Distribution by Loan Status
        st.markdown("#### Age Distribution by Loan Status")
        filtered_data = data[data["person_age"] <= 80]
        fig, ax = plt.subplots()
        sns.histplot(
            data=filtered_data,
            x="person_age",
            hue="loan_status",
            multiple="stack",
            kde=True,
            bins=30,
            palette=["#1f77b4", "#d62728"],
            ax=ax
        )
        ax.set_xlim(18, 80)
        ax.set_xlabel("Person Age")
        ax.set_ylabel("Number of Applicants")
        st.pyplot(fig,width=600)

        # 3 Income vs Loan Amount
        st.markdown("#### Relationship Between Income and Loan Amount")

        data["income_group"] = pd.cut(
            data["person_income"],
            bins=[0, 20000, 40000, 60000, 80000, 100000, 150000, 200000, np.inf],
            labels=[
                "<20k", "20k", "40k", "60k", "80k",
                "100k", "150k", "200k+"
            ]
        )

        avg_loan_by_income = (
            data.groupby("income_group",observed=True)["loan_amnt"]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=avg_loan_by_income,
            x="income_group",
            y="loan_amnt",
            marker="o",
            color="steelblue",
            linewidth=2.5,
            ax=ax
        )

        ax.set_xlabel("Income Range (USD)")
        ax.set_ylabel("Average Loan Amount (USD)")
        ax.set_title("Average Loan Amount Across Income Levels")
        plt.xticks(rotation=30)
        plt.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig,width=600)

# Footer
st.markdown("---")
st.write(" Copyright © . All rights reserved. Credit Risk Prediction.")