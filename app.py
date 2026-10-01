import streamlit as st
import pandas as pd
import joblib
import os


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)


# -----------------------------
# Load Model Files
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "notebooks")

loaded_model = joblib.load(
    os.path.join(MODEL_DIR, "fraud_detection_model.pkl")
)

loaded_features = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)


# -----------------------------
# App Title
# -----------------------------
st.title("🛡️ Fraud Transaction Detection System")

st.markdown(
    """
    This application uses a trained Machine Learning model
    to predict whether a financial transaction is potentially
    fraudulent or legitimate.

    Enter the transaction details below and click
    **Predict Fraud** to see the result.
    """
)

st.divider()


# -----------------------------
# Input Fields
# -----------------------------
step = st.number_input(
    "Step",
    min_value=1,
    value=10
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=500000.0
)

oldbalanceOrg = st.number_input(
    "Old Balance (Origin)",
    min_value=0.0,
    value=600000.0
)

newbalanceOrig = st.number_input(
    "New Balance (Origin)",
    min_value=0.0,
    value=100000.0
)

oldbalanceDest = st.number_input(
    "Old Balance (Destination)",
    min_value=0.0,
    value=200000.0
)

newbalanceDest = st.number_input(
    "New Balance (Destination)",
    min_value=0.0,
    value=700000.0
)

isFlaggedFraud = st.selectbox(
    "Is Flagged Fraud?",
    [0, 1]
)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Fraud"):

    # Validation
    if amount <= 0:
        st.warning("Please enter an amount greater than zero.")

    elif oldbalanceOrg < 0 or newbalanceOrig < 0:
        st.warning("Origin balances cannot be negative.")

    elif oldbalanceDest < 0 or newbalanceDest < 0:
        st.warning("Destination balances cannot be negative.")

    else:

        try:

            # -----------------------------
            # Feature Engineering
            # -----------------------------
            balance_change_orig = (
                oldbalanceOrg - newbalanceOrig
            )

            balance_change_dest = (
                newbalanceDest - oldbalanceDest
            )

            errorBalanceOrig = (
                newbalanceOrig + amount - oldbalanceOrg
            )

            errorBalanceDest = (
                oldbalanceDest + amount - newbalanceDest
            )


            # -----------------------------
            # Create Transaction DataFrame
            # -----------------------------
            transaction = pd.DataFrame({
                "step": [step],
                "amount": [amount],
                "oldbalanceOrg": [oldbalanceOrg],
                "newbalanceOrig": [newbalanceOrig],
                "oldbalanceDest": [oldbalanceDest],
                "newbalanceDest": [newbalanceDest],
                "isFlaggedFraud": [isFlaggedFraud],

                "balance_change_orig": [
                    balance_change_orig
                ],

                "balance_change_dest": [
                    balance_change_dest
                ],

                "errorBalanceOrig": [
                    errorBalanceOrig
                ],

                "errorBalanceDest": [
                    errorBalanceDest
                ],

                "type_CASH_OUT": [
                    1 if transaction_type == "CASH_OUT" else 0
                ],

                "type_DEBIT": [
                    1 if transaction_type == "DEBIT" else 0
                ],

                "type_PAYMENT": [
                    1 if transaction_type == "PAYMENT" else 0
                ],

                "type_TRANSFER": [
                    1 if transaction_type == "TRANSFER" else 0
                ]
            })


            # -----------------------------
            # Arrange Features
            # -----------------------------
            transaction = transaction[loaded_features]


            # -----------------------------
            # Make Prediction
            # -----------------------------
            prediction = loaded_model.predict(transaction)[0]

            probability = loaded_model.predict_proba(
                transaction
            )[0][1]


            # -----------------------------
            # Display Result
            # -----------------------------
            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION"
                )

            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )


            st.metric(
                label="Fraud Probability",
                value=f"{probability * 100:.2f}%"
            )

            st.caption(
                "This is a model prediction, not a definitive confirmation of fraud."
            )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )