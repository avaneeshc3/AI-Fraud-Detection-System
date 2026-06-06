import streamlit as st
import pickle
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)
    
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Fraud Detection App")
st.markdown('Enter transaction details and click "Predict Fraud Risk" to get a fraud prediction.')
st.divider()

model = load_model()

st.header("Transaction Details")
transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=0.0)
old_balance_orig = st.number_input("Old Balance (Origin)", min_value=0.0, value=0.0)
new_balance_orig = st.number_input("New Balance (Origin)", min_value=0.0, value=0.0)
old_balance_dest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
new_balance_dest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

# Feature engineering for model prediction
balance_diff_orig = old_balance_orig - new_balance_orig
balance_diff_dest = new_balance_dest - old_balance_dest
zero_balance_transfer = int((old_balance_orig > 0) and (new_balance_orig == 0))
amount_to_oldbal = amount / (old_balance_orig + 1)
same_orig_dest = 0

input_data = pd.DataFrame({
    "type": [transaction_type],
    "amount": [amount],
    "oldbalanceOrg": [old_balance_orig],
    "newbalanceOrig": [new_balance_orig],
    "oldbalanceDest": [old_balance_dest],
    "newbalanceDest": [new_balance_dest],
    "balanceDiffOrig": [balance_diff_orig],
    "balanceDiffDest": [balance_diff_dest],
    "zero_balance_transfer": [zero_balance_transfer],
    "amount_to_oldbal": [amount_to_oldbal],
    "same_orig_dest": [same_orig_dest],
})

# Generate prediction on button click
if st.button("Predict Fraud Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    st.header("Prediction")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"⚠️ POTENTIAL FRAUD DETECTED!")
        else:
            st.success("✅ Transaction looks good.")
    
    # Display fraud probability
    st.metric("Fraud Probability", f"{prob:.2%}")