import streamlit as st
import pandas as pd
import pickle 
import numpy as np
# Load model
with open("model_best.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Loan Approval Prediction App")
st.markdown("---")

# Inputs
dependents = st.number_input("Number of Dependents", 0, 10)
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil = st.number_input("CIBIL Score")
res_assets = st.number_input("Residential Assets Value")
com_assets = st.number_input("Commercial Assets Value")
lux_assets = st.number_input("Luxury Assets Value")
bank_assets = st.number_input("Bank Asset Value")

self_employed = 1 if self_employed == "Yes" else 0
# Prediction
if st.button("Predict Loan Status"):
    
    input_data = np.array([[dependents, self_employed,
                            income, loan_amount, loan_term, cibil,
                            res_assets, com_assets, lux_assets, bank_assets]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")