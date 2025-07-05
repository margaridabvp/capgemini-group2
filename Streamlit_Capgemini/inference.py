import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and data
model = joblib.load("xgb_balanced_model.pkl")
supplier_agg = pd.read_csv("supplier_agg.csv")
product_agg = pd.read_csv("incidents_by_product.csv")

# Set up page
st.set_page_config(page_title="Bad Packaging Predictor", layout="centered")

st.title("📦 Bad Packaging Probability Predictor")
st.markdown("Enter packaging details to estimate the risk of bad packaging.")

# === 1. USER INPUTS ===
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        supplier = st.selectbox("Supplier Name", sorted(supplier_agg['SupplierName'].unique()))
        product_ref = st.selectbox("Product Reference", sorted(product_agg['ProductReference'].unique()))
        weight = st.number_input("Weight (kg)", min_value=0.01, value=0.5)
        proposed_units = st.number_input("Proposed Units per Carton", min_value=1, value=10)
        size = st.selectbox("Size", ['XS', 'S', 'M', 'L', 'XL'])
        collection = st.selectbox("Collection", ['Spring', 'Summer', 'Autumn', 'Winter'])

    with col2:
        garment = st.selectbox("Garment Type", ['Blouse', 'Coat', 'Dress', 'Hoodie', 'Jacket',
                                                'Pants', 'Shirt', 'Shorts', 'Skirt', 'Suit', 'Sweater', 'T-Shirt'])
        material = st.selectbox("Material", ['Cotton', 'Denim', 'Linen', 'Polyester', 'Silk', 'Wool'])
        folding = st.selectbox("Proposed Folding Method", ['Method1', 'Method2', 'Method3', 'Foldx'])
        layout = st.selectbox("Proposed Layout", ['LAYOUTA', 'LAYOUTB', 'LAYOUTC', 'LAYOUTD', 'LAYOUTE', 'LAYOUT_OTHER'])

    submitted = st.form_submit_button("🔍 Predict")

# === 2. AGG LOOKUP FUNCTIONS ===
def get_supplier_agg(supplier_name):
    row = supplier_agg[supplier_agg['SupplierName'] == supplier_name]
    if row.empty:
        return [0] * 5
    return row.iloc[0][[
        'Avg_BadPackagingRate (Sup)',
        'Avg_Incidents_PerMonth (Sup)',
        'Avg_CostPerIncident_Scorecard (Sup)',
        'Avg_OnTimeDeliveryRate (Sup)',
        'Avg_Anomalies_PerMonth (Sup)'
    ]].values

def get_product_agg(product_ref):
    row = product_agg[product_agg['ProductReference'] == product_ref]
    if row.empty:
        return [0] * 3
    return row.iloc[0][[
        'Avg_CostImpact_Product (ProdRef)',
        'UnresolvedRate_Product (ProdRef)',
        'Total_Incidents_Product (ProdRef)'
    ]].values

# === 3. PROCESS INPUT AND PREDICT ===
if submitted:
    # Get aggregations
    s1, s2, s3, s4, s5 = get_supplier_agg(supplier)
    p1, p2, p3 = get_product_agg(product_ref)

    # Raw + engineered features
    data = {
        'Weight': weight,
        'ProposedUnitsPerCarton': proposed_units,
        'Avg_BadPackagingRate (Sup)': s1,
        'Avg_Incidents_PerMonth (Sup)': s2,
        'Avg_CostPerIncident_Scorecard (Sup)': s3,
        'Avg_OnTimeDeliveryRate (Sup)': s4,
        'Avg_Anomalies_PerMonth (Sup)': s5,
        'Avg_CostImpact_Product (ProdRef)': p1,
        'UnresolvedRate_Product (ProdRef)': p2,
        'Total_Incidents_Product (ProdRef)': p3,
        'EstimatedPackageWeight': weight * proposed_units,
        'ProductRiskScore': p1 * p2,
        'SupplierRiskIndex': s1 * s2,
        'SupplierPerformanceRatio': s4 / (s5 + 1e-5),
        'RelativeIncidentCost': p1 / (s3 + 1e-5),
        'IncidentsPerUnit': p3 / (proposed_units + 1e-5),
        'AnomalyToIncidentRatio': s5 / (s2 + 1e-5),
        'UnresolvedImpact': p2 * p1,
        'WeightToIncidentRatio': weight / (p3 + 1e-5),
        'SupplierProductInteractionScore': (s1 * s2) * p1,
        'AnomalyPerWeightUnit': s5 / (weight + 1e-3),
        'IncidentDensity': p3 / (proposed_units + 1)
    }

    df = pd.DataFrame([data])

    # One-hot encode categorical variables
    def encode_and_add(prefix, value, options):
        for opt in options:
            df[f"{prefix}_{opt}"] = 1 if value == opt else 0

    encode_and_add("Supplier", supplier, [
        'Suppliera', 'Supplierb', 'Supplierc', 'Supplierd', 'Suppliere',
        'Supplierf', 'Supplierg', 'Supplierh'
    ])
    encode_and_add("Size", size, ['XS', 'S', 'M', 'L', 'XL'])
    encode_and_add("Garment", garment, [
        'Blouse', 'Coat', 'Dress', 'Hoodie', 'Jacket', 'Pants', 'Shirt',
        'Shorts', 'Skirt', 'Suit', 'Sweater', 'T-Shirt'
    ])
    encode_and_add("Collection", collection, ['Spring', 'Summer', 'Autumn', 'Winter'])
    encode_and_add("Material", material, ['Cotton', 'Denim', 'Linen', 'Polyester', 'Silk', 'Wool'])
    encode_and_add("Folding", folding, ['Method1', 'Method2', 'Method3', 'Foldx'])
    encode_and_add("Layout", layout, [
        'LAYOUTA', 'LAYOUTB', 'LAYOUTC', 'LAYOUTD', 'LAYOUTE', 'LAYOUT_OTHER'
    ])

    # Fill any missing columns (required by model)
    expected_cols = model.get_booster().feature_names
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order
    df = df[expected_cols]

    # Prediction
    prob = model.predict_proba(df)[0][1]

    st.success(f"🧪 **Predicted probability of BAD packaging: {prob:.2%}**")

    if prob > 0.6:
        st.error("⚠️ High risk! Investigate this packaging plan.")
    elif prob > 0.3:
        st.warning("🟠 Medium risk. Consider reviewing the design.")
    else:
        st.info("✅ Low risk. No major concerns detected.")

