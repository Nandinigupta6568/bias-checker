import streamlit as st
import pandas as pd
import pickle
import os
from bias.bias_check import check_bias

st.set_page_config(page_title="AI Fairness Auditor", layout="wide")
st.title("⚖️ Unbiased AI Decision Checker")

# --- Asset Loader ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("model/model.pkl", "rb"))
        encoder = pickle.load(open("model/encoder.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        return model, encoder, scaler
    except:
        return None, None, None

model, encoder, scaler = load_assets()

if model is None:
    st.error("❌ Model assets missing. Please run: python model/train_model.py")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    display_df = df.copy()

    try:
        # 1. Preprocess using EXACT names from training
        gender_nums = encoder.transform(df['gender'])
        X_input = pd.DataFrame({
            'gender': gender_nums,
            'age': df['age'],
            'income': df['income']
        })

        # 2. Predict
        X_scaled = scaler.transform(X_input)
        display_df['prediction'] = model.predict(X_scaled)

        # 3. Audit
        result = check_bias(display_df, 'gender', 'prediction')

        # --- UI Results ---
        st.header("📊 Audit Results")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{result['group1']} Approval", f"{result['rate1']:.1%}")
        col2.metric(f"{result['group2']} Approval", f"{result['rate2']:.1%}")
        col3.metric("Fairness Ratio", f"{result['ratio']:.2f}")

        if result['biased']:
            st.error("🚨 BIAS DETECTED")
        else:
            st.success("✅ SYSTEM IS FAIR")

        st.bar_chart({result['group1']: result['rate1'], result['group2']: result['rate2']})
        st.write("### Prediction Preview")
        st.dataframe(display_df)

    except Exception as e:
        st.error(f"Processing Error: {e}")