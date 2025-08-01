# Paste your entire streamlit code below

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# --- Page Setup ---
st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛒 Shopper Spectrum")
st.subheader("📦 Product Recommendations & 💡 Customer Segmentation")

# --- Load Models and Data ---
@st.cache_resource
def load_model():
  kmeans = joblib.load("kmeans_model.pkl")
  scaler = joblib.load("scaler.pkl")
  return kmeans, scaler

@st.cache_data
def load_data():
    product_matrix = pd.read_csv("product_matrix.csv", index_col=0)
    product_mapping = pd.read_csv("product_mapping.csv")
    return product_matrix, product_mapping

kmeans, scaler = load_model()
product_matrix, product_mapping = load_data()

# --- Product Recommendation Module ---
st.markdown("## 🎯 Product Recommendation")
product_input = st.text_input("Enter a product name:")

if st.button("Get Recommendations"):
    if product_input in product_matrix.columns:
        similarity = cosine_similarity(product_matrix.T)
        product_idx = product_matrix.columns.get_loc(product_input)
        similarity_scores = list(enumerate(similarity[product_idx]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_indices = [idx for idx, score in sorted_scores[1:6]]
        recommended_products = product_matrix.columns[recommended_indices]

        st.success("Here are 5 similar products:")
        for i, prod in enumerate(recommended_products, 1):
            st.markdown(f"{i}. **{prod}**")
    else:
        st.error("Product not found. Try another name.")

# --- Customer Segmentation Module ---
st.markdown("---")
st.markdown("## 🔍 Customer Segmentation")

with st.form("segmentation_form"):
    recency = st.number_input("Recency (days ago)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)
    submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        input_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_scaled)[0]

        # Label Mapping
        def get_label(c):
            if c == 0:
                return "💎 High-Value"
            elif c == 1:
                return "🧱 Regular"
            elif c == 2:
                return "🌙 Occasional"
            elif c == 3:
                return "⚠️ At-Risk"
            else:
                return "Unlabeled"

        label = get_label(cluster)
        st.success(f"This customer belongs to: **{label}** Segment")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

