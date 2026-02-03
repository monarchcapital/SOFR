import streamlit as st

st.set_page_config(layout="wide", page_title="Yield Curve PCA Suite")

st.title("📊 Yield Curve Analysis Dashboard")
st.markdown("---")

st.write("""
Welcome to the PCA Analysis Suite. Use the sidebar on the left to navigate 
between different yield curve models.
""")

# Optional: Add a high-level summary or instructions here
st.info("Select a page in the sidebar to begin your analysis.")
