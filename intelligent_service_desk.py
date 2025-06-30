# Intelligent Service Desk Automation with SLA Prediction
# Streamlit + ML-Based Web App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Generate Synthetic Ticket Data ----------
tickets = [
    ("Cannot access email", "Access Issue", "Low"),
    ("System crash when opening SAP", "Application Error", "High"),
    ("Forgot password", "Password Reset", "Low"),
    ("Printer not working", "Hardware Issue", "Medium"),
    ("VPN connection dropping frequently", "Network Issue", "Medium"),
    ("Laptop overheating", "Hardware Issue", "High"),
    ("Request for software installation", "Request", "Low"),
    ("Unable to login to Active Directory", "Access Issue", "High"),
    ("Teams chat not syncing", "Application Error", "Medium"),
    ("Screen flickering intermittently", "Hardware Issue", "Medium"),
] * 10  # Expand for more data

np.random.seed(42)
df = pd.DataFrame(tickets, columns=["description", "category", "sla_risk"])
df["ticket_id"] = range(1, len(df) + 1)
df["resolution_time_hr"] = np.random.randint(1, 72, size=len(df))
df["sla_breach"] = df["resolution_time_hr"] > 48  # Breach if > 48 hrs

# Encode target
df["sla_risk_code"] = df["sla_risk"].astype("category").cat.codes

# ---------- Machine Learning Model ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["sla_risk_code"], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

label_map = dict(enumerate(df["sla_risk"].astype("category").cat.categories))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Smart IT Service Desk", layout="centered")
st.title("🖥️ Intelligent Service Desk Automation")

menu = ["Predict SLA Risk", "View Ticket Dashboard"]
choice = st.sidebar.selectbox("Choose Module", menu)

if choice == "Predict SLA Risk":
    st.subheader("Predict SLA Risk for a New Ticket")
    user_input = st.text_area("Enter Ticket Description", "Example: User unable to connect to VPN")

    if st.button("Predict"):
        pred_code = pipeline.predict([user_input])[0]
        risk = label_map[pred_code]
        st.success(f"Predicted SLA Risk Level: **{risk}**")

elif choice == "View Ticket Dashboard":
    st.subheader("Ticket Statistics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        sla_counts = df["sla_risk"].value_counts()
        st.bar_chart(sla_counts)

    with col2:
        breach_counts = df["sla_breach"].value_counts()
        st.bar_chart(breach_counts)

    st.markdown("---")
    st.write("### Ticket Resolution Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["resolution_time_hr"], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.dataframe(df.head(10))

