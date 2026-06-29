import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Resolve paths relative to this file so the app works from any cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(os.path.join(BASE_DIR, "data", "diabetes.csv"))

# =========================
# LOAD MODEL & SCALER  ← FIX: Real model load karo
# =========================
@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "diabetes_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 Diabetes Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "EDA Dashboard",
        "Visualizations",
        "Prediction",
        "Model Performance",
        "About"
    ]
)

# =========================
# HOME
# =========================
if page == "Home":

    st.title("🩺 Diabetes Prediction System")

    st.markdown("""
    ### Predict Diabetes using Machine Learning

    This project uses the Pima Indians Diabetes Dataset
    to predict whether a patient is diabetic or not.

    #### Features:
    - Data Analysis
    - Visualization
    - Machine Learning
    - Diabetes Risk Prediction
    """)

    st.image(
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef",
        use_container_width=True
    )

# =========================
# DATASET OVERVIEW
# =========================
elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write("Shape of Dataset")
    st.write(df.shape)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    st.write("Statistical Summary")
    st.dataframe(df.describe())

# =========================
# EDA DASHBOARD
# =========================
elif page == "EDA Dashboard":

    st.title("📈 EDA Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

    with col2:
        st.subheader("Outcome Count")

        fig, ax = plt.subplots()
        df["Outcome"].value_counts().plot(
            kind="bar",
            ax=ax
        )
        st.pyplot(fig)

# =========================
# VISUALIZATIONS
# =========================
elif page == "Visualizations":

    st.title("📉 Data Visualizations")

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Glucose Distribution")

    fig2, ax2 = plt.subplots()

    sns.histplot(
        df["Glucose"],
        kde=True,
        ax=ax2
    )

    st.pyplot(fig2)

# =========================
# PREDICTION PAGE  ← FIX: Real model use hoga ab
# =========================
elif page == "Prediction":

    st.title("🤖 Diabetes Prediction")

    st.markdown("### Input Features")

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            value=120
        )

        bloodpressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            value=70
        )

        skinthickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            value=79
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            value=30
        )

    # Warn on physiologically impossible zeros (kept as a soft warning, not a block)
    zero_flags = [
        name for name, val in [
            ("Glucose", glucose),
            ("Blood Pressure", bloodpressure),
            ("BMI", bmi),
        ] if val == 0
    ]
    if zero_flags:
        st.warning(
            "These values are 0, which is physiologically implausible and may "
            "skew the prediction: " + ", ".join(zero_flags)
        )

    if st.button("🔍 Predict Diabetes"):

        # Build a DataFrame with the exact training column names so the scaler
        # receives valid feature names (avoids sklearn feature-name warnings).
        input_data = pd.DataFrame([[
            pregnancies,
            glucose,
            bloodpressure,
            skinthickness,
            insulin,
            bmi,
            dpf,
            age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])

        # Scale the input (same scaler used during training)
        input_scaled = scaler.transform(input_data)

        # Real prediction
        prediction = model.predict(input_scaled)[0]           # 0 or 1
        probability = model.predict_proba(input_scaled)[0]    # [prob_0, prob_1]

        diabetic_prob = round(probability[1] * 100, 2)
        non_diabetic_prob = round(probability[0] * 100, 2)

        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
            st.metric("Diabetic Probability", f"{diabetic_prob}%")
        else:
            st.success("✅ Low Risk of Diabetes")
            st.metric("Non-Diabetic Probability", f"{non_diabetic_prob}%")

        # Extra: Probability bar chart
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            "Outcome": ["Non-Diabetic", "Diabetic"],
            "Probability (%)": [non_diabetic_prob, diabetic_prob]
        })
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.barplot(x="Outcome", y="Probability (%)", data=prob_df,
                    palette=["green", "red"], ax=ax3)
        ax3.set_ylim(0, 100)
        for i, v in enumerate([non_diabetic_prob, diabetic_prob]):
            ax3.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
        st.pyplot(fig3)

# =========================
# MODEL PERFORMANCE
# =========================
elif page == "Model Performance":

    st.title("🏆 Model Performance")

    st.info(
        "The deployed prediction model is a **Random Forest** trained on the "
        "8 raw features. The table below shows tuned accuracy from the notebook "
        "experiments for comparison."
    )

    performance = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "LightGBM"
        ],

        "Accuracy": [
            87.31,
            86.16,
            86.15,
            88.28,
            89.74,
            90.07
        ]
    })

    st.dataframe(performance)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x="Model",
        y="Accuracy",
        data=performance,
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

# =========================
# ABOUT
# =========================
elif page == "About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ### Diabetes Classification Project

    Technologies Used:

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - Streamlit
    - Seaborn
    - Matplotlib
    - XGBoost
    - LightGBM

    Dataset:
    Pima Indians Diabetes Dataset

    Developed By:
    Vishal Yadav
    """)
