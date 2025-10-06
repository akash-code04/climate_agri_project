import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="ClimateAgriProject", page_icon="📊", layout="wide")
st.title("🌾 Climate Change Effects on Agriculture")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("data/climate_change_impact_on_agriculture_2024.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found! Please upload or place it in 'data/' folder.")
    st.stop()

# -----------------------------
# Tabs for Navigation
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📘 Data Overview", "📊 Column-wise Trends", "🔍 EDA Insights"])

# -------------------------------------------------------------------
# TAB 1: DATA OVERVIEW
# -------------------------------------------------------------------
with tab1:
    st.header("1. Dataset Overview")

    # Basic Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Rows", df.shape[0])
    with col2: st.metric("Total Columns", df.shape[1])
    with col3: st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4: st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

    # Preview Data
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Column Info
    st.header("2. Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Column Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)

    with col2:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Missing Values by Column')
            st.pyplot(fig)
        else:
            st.success("✅ No missing values found!")

    # Statistical Summary
    st.header("3. Statistical Summary")
    st.subheader("Numerical Features")
    st.dataframe(df.describe(), use_container_width=True)

    if len(df.select_dtypes(include=['object']).columns) > 0:
        st.subheader("Categorical Features")
        st.dataframe(df.describe(include=['object']), use_container_width=True)

    # Export Option
    st.header("4. Export Data")
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Processed Data", data=csv, file_name="processed_data.csv", mime="text/csv")

# -------------------------------------------------------------------
# TAB 2: AUTOMATIC COLUMN-WISE PLOTS
# -------------------------------------------------------------------
with tab2:
    st.header("📈 Automatic Yearly Trends for All Numerical Columns")
    st.markdown("Below plots show how each numerical factor (e.g., temperature, precipitation, fertilizers, yield, etc.) varies over the years.")

    # Ensure 'Year' column exists
    if 'Year' not in df.columns:
        st.error("⚠️ 'Year' column not found in dataset!")
    else:
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'Year']

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x='Year', y=col, data=df, marker='o', ax=ax)
            ax.set_title(f"{col} Over Years")
            ax.set_xlabel("Year")
            ax.set_ylabel(col)
            st.pyplot(fig)

# -------------------------------------------------------------------
# TAB 3: INSIGHTFUL EDA PLOTS
# -------------------------------------------------------------------
with tab3:
    st.header("🔍 Exploratory Data Analysis (EDA) Insights")

    # 1️⃣ Climate Trends
    st.subheader("1️⃣ Climate Trends")
    st.markdown("- How have average temperature and precipitation changed across years?")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(x='Year', y='Average_Temperature_C', data=df, label='Temperature (°C)', ax=ax)
    if 'Precipitation_mm' in df.columns:
        sns.lineplot(x='Year', y='Precipitation_mm', data=df, label='Precipitation (mm)', ax=ax)
    ax.set_title("Climate Trends Over Years")
    ax.legend()
    st.pyplot(fig)

    # 2️⃣ Agriculture Trends
    st.subheader("2️⃣ Agriculture Trends")
    st.markdown("- How has crop yield changed across different crop types over time?")
    if 'Crop' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.lineplot(x='Year', y='Yield_tonnes_per_hectare', hue='Crop', data=df, ax=ax)
        ax.set_title("Crop Yield Trends Over Years by Crop Type")
        st.pyplot(fig)

    # 3️⃣ Climate–Agriculture Relationship
    st.subheader("3️⃣ Climate–Agriculture Relationship")
    if 'Average_Temperature_C' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x='Average_Temperature_C', y='Yield_tonnes_per_hectare', data=df)
        ax.set_title("Temperature vs Crop Yield")
        st.pyplot(fig)

    if 'Precipitation_mm' in df.columns:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x='Precipitation_mm', y='Yield_tonnes_per_hectare', data=df)
        ax.set_title("Precipitation vs Yield")
        st.pyplot(fig)

    # 4️⃣ Inputs & Sustainability
    st.subheader("4️⃣ Inputs & Sustainability")
    if 'Fertilizer_Use_kg_per_ha' in df.columns:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x='Fertilizer_Use_kg_per_ha', y='Yield_tonnes_per_hectare', data=df)
        ax.set_title("Fertilizer Use vs Crop Yield")
        st.pyplot(fig)

    if 'Irrigation_Access_Percent' in df.columns:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.lineplot(x='Year', y='Irrigation_Access_Percent', data=df)
        ax.set_title("Irrigation Access Over Time")
        st.pyplot(fig)

    # 5️⃣ Adaptation & Resilience
    st.subheader("5️⃣ Adaptation & Resilience")
    if 'Adaptation_Strategy' in df.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x='Adaptation_Strategy', y='Yield_tonnes_per_hectare', data=df)
        ax.set_title("Yield by Adaptation Strategy")
        st.pyplot(fig)

    # 6️⃣ Economic Impact
    st.subheader("6️⃣ Economic Impact")
    if 'Economic_Impact_USD' in df.columns and 'Extreme_Weather_Events' in df.columns:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x='Extreme_Weather_Events', y='Economic_Impact_USD', data=df)
        ax.set_title("Economic Impact vs Extreme Weather Events")
        st.pyplot(fig)

    st.markdown("---")
    st.info("""
    💡 **Interpretation Hints:**
    - Temperature and precipitation trends reveal climate shifts.
    - Yield–climate correlations can show sensitivity of crops.
    - Fertilizer and irrigation plots help evaluate sustainability.
    - Adaptation strategies highlight resilience to climate shocks.
    """)

