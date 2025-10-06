import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="ClimateAgriProject", page_icon="📊", layout="wide")
st.title("🌾 Climate Change Effects on Agriculture")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("data/climate_change_impact_on_agriculture_2024.csv")
    print(df.columns.tolist())
except FileNotFoundError:
    st.error("❌ Dataset not found! Please place it in the 'data/' folder.")
    st.stop()

# -----------------------------
# Tabs for Navigation
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📘 Data Overview", "📈 Column-wise Trends", "🔍 EDA Insights"])

# -------------------------------------------------------------------
# TAB 1: DATA OVERVIEW
# -------------------------------------------------------------------
with tab1:
    st.header("1. Dataset Overview")

    # Summary Metrics
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
            ax.grid(True, linestyle='--', alpha=0.6)
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
    st.markdown("Below are line plots showing how each numerical factor (e.g., temperature, precipitation, fertilizers, yield, etc.) varies over the years.")

    if 'Year' not in df.columns:
        st.error("⚠️ 'Year' column not found in dataset!")
    else:
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'Year']

        # Create plots in two columns
        cols = st.columns(2)
        for idx, col in enumerate(numeric_cols):
            with cols[idx % 2]:  # alternate between left and right columns
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.lineplot(x='Year', y=col, data=df, marker='o', ax=ax, linewidth=2)
                ax.set_title(f"{col} Over Years", fontsize=12)
                ax.set_xlabel("Year")
                ax.set_ylabel(col)
                ax.grid(True, linestyle='--', alpha=0.7)  # add gridlines
                st.pyplot(fig)

# -------------------------------------------------------------------
# TAB 3: EDA INSIGHTS (in two-column layout)
# -------------------------------------------------------------------
with tab3:
    st.header("🔍 Exploratory Data Analysis (EDA) Insights")
    st.markdown("Each section below explores specific analytical questions related to climate, agriculture, sustainability, and economics.")

    # Define reusable two-column layout
    def two_col_plot(plot_func_list):
        """Utility to arrange multiple plots side by side"""
        cols = st.columns(2)
        for idx, func in enumerate(plot_func_list):
            with cols[idx % 2]:
                func()

    # 1️⃣ Climate Trends
    st.subheader("1️⃣ Climate Trends")
    st.markdown("- How have average temperature and precipitation changed across years and regions?")

    def climate_temp_plot():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='Average_Temperature_C', data=df, marker='o', linewidth=2, ax=ax)
        ax.set_title("Average Temperature Over Years")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    def climate_precip_plot():
        if 'Precipitation_mm' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(x='Year', y='Precipitation_mm', data=df, marker='o', linewidth=2, ax=ax, color='teal')
            ax.set_title("Precipitation Over Years")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    two_col_plot([climate_temp_plot, climate_precip_plot])

    # 2️⃣ Agriculture Trends
    st.subheader("2️⃣ Agriculture Trends")
    st.markdown("- How has crop yield changed across different crop types over time?")

    def yield_crop_plot():
        if 'Crop' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(x='Year', y='Yield_tonnes_per_hectare', hue='Crop', data=df, ax=ax, linewidth=2)
            ax.set_title("Yield Over Time by Crop")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    def region_yield_plot():
        if 'Region' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(x='Year', y='Yield_tonnes_per_hectare', hue='Region', data=df, ax=ax, linewidth=2)
            ax.set_title("Yield Over Time by Region")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    two_col_plot([Crop_Yield_MT_per_HA, region_yield_plot])

    # 3️⃣ Climate–Agriculture Relationship
    st.subheader("3️⃣ Climate–Agriculture Relationship")

    def temp_yield_corr():
        if 'Average_Temperature_C' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x='Average_Temperature_C', y='Yield_tonnes_per_hectare', data=df, scatter_kws={'alpha':0.6})
            ax.set_title("Temperature vs Yield")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    def precip_yield_corr():
        if 'Precipitation_mm' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x='Precipitation_mm', y='Yield_tonnes_per_hectare', data=df, scatter_kws={'alpha':0.6}, color='seagreen')
            ax.set_title("Precipitation vs Yield")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    two_col_plot([temp_yield_corr, precip_yield_corr])

    # 4️⃣ Inputs & Sustainability
    st.subheader("4️⃣ Inputs & Sustainability")

    def fertilizer_yield():
        if 'Fertilizer_Use_kg_per_ha' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x='Fertilizer_Use_kg_per_ha', y='Yield_tonnes_per_hectare', data=df, scatter_kws={'alpha':0.6})
            ax.set_title("Fertilizer Use vs Yield")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    def irrigation_trend():
        if 'Irrigation_Access_Percent' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.lineplot(x='Year', y='Irrigation_Access_Percent', data=df, marker='o', linewidth=2, ax=ax, color='dodgerblue')
            ax.set_title("Irrigation Access Over Time")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    two_col_plot([fertilizer_yield, irrigation_trend])

    # 5️⃣ Adaptation & Resilience
    st.subheader("5️⃣ Adaptation & Resilience")

    def adaptation_yield():
        if 'Adaptation_Strategy' in df.columns and 'Yield_tonnes_per_hectare' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x='Adaptation_Strategy', y='Yield_tonnes_per_hectare', data=df)
            ax.set_title("Yield by Adaptation Strategy")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=30)
            st.pyplot(fig)

    def adaptation_economics():
        if 'Adaptation_Strategy' in df.columns and 'Economic_Impact_USD' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x='Adaptation_Strategy', y='Economic_Impact_USD', data=df, palette='Set2')
            ax.set_title("Economic Impact by Adaptation Strategy")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=30)
            st.pyplot(fig)

    two_col_plot([adaptation_yield, adaptation_economics])

    # 6️⃣ Economic Impact
    st.subheader("6️⃣ Economic Impact")

    def econ_vs_weather():
        if 'Economic_Impact_USD' in df.columns and 'Extreme_Weather_Events' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.regplot(x='Extreme_Weather_Events', y='Economic_Impact_USD', data=df, scatter_kws={'alpha':0.6})
            ax.set_title("Economic Impact vs Extreme Weather Events")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    def econ_by_crop():
        if 'Crop' in df.columns and 'Economic_Impact_USD' in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x='Crop', y='Economic_Impact_USD', data=df, palette='pastel')
            ax.set_title("Economic Impact by Crop Type")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=30)
            st.pyplot(fig)

    two_col_plot([econ_vs_weather, econ_by_crop])

    st.markdown("---")
    st.info("""
    💡 **Interpretation Tips**
    - Gridlines make pattern recognition easier.
    - Two-column layout helps you visually compare related metrics.
    - Regression lines reveal correlations between climate and yield.
    """)
