import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
    st.success("✅ Dataset loaded successfully!")
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
            plt.close(fig)
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
# TAB 2: AUTOMATIC COLUMN-WISE PLOTS (Country-wise Analysis)
# -------------------------------------------------------------------
with tab2:
    st.header("📈 Automatic Yearly Trends for All Numerical Columns (Country-wise)")
    st.markdown("Select a country from the dropdown below to visualize how each numerical factor varies over the years for that country.")

    # Check if 'Country' and 'Year' columns exist
    if 'Country' not in df.columns:
        st.error("⚠️ 'Country' column not found in dataset!")
    elif 'Year' not in df.columns:
        st.error("⚠️ 'Year' column not found in dataset!")
    else:
        # Country selection dropdown
        countries = sorted(df['Country'].dropna().unique())
        selected_country = st.selectbox("🌍 Select Country", countries, index=0)

        # Filter dataset for the selected country
        df_country = df[df['Country'] == selected_country]

        if df_country.empty:
            st.warning(f"No data available for {selected_country}.")
        else:
            # Note: Ensure np is imported (import numpy as np)
            numeric_cols = [col for col in df_country.select_dtypes(include=np.number).columns if col != 'Year']

            st.markdown(f"### 📊 Yearly Trends for **{selected_country}**")
            cols = st.columns(2)

            for idx, col in enumerate(numeric_cols):
                with cols[idx % 2]:
                    # ----------------------------------------------------------
                    # INTERACTIVE PLOTLY EXPRESS IMPLEMENTATION (REPLACING MPL)
                    # ----------------------------------------------------------
                    
                    # 1. Create the interactive line plot using Plotly Express
                    # We sort by 'Year' to ensure the line is drawn in chronological order
                    fig = px.line(
                        df_country.sort_values(by='Year'),
                        x='Year',
                        y=col,
                        title=f"{col} Over Years ({selected_country})",
                        markers=True  # Display a circle on each data point
                    )
                    
                    # 2. Customize layout for better fit and aesthetic
                    fig.update_layout(
                        title_x=0,  # Align title to the left (0 = left, 0.5 = center)
                        height=350,
                        margin=dict(t=50, b=20, l=40, r=20),
                        hovermode="x unified",
                        plot_bgcolor='rgba(128,128,128,0.75)',  # Grey background with 75% opacity
                    
                        xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.4)',  # lighter grid lines for contrast
                        gridwidth=1,
                        zeroline=False,
                        ),
                        yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.4)',
                        gridwidth=1,
                        zeroline=False,
                        )
                    )

                    
                    # 3. Display the interactive chart in Streamlit
                    # use_container_width=True ensures it fills the column width
                    st.plotly_chart(fig, use_container_width=True)
                    # ----------------------------------------------------------
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
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    def climate_precip_plot():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='Total_Precipitation_mm', data=df, marker='o', linewidth=2, ax=ax, color='teal')
        ax.set_title("Total Precipitation Over Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Precipitation (mm)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([climate_temp_plot, climate_precip_plot])

    # 2️⃣ Agriculture Trends
    st.subheader("2️⃣ Agriculture Trends")
    st.markdown("- How has crop yield changed across different crop types over time?")

    def yield_crop_plot():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='Crop_Yield_MT_per_HA', hue='Crop_Type', data=df, ax=ax, linewidth=2)
        ax.set_title("Crop Yield Over Time by Crop Type")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        plt.close(fig)

    def region_yield_plot():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='Crop_Yield_MT_per_HA', hue='Region', data=df, ax=ax, linewidth=2)
        ax.set_title("Crop Yield Over Time by Region")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([yield_crop_plot, region_yield_plot])

    # 3️⃣ Climate–Agriculture Relationship
    st.subheader("3️⃣ Climate–Agriculture Relationship")
    st.markdown("- How do temperature and precipitation correlate with crop yield?")

    def temp_yield_corr():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='Average_Temperature_C', y='Crop_Yield_MT_per_HA', data=df, scatter_kws={'alpha':0.6}, ax=ax)
        ax.set_title("Temperature vs Crop Yield")
        ax.set_xlabel("Average Temperature (°C)")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    def precip_yield_corr():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='Total_Precipitation_mm', y='Crop_Yield_MT_per_HA', data=df, scatter_kws={'alpha':0.6}, color='seagreen', ax=ax)
        ax.set_title("Precipitation vs Crop Yield")
        ax.set_xlabel("Total Precipitation (mm)")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([temp_yield_corr, precip_yield_corr])

    # 4️⃣ CO2 Emissions & Extreme Weather
    st.subheader("4️⃣ CO2 Emissions & Extreme Weather Impact")
    st.markdown("- How do CO2 emissions and extreme weather events affect crop yield?")

    def co2_yield():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='CO2_Emissions_MT', y='Crop_Yield_MT_per_HA', data=df, scatter_kws={'alpha':0.6}, color='darkred', ax=ax)
        ax.set_title("CO2 Emissions vs Crop Yield")
        ax.set_xlabel("CO2 Emissions (MT)")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    def weather_yield():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='Extreme_Weather_Events', y='Crop_Yield_MT_per_HA', data=df, scatter_kws={'alpha':0.6}, color='orange', ax=ax)
        ax.set_title("Extreme Weather Events vs Crop Yield")
        ax.set_xlabel("Extreme Weather Events")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([co2_yield, weather_yield])

    # 5️⃣ Inputs & Sustainability
    st.subheader("5️⃣ Agricultural Inputs & Sustainability")
    st.markdown("- How do fertilizer, pesticide use, and irrigation affect crop yield?")

    def fertilizer_yield():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='Fertilizer_Use_KG_per_HA', y='Crop_Yield_MT_per_HA', data=df, scatter_kws={'alpha':0.6}, ax=ax)
        ax.set_title("Fertilizer Use vs Crop Yield")
        ax.set_xlabel("Fertilizer Use (KG/HA)")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    def pesticide_yield():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='Pesticide_Use_KG_per_HA', y='Crop_Yield_MT_per_HA', data=df, scatter_kws={'alpha':0.6}, color='purple', ax=ax)
        ax.set_title("Pesticide Use vs Crop Yield")
        ax.set_xlabel("Pesticide Use (KG/HA)")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([fertilizer_yield, pesticide_yield])

    def irrigation_trend():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='Irrigation_Access_%', data=df, marker='o', linewidth=2, ax=ax, color='dodgerblue')
        ax.set_title("Irrigation Access Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Irrigation Access (%)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    def soil_health_trend():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='Soil_Health_Index', data=df, marker='o', linewidth=2, ax=ax, color='brown')
        ax.set_title("Soil Health Index Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Soil Health Index")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([irrigation_trend, soil_health_trend])

    # 6️⃣ Adaptation & Resilience
    st.subheader("6️⃣ Adaptation Strategies & Resilience")
    st.markdown("- How do different adaptation strategies affect crop yield and economic impact?")

    def adaptation_yield():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x='Adaptation_Strategies', y='Crop_Yield_MT_per_HA', data=df, ax=ax, palette='Set2')
        ax.set_title("Crop Yield by Adaptation Strategy")
        ax.set_xlabel("Adaptation Strategy")
        ax.set_ylabel("Crop Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)

    def adaptation_economics():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x='Adaptation_Strategies', y='Economic_Impact_Million_USD', data=df, palette='viridis', ax=ax)
        ax.set_title("Economic Impact by Adaptation Strategy")
        ax.set_xlabel("Adaptation Strategy")
        ax.set_ylabel("Economic Impact (Million USD)")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([adaptation_yield, adaptation_economics])

    # 7️⃣ Economic Impact Analysis
    st.subheader("7️⃣ Economic Impact Analysis")
    st.markdown("- How do extreme weather events and crop types affect economic impact?")

    def econ_vs_weather():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.regplot(x='Extreme_Weather_Events', y='Economic_Impact_Million_USD', data=df, scatter_kws={'alpha':0.6}, color='crimson', ax=ax)
        ax.set_title("Economic Impact vs Extreme Weather Events")
        ax.set_xlabel("Extreme Weather Events")
        ax.set_ylabel("Economic Impact (Million USD)")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    def econ_by_crop():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x='Crop_Type', y='Economic_Impact_Million_USD', data=df, palette='pastel', ax=ax)
        ax.set_title("Economic Impact by Crop Type")
        ax.set_xlabel("Crop Type")
        ax.set_ylabel("Economic Impact (Million USD)")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([econ_vs_weather, econ_by_crop])

    # 8️⃣ Regional & Country Analysis
    st.subheader("8️⃣ Regional & Country Analysis")
    st.markdown("- How do crop yields vary across regions and countries?")

    def yield_by_region():
        fig, ax = plt.subplots(figsize=(7, 4))
        region_avg = df.groupby('Region')['Crop_Yield_MT_per_HA'].mean().sort_values(ascending=False)
        region_avg.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Average Crop Yield by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Average Yield (MT/HA)")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)

    def top_countries():
        fig, ax = plt.subplots(figsize=(7, 4))
        country_avg = df.groupby('Country')['Crop_Yield_MT_per_HA'].mean().sort_values(ascending=False).head(10)
        country_avg.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title("Top 10 Countries by Average Crop Yield")
        ax.set_xlabel("Average Yield (MT/HA)")
        ax.set_ylabel("Country")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

    two_col_plot([yield_by_region, top_countries])

    st.markdown("---")
    st.info("""
    💡 **Interpretation Tips**
    - Gridlines make pattern recognition easier.
    - Two-column layout helps you visually compare related metrics.
    - Regression lines reveal correlations between climate factors and crop yield.
    - Box plots show distribution and outliers across categories.
    """)
