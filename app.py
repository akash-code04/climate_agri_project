import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="ClimateAgriProject", page_icon="📊", layout="wide")
# Title 
st.title("Climate Change Effects On Agriculture") 

# Load dataset from data folder 
try: 
    df = pd.read_csv("data/climate_change_impact_on_agriculture_2024.csv") 
except FileNotFoundError: 
    st.error("Dataset Not Found!") 
    st.stop()

if df is not None:
    # Create two tabs
    tab1, tab2 = st.tabs(["Data Overview", "Data Visualization"])

    with tab1:
        # Parts 1, 2, 3 and 5 go here
        
        # 1. Dataset Overview
        st.header("1. Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # 2. Dataset Information
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
                ax.set_xlabel('Columns')
                ax.set_ylabel('Number of Missing Values')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            else:
                st.success("No missing values found! 🎉")

        st.markdown("---")

        # 3. Statistical Summary
        st.header("3. Statistical Summary")
        st.subheader("Numerical Features")
        st.dataframe(df.describe(), use_container_width=True)

        if len(df.select_dtypes(include=['object']).columns) > 0:
            st.subheader("Categorical Features")
            st.dataframe(df.describe(include=['object']), use_container_width=True)

        st.markdown("---")

        # 5. Export Data
        st.header("5. Export Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv',
        )

    with tab2:
        # Part 4 - Visualizations
        st.header("4. Data Visualizations")
        st.subheader("Sample Bar Plot")

        fig, ax = plt.subplots(figsize=(10, 6))
        # Optional: Convert Year to string to avoid matplotlib issues
        ax.bar(df['Year'].astype(str), df['Average_Temperature_C'], color='coral')
        ax.set_title('Average Temperature by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Temp (°C)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

