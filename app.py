import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# Title
st.title("📊 Data Exploration & Visualization Dashboard")
st.markdown("---")

# Load dataset from data folder
try:
    df = pd.read_csv("data/climate_change_impact_on_agriculture_2024.csv")  # Replace 'your_dataset.csv' with your actual filename
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found! Please ensure your CSV file is in the 'data' folder.")
    st.stop()  # Stop execution if file not found

# Main EDA Section
if df is not None:
    
    # Display dataset info
    st.header("1️⃣ Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    
    # Show raw data
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Dataset Information
    st.markdown("---")
    st.header("2️⃣ Dataset Information")
    
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
    
    # Statistical Summary
    st.markdown("---")
    st.header("3️⃣ Statistical Summary")
    
    st.subheader("Numerical Features")
    st.dataframe(df.describe(), use_container_width=True)
    
    if len(df.select_dtypes(include=['object']).columns) > 0:
        st.subheader("Categorical Features")
        st.dataframe(df.describe(include=['object']), use_container_width=True)
    
    # Visualizations
    st.markdown("---")
    st.header("4️⃣ Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Distribution plots
    if numeric_cols:
        st.subheader("Distribution of Numerical Features")
        selected_num_col = st.selectbox("Select a numerical column:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            df[selected_num_col].hist(bins=30, edgecolor='black', ax=ax)
            ax.set_title(f'Histogram of {selected_num_col}')
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=selected_num_col, ax=ax)
            ax.set_title(f'Boxplot of {selected_num_col}')
            ax.set_ylabel(selected_num_col)
            st.pyplot(fig)
    
    # Categorical plots
    if categorical_cols:
        st.subheader("Categorical Features Analysis")
        selected_cat_col = st.selectbox("Select a categorical column:", categorical_cols)
        
        value_counts = df[selected_cat_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Count of {selected_cat_col}')
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        with col2:
            fig = px.pie(values=value_counts.values, names=value_counts.index, 
                        title=f'Distribution of {selected_cat_col}')
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    if len(numeric_cols) > 1:
        st.markdown("---")
        st.header("5️⃣ Correlation Analysis")
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)
        
        # Scatter plot for relationship
        st.subheader("Relationship between Features")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis:", numeric_cols, key='x')
        with col2:
            y_axis = st.selectbox("Select Y-axis:", numeric_cols, key='y')
        
        if x_axis and y_axis:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{x_axis} vs {y_axis}')
            st.plotly_chart(fig, use_container_width=True)
    
    # Download processed data
    st.markdown("---")
    st.header("6️⃣ Export Data")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )