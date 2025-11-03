"""
================================================================================
PROJECT: Climate Change Impact on Agriculture — Country-Focused EDA
COURSE: 2nd-Year Engineering Mini-Project
================================================================================

PURPOSE:
This Streamlit app performs Exploratory Data Analysis (EDA) on climate and 
agriculture data. Focus: single country analysis to understand how climate 
change affects agricultural productivity over time.

DATASET ASSUMPTIONS:
Expected columns (handles missing gracefully):
- Country, Region, Year, Crop_Type, Crop_Yield_MT_per_HA
- Average_Temperature_C, Total_Precipitation_mm, CO2_Emissions_MT
- Extreme_Weather_Events, Irrigation_Access_%, Fertilizer_Use_KG_per_HA
- Pesticide_Use_KG_per_HA, Soil_Health_Index, Adaptation_Strategies
- Economic_Impact_Million_USD

HOW TO RUN:
1. Place your CSV file in: data/climate_change_impact_on_agriculture_2024.csv
2. Install dependencies: pip install streamlit pandas numpy plotly scipy seaborn
3. Run: streamlit run streamlit_app.py
4. Select country in Tab 1, then explore other tabs

AUTHOR: Student Project Template
DATE: 2024
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Climate-Agriculture EDA",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS - DATA LOADING & CACHING
# ============================================================================

@st.cache_data
def load_data():
    """
    Load the dataset from CSV file.
    Returns: DataFrame or None if file not found
    """
    try:
        df = pd.read_csv("data/climate_change_impact_on_agriculture_2024.csv")
        return df
    except FileNotFoundError:
        return None

def filter_country(df, country_name):
    """
    Filter dataset for a specific country.
    Args:
        df: Full dataset
        country_name: Selected country
    Returns: Filtered DataFrame
    """
    return df[df['Country'] == country_name].copy()

# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def clean_data(df):
    """
    Clean and preprocess the filtered country data.
    Steps:
    1. Convert Year to integer
    2. Convert numeric columns properly
    3. Handle missing values
    4. Detect outliers
    5. Create derived features
    
    Returns: cleaned_df, cleaning_report (dict)
    """
    cleaning_report = {}
    df_clean = df.copy()
    
    # Store original row count
    original_rows = len(df_clean)
    cleaning_report['original_rows'] = original_rows
    
    # STEP 1: Convert Year to integer
    if 'Year' in df_clean.columns:
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Year'])
        df_clean['Year'] = df_clean['Year'].astype(int)
    
    # STEP 2: Identify and convert numeric columns
    numeric_cols = [
        'Crop_Yield_MT_per_HA', 'Average_Temperature_C', 'Total_Precipitation_mm',
        'CO2_Emissions_MT', 'Extreme_Weather_Events', 'Irrigation_Access_%',
        'Fertilizer_Use_KG_per_HA', 'Pesticide_Use_KG_per_HA', 
        'Soil_Health_Index', 'Economic_Impact_Million_USD'
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # STEP 3: Handle missing values
    # Drop rows missing essential fields
    essential_cols = ['Year', 'Crop_Type', 'Crop_Yield_MT_per_HA']
    existing_essential = [col for col in essential_cols if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=existing_essential)
    
    # Count missing values before imputation
    missing_before = df_clean.isnull().sum().sum()
    cleaning_report['missing_before_imputation'] = missing_before
    
    # Impute climate variables using forward/backward fill per region if possible
    climate_vars = ['Average_Temperature_C', 'Total_Precipitation_mm']
    for col in climate_vars:
        if col in df_clean.columns and df_clean[col].isnull().any():
            if 'Region' in df_clean.columns:
                df_clean[col] = df_clean.groupby('Region')[col].fillna(method='ffill').fillna(method='bfill')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Impute other variables with median
    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    missing_after = df_clean.isnull().sum().sum()
    cleaning_report['missing_after_imputation'] = missing_after
    
    # STEP 4: Outlier detection using IQR for yield
    if 'Crop_Yield_MT_per_HA' in df_clean.columns:
        Q1 = df_clean['Crop_Yield_MT_per_HA'].quantile(0.25)
        Q3 = df_clean['Crop_Yield_MT_per_HA'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[
            (df_clean['Crop_Yield_MT_per_HA'] < lower_bound) | 
            (df_clean['Crop_Yield_MT_per_HA'] > upper_bound)
        ]
        cleaning_report['outlier_count'] = len(outliers)
        df_clean['is_outlier'] = (
            (df_clean['Crop_Yield_MT_per_HA'] < lower_bound) | 
            (df_clean['Crop_Yield_MT_per_HA'] > upper_bound)
        )
    
    # STEP 5: Create derived features
    # Yield anomaly (deviation from 5-year rolling mean)
    if 'Crop_Yield_MT_per_HA' in df_clean.columns and 'Crop_Type' in df_clean.columns:
        df_clean = df_clean.sort_values(['Crop_Type', 'Year'])
        df_clean['yield_rolling_mean'] = df_clean.groupby('Crop_Type')['Crop_Yield_MT_per_HA'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df_clean['yield_anomaly'] = df_clean['Crop_Yield_MT_per_HA'] - df_clean['yield_rolling_mean']
    
    # Final row count
    cleaning_report['final_rows'] = len(df_clean)
    cleaning_report['rows_removed'] = original_rows - len(df_clean)
    
    return df_clean, cleaning_report

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_correlation_table(df, target_col='Crop_Yield_MT_per_HA'):
    """
    Compute correlation between climate/input variables and yield.
    Returns: DataFrame with correlations and p-values
    """
    feature_cols = [
        'Average_Temperature_C', 'Total_Precipitation_mm', 
        'Extreme_Weather_Events', 'Fertilizer_Use_KG_per_HA',
        'Irrigation_Access_%'
    ]
    
    # Filter columns that exist in dataframe
    existing_features = [col for col in feature_cols if col in df.columns]
    
    if target_col not in df.columns or len(existing_features) == 0:
        return None
    
    results = []
    for feature in existing_features:
        # Remove NaN values
        valid_data = df[[feature, target_col]].dropna()
        if len(valid_data) > 2:
            pearson_corr, pearson_p = pearsonr(valid_data[feature], valid_data[target_col])
            spearman_corr, spearman_p = spearmanr(valid_data[feature], valid_data[target_col])
            
            results.append({
                'Feature': feature,
                'Pearson_r': round(pearson_corr, 3),
                'Pearson_p': round(pearson_p, 4),
                'Spearman_r': round(spearman_corr, 3),
                'Spearman_p': round(spearman_p, 4),
                'Sample_Size': len(valid_data)
            })
    
    return pd.DataFrame(results)

def compute_linear_regression(x, y):
    """
    Compute linear regression statistics.
    Returns: slope, intercept, r_value, p_value, r_squared
    """
    valid_idx = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    
    if len(x_valid) < 3:
        return None, None, None, None, None
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
    r_squared = r_value ** 2
    
    return slope, intercept, r_value, p_value, r_squared

# ============================================================================
# INSIGHT GENERATION
# ============================================================================

def generate_insights(df):
    """
    Generate key insights from the analysis.
    Returns: List of insight strings with evidence
    """
    insights = []
    
    # Insight 1: Temperature trend
    if 'Average_Temperature_C' in df.columns and 'Year' in df.columns:
        temp_start = df[df['Year'] == df['Year'].min()]['Average_Temperature_C'].mean()
        temp_end = df[df['Year'] == df['Year'].max()]['Average_Temperature_C'].mean()
        temp_change = temp_end - temp_start
        year_start = df['Year'].min()
        year_end = df['Year'].max()
        
        insights.append(
            f"🌡️ **Temperature Trend**: Average temperature changed by {temp_change:.2f}°C "
            f"between {year_start} and {year_end} (from {temp_start:.1f}°C to {temp_end:.1f}°C)."
        )
    
    # Insight 2: Yield trend
    if 'Crop_Yield_MT_per_HA' in df.columns and 'Year' in df.columns:
        yield_start = df[df['Year'] == df['Year'].min()]['Crop_Yield_MT_per_HA'].mean()
        yield_end = df[df['Year'] == df['Year'].max()]['Crop_Yield_MT_per_HA'].mean()
        yield_change_pct = ((yield_end - yield_start) / yield_start) * 100
        
        insights.append(
            f"🌾 **Yield Trend**: Average crop yield changed by {yield_change_pct:.1f}% "
            f"(from {yield_start:.2f} to {yield_end:.2f} MT/HA)."
        )
    
    # Insight 3: Temperature-Yield correlation
    if 'Average_Temperature_C' in df.columns and 'Crop_Yield_MT_per_HA' in df.columns:
        valid_data = df[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']].dropna()
        if len(valid_data) > 2:
            corr, p_val = pearsonr(valid_data['Average_Temperature_C'], valid_data['Crop_Yield_MT_per_HA'])
            slope, _, _, _, r_sq = compute_linear_regression(
                valid_data['Average_Temperature_C'].values,
                valid_data['Crop_Yield_MT_per_HA'].values
            )
            
            if slope is not None:
                insights.append(
                    f"🔗 **Climate-Yield Relationship**: Temperature shows correlation of {corr:.3f} "
                    f"with crop yield (R²={r_sq:.3f}, p={p_val:.4f}). "
                    f"Each 1°C increase associates with {slope:.3f} MT/HA yield change."
                )
    
    # Insight 4: Extreme weather impact
    if 'Extreme_Weather_Events' in df.columns and 'Crop_Yield_MT_per_HA' in df.columns:
        high_events = df[df['Extreme_Weather_Events'] >= 2]['Crop_Yield_MT_per_HA'].mean()
        low_events = df[df['Extreme_Weather_Events'] < 2]['Crop_Yield_MT_per_HA'].mean()
        
        if not np.isnan(high_events) and not np.isnan(low_events):
            diff_pct = ((high_events - low_events) / low_events) * 100
            insights.append(
                f"⚠️ **Extreme Weather Impact**: Years with 2+ extreme events show "
                f"{diff_pct:.1f}% {'lower' if diff_pct < 0 else 'higher'} average yield "
                f"({high_events:.2f} vs {low_events:.2f} MT/HA)."
            )
    
    # Insight 5: Best performing crop
    if 'Crop_Type' in df.columns and 'Crop_Yield_MT_per_HA' in df.columns:
        crop_yields = df.groupby('Crop_Type')['Crop_Yield_MT_per_HA'].mean().sort_values(ascending=False)
        if len(crop_yields) > 0:
            best_crop = crop_yields.index[0]
            best_yield = crop_yields.values[0]
            insights.append(
                f"🏆 **Top Performing Crop**: {best_crop} shows highest average yield "
                f"at {best_yield:.2f} MT/HA."
            )
    
    # Insight 6: Irrigation impact
    if 'Irrigation_Access_%' in df.columns:
        irr_start = df[df['Year'] == df['Year'].min()]['Irrigation_Access_%'].mean()
        irr_end = df[df['Year'] == df['Year'].max()]['Irrigation_Access_%'].mean()
        irr_change = irr_end - irr_start
        
        insights.append(
            f"💧 **Irrigation Access**: Changed by {irr_change:.1f} percentage points "
            f"(from {irr_start:.1f}% to {irr_end:.1f}%)."
        )
    
    return insights

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🌾 Climate Change Impact on Agriculture")
    st.markdown("**2nd-Year Engineering Mini-Project | Country-Focused EDA**")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("❌ Dataset not found! Please place your CSV file in: `data/climate_change_impact_on_agriculture_2024.csv`")
        st.stop()
    
    # Initialize session state
    if 'cleaned_df' not in st.session_state:
        st.session_state['cleaned_df'] = None
    if 'selected_country' not in st.session_state:
        st.session_state['selected_country'] = None
    if 'cleaning_report' not in st.session_state:
        st.session_state['cleaning_report'] = None
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Country Overview",
        "🌡️ Climate Trends",
        "🌾 Agriculture Trends",
        "🔗 Climate vs Agriculture",
        "💡 Key Insights & Future Scope"
    ])
    
    # ========================================================================
    # TAB 1: COUNTRY OVERVIEW
    # ========================================================================
    with tab1:
        st.header("1️⃣ Country Selection & Data Overview")
        st.markdown("Select a country to analyze its climate-agriculture relationship.")
        
        # Get list of countries
        countries = sorted(df['Country'].unique())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_country = st.selectbox(
                "🌍 Select Country:",
                options=countries,
                index=0,
                key='country_selector'
            )
        
        with col2:
            load_button = st.button("🔄 Load & Clean Data", type="primary", use_container_width=True)
        
        # Load and clean data when button clicked
        if load_button or (st.session_state['selected_country'] != selected_country):
            with st.spinner(f"Loading and cleaning data for {selected_country}..."):
                # Filter for selected country
                df_country = filter_country(df, selected_country)
                
                # Clean the data
                df_cleaned, report = clean_data(df_country)
                
                # Store in session state
                st.session_state['cleaned_df'] = df_cleaned
                st.session_state['selected_country'] = selected_country
                st.session_state['cleaning_report'] = report
                
                st.success(f"✅ Data loaded and cleaned for **{selected_country}**!")
        
        # Display information if data is loaded
        if st.session_state['cleaned_df'] is not None:
            df_clean = st.session_state['cleaned_df']
            report = st.session_state['cleaning_report']
            
            st.markdown("---")
            st.subheader("📊 Data Cleaning Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rows", report['original_rows'])
            with col2:
                st.metric("Final Rows", report['final_rows'])
            with col3:
                st.metric("Rows Removed", report['rows_removed'])
            with col4:
                st.metric("Outliers Detected", report.get('outlier_count', 0))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing Values (Before)", report['missing_before_imputation'])
            with col2:
                st.metric("Missing Values (After)", report['missing_after_imputation'])
            
            st.markdown("---")
            st.subheader("📈 Dataset Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df_clean))
            with col2:
                years_covered = f"{df_clean['Year'].min()} - {df_clean['Year'].max()}"
                st.metric("Years Covered", years_covered)
            with col3:
                unique_crops = df_clean['Crop_Type'].nunique() if 'Crop_Type' in df_clean.columns else 0
                st.metric("Unique Crops", unique_crops)
            with col4:
                unique_regions = df_clean['Region'].nunique() if 'Region' in df_clean.columns else 0
                st.metric("Regions", unique_regions)
            
            st.markdown("---")
            st.subheader("🔍 Column Information")
            
            # Create column info table
            col_info = []
            for col in df_clean.columns:
                missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
                example_val = df_clean[col].dropna().iloc[0] if len(df_clean[col].dropna()) > 0 else "N/A"
                col_info.append({
                    'Column Name': col,
                    'Data Type': str(df_clean[col].dtype),
                    '% Missing': f"{missing_pct:.1f}%",
                    'Example Value': str(example_val)[:50]
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True, height=300)
            
            st.markdown("---")
            st.subheader("📄 Data Preview (First 50 Rows)")
            st.dataframe(df_clean.head(50), use_container_width=True, height=400)
            
            st.markdown("---")
            st.subheader("💾 Export Cleaned Data")
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name=f"{selected_country}_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            st.info("👆 Please select a country and click 'Load & Clean Data' to begin analysis.")
    
    # ========================================================================
    # TAB 2: CLIMATE TRENDS
    # ========================================================================
    with tab2:
        if st.session_state['cleaned_df'] is None:
            st.warning("⚠️ Please load data in the 'Country Overview' tab first!")
        else:
            df_clean = st.session_state['cleaned_df']
            country = st.session_state['selected_country']
            
            st.header(f"🌡️ Climate Trends - {country}")
            st.markdown("Explore how climate variables have changed over time.")
            
            # Controls
            col1, col2 = st.columns([3, 1])
            with col1:
                year_range = st.slider(
                    "Select Year Range:",
                    min_value=int(df_clean['Year'].min()),
                    max_value=int(df_clean['Year'].max()),
                    value=(int(df_clean['Year'].min()), int(df_clean['Year'].max())),
                    key='climate_year_range'
                )
            with col2:
                smoothing = st.toggle("Apply Smoothing", value=False, key='climate_smoothing')
            
            # Filter by year range
            df_filtered = df_clean[(df_clean['Year'] >= year_range[0]) & (df_clean['Year'] <= year_range[1])]
            
            # Aggregate by year
            df_yearly = df_filtered.groupby('Year').agg({
                'Average_Temperature_C': 'mean',
                'Total_Precipitation_mm': 'mean',
                'CO2_Emissions_MT': 'mean',
                'Extreme_Weather_Events': 'sum'
            }).reset_index()
            
            # Plot 1: Temperature over time
            st.subheader("Temperature Trend")
            if 'Average_Temperature_C' in df_yearly.columns:
                fig_temp = px.line(
                    df_yearly, 
                    x='Year', 
                    y='Average_Temperature_C',
                    title=f"Average Temperature Over Time - {country}",
                    labels={'Average_Temperature_C': 'Temperature (°C)'},
                    markers=True
                )
                
                if smoothing:
                    df_yearly['temp_smooth'] = df_yearly['Average_Temperature_C'].rolling(window=3, center=True).mean()
                    fig_temp.add_scatter(x=df_yearly['Year'], y=df_yearly['temp_smooth'], 
                                       mode='lines', name='Smoothed', line=dict(color='red', width=2))
                
                fig_temp.update_layout(hovermode='x unified', height=400)
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Interpretation
                temp_change = df_yearly['Average_Temperature_C'].iloc[-1] - df_yearly['Average_Temperature_C'].iloc[0]
                st.caption(f"📊 Temperature changed by {temp_change:.2f}°C over the selected period.")
            
            # Plot 2: Precipitation over time
            st.subheader("Precipitation Trend")
            if 'Total_Precipitation_mm' in df_yearly.columns:
                fig_precip = px.line(
                    df_yearly,
                    x='Year',
                    y='Total_Precipitation_mm',
                    title=f"Total Precipitation Over Time - {country}",
                    labels={'Total_Precipitation_mm': 'Precipitation (mm)'},
                    markers=True,
                    color_discrete_sequence=['teal']
                )
                
                if smoothing:
                    df_yearly['precip_smooth'] = df_yearly['Total_Precipitation_mm'].rolling(window=3, center=True).mean()
                    fig_precip.add_scatter(x=df_yearly['Year'], y=df_yearly['precip_smooth'],
                                         mode='lines', name='Smoothed', line=dict(color='darkgreen', width=2))
                
                fig_precip.update_layout(hovermode='x unified', height=400)
                st.plotly_chart(fig_precip, use_container_width=True)
                
                # Interpretation
                precip_change_pct = ((df_yearly['Total_Precipitation_mm'].iloc[-1] - df_yearly['Total_Precipitation_mm'].iloc[0]) / 
                                    df_yearly['Total_Precipitation_mm'].iloc[0]) * 100
                st.caption(f"📊 Precipitation changed by {precip_change_pct:.1f}% over the selected period.")
            
            # Plot 3: CO2 emissions (if available)
            st.subheader("CO₂ Emissions Trend")
            if 'CO2_Emissions_MT' in df_yearly.columns and df_yearly['CO2_Emissions_MT'].notna().any():
                fig_co2 = px.line(
                    df_yearly,
                    x='Year',
                    y='CO2_Emissions_MT',
                    title=f"CO₂ Emissions Over Time - {country}",
                    labels={'CO2_Emissions_MT': 'CO₂ Emissions (MT)'},
                    markers=True,
                    color_discrete_sequence=['darkred']
                )
                fig_co2.update_layout(hovermode='x unified', height=400)
                st.plotly_chart(fig_co2, use_container_width=True)
            else:
                st.info("CO₂ emissions data not available for this country.")
            
            # Plot 4: Extreme weather events
            st.subheader("Extreme Weather Events")
            if 'Extreme_Weather_Events' in df_yearly.columns:
                fig_events = px.bar(
                    df_yearly,
                    x='Year',
                    y='Extreme_Weather_Events',
                    title=f"Extreme Weather Events Frequency - {country}",
                    labels={'Extreme_Weather_Events': 'Number of Events'},
                    color='Extreme_Weather_Events',
                    color_continuous_scale='Reds'
                )
                fig_events.update_layout(hovermode='x unified', height=400)
                st.plotly_chart(fig_events, use_container_width=True)
                
                # Interpretation
                avg_events = df_yearly['Extreme_Weather_Events'].mean()
                st.caption(f"📊 Average of {avg_events:.1f} extreme weather events per year.")
    
    # ========================================================================
    # TAB 3: AGRICULTURE TRENDS
    # ========================================================================
    with tab3:
        if st.session_state['cleaned_df'] is None:
            st.warning("⚠️ Please load data in the 'Country Overview' tab first!")
        else:
            df_clean = st.session_state['cleaned_df']
            country = st.session_state['selected_country']
            
            st.header(f"🌾 Agriculture Trends - {country}")
            st.markdown("Analyze agricultural productivity and input usage patterns.")
            
            # Controls
            if 'Crop_Type' in df_clean.columns:
                available_crops = sorted(df_clean['Crop_Type'].unique())
                selected_crops = st.multiselect(
                    "Select Crops to Display (Max 4):",
                    options=available_crops,
                    default=available_crops[:min(4, len(available_crops))],
                    max_selections=4,
                    key='agri_crops'
                )
            else:
                selected_crops = []
            
            # Plot 1: Yield trends by crop
            st.subheader("Crop Yield Trends")
            if 'Crop_Type' in df_clean.columns and 'Crop_Yield_MT_per_HA' in df_clean.columns and len(selected_crops) > 0:
                df_crops = df_clean[df_clean['Crop_Type'].isin(selected_crops)]
                df_crop_yearly = df_crops.groupby(['Year', 'Crop_Type'])['Crop_Yield_MT_per_HA'].mean().reset_index()
                
                fig_yield = px.line(
                    df_crop_yearly,
                    x='Year',
                    y='Crop_Yield_MT_per_HA',
                    color='Crop_Type',
                    title=f"Crop Yield Over Time by Crop Type - {country}",
                    labels={'Crop_Yield_MT_per_HA': 'Yield (MT/HA)', 'Crop_Type': 'Crop'},
                    markers=True
                )
                fig_yield.update_layout(hovermode='x unified', height=450)
                st.plotly_chart(fig_yield, use_container_width=True)
                
                # Top crops by yield change
                st.markdown("**Top Crops by Yield Change (Last 10 Years):**")
                recent_years = df_clean['Year'].max()
                df_recent = df_clean[df_clean['Year'] >= recent_years - 10]
                df_old = df_clean[df_clean['Year'] < recent_years - 10]
                
                if len(df_old) > 0:
                    recent_yields = df_recent.groupby('Crop_Type')['Crop_Yield_MT_per_HA'].mean()
                    old_yields = df_old.groupby('Crop_Type')['Crop_Yield_MT_per_HA'].mean()
                    yield_change = ((recent_yields - old_yields) / old_yields * 100).sort_values(ascending=False).head(3)
                    
                    for crop, change in yield_change.items():
                        st.write(f"- **{crop}**: {change:+.1f}% change")
            
            # Plot 2: Yield distribution by year
            st.subheader("Yield Distribution Analysis")
            if 'Crop_Yield_MT_per_HA' in df_clean.columns and len(selected_crops) > 0:
                df_crops = df_clean[df_clean['Crop_Type'].isin(selected_crops)]
                
                fig_box = px.box(
                    df_crops,
                    x='Crop_Type',
                    y='Crop_Yield_MT_per_HA',
                    color='Crop_Type',
                    title=f"Yield Distribution by Crop Type - {country}",
                    labels={'Crop_Yield_MT_per_HA': 'Yield (MT/HA)', 'Crop_Type': 'Crop'}
                )
                fig_box.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Plot 3: Fertilizer use trend
            st.subheader("Fertilizer Usage Trend")
            if 'Fertilizer_Use_KG_per_HA' in df_clean.columns:
                df_fert = df_clean.groupby('Year')['Fertilizer_Use_KG_per_HA'].mean().reset_index()
                
                fig_fert = px.line(
                    df_fert,
                    x='Year',
                    y='Fertilizer_Use_KG_per_HA',
                    title=f"Average Fertilizer Use Over Time - {country}",
                    labels={'Fertilizer_Use_KG_per_HA': 'Fertilizer (KG/HA)'},
                    markers=True,
                    color_discrete_sequence=['green']
                )
                fig_fert.update_layout(hovermode='x unified', height=400)
                st.plotly_chart(fig_fert, use_container_width=True)
                
                fert_change = df_fert['Fertilizer_Use_KG_per_HA'].iloc[-1] - df_fert['Fertilizer_Use_KG_per_HA'].iloc[0]
                st.caption(f"📊 Fertilizer use changed by {fert_change:.1f} KG/HA over the period.")
            
            # Plot 4: Irrigation access trend
            st.subheader("Irrigation Access Trend")
            if 'Irrigation_Access_%' in df_clean.columns:
                df_irr = df_clean.groupby('Year')['Irrigation_Access_%'].mean().reset_index()
                
                fig_irr = px.line(
                    df_irr,
                    x='Year',
                    y='Irrigation_Access_%',
                    title=f"Irrigation Access Over Time - {country}",
                    labels={'Irrigation_Access_%': 'Irrigation Access (%)'},
                    markers=True,
                    color_discrete_sequence=['dodgerblue']
                )
                fig_irr.update_layout(hovermode='x unified', height=400)
                st.plotly_chart(fig_irr, use_container_width=True)
                
                latest_irr = df_irr['Irrigation_Access_%'].iloc[-1]
                st.caption(f"📊 Latest irrigation access: {latest_irr:.1f}%")
    
    # ========================================================================
    # TAB 4: CLIMATE VS AGRICULTURE RELATIONSHIP
    # ========================================================================
    with tab4:
        if st.session_state['cleaned_df'] is None:
            st.warning("⚠️ Please load data in the 'Country Overview' tab first!")
        else:
            df_clean = st.session_state['cleaned_df']
            country = st.session_state['selected_country']
            
            st.header(f"🔗 Climate vs Agriculture Relationship - {country}")
            st.markdown("Analyze correlations between climate factors and agricultural productivity.")
            
            # Plot 1: Temperature vs Yield
            st.subheader("Temperature vs Crop Yield")
            if 'Average_Temperature_C' in df_clean.columns and 'Crop_Yield_MT_per_HA' in df_clean.columns:
                df_plot = df_clean[['Average_Temperature_C', 'Crop_Yield_MT_per_HA', 'Crop_Type']].dropna()
                
                fig_temp_yield = px.scatter(
                    df_plot,
                    x='Average_Temperature_C',
                    y='Crop_Yield_MT_per_HA',
                    color='Crop_Type',
                    title=f"Temperature vs Yield - {country}",
                    labels={'Average_Temperature_C': 'Temperature (°C)', 'Crop_Yield_MT_per_HA': 'Yield (MT/HA)'},
                    trendline='ols',
                    opacity=0.6
                )
                fig_temp_yield.update_layout(height=450)
                st.plotly_chart(fig_temp_yield, use_container_width=True)
                
                # Compute statistics
                slope, intercept, r_value, p_value, r_squared = compute_linear_regression(
                    df_plot['Average_Temperature_C'].values,
                    df_plot['Crop_Yield_MT_per_HA'].values
                )
                
                if slope is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Slope", f"{slope:.4f}")
                    with col2:
                        st.metric("R²", f"{r_squared:.4f}")
                    with col3:
                        st.metric("Correlation", f"{r_value:.4f}")
                    with col4:
                        st.metric("P-value", f"{p_value:.4f}")
                    
                    st.caption(f"📊 Interpretation: Each 1°C temperature increase is associated with {slope:.4f} MT/HA yield change. "
                             f"R² = {r_squared:.3f} indicates {'strong' if r_squared > 0.5 else 'moderate' if r_squared > 0.3 else 'weak'} relationship.")
            
            # Plot 2: Precipitation vs Yield
            st.subheader("Precipitation vs Crop Yield")
            if 'Total_Precipitation_mm' in df_clean.columns and 'Crop_Yield_MT_per_HA' in df_clean.columns:
                df_plot = df_clean[['Total_Precipitation_mm', 'Crop_Yield_MT_per_HA', 'Crop_Type']].dropna()
                
                fig_precip_yield = px.scatter(
                    df_plot,
                    x='Total_Precipitation_mm',
                    y='Crop_Yield_MT_per_HA',
                    color='Crop_Type',
                    title=f"Precipitation vs Yield - {country}",
                    labels={'Total_Precipitation_mm': 'Precipitation (mm)', 'Crop_Yield_MT_per_HA': 'Yield (MT/HA)'},
                    trendline='lowess',
                    opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_precip_yield.update_layout(height=450)
                st.plotly_chart(fig_precip_yield, use_container_width=True)
                
                # Compute statistics
                slope, intercept, r_value, p_value, r_squared = compute_linear_regression(
                    df_plot['Total_Precipitation_mm'].values,
                    df_plot['Crop_Yield_MT_per_HA'].values
                )
                
                if slope is not None:
                    st.caption(f"📊 Correlation between precipitation and yield: r = {r_value:.3f}, R² = {r_squared:.3f}, p = {p_value:.4f}")
            
            # Plot 3: Extreme weather impact
            st.subheader("Extreme Weather Impact on Yield")
            if 'Extreme_Weather_Events' in df_clean.columns and 'Crop_Yield_MT_per_HA' in df_clean.columns:
                # Create categories
                df_clean['Event_Category'] = pd.cut(
                    df_clean['Extreme_Weather_Events'],
                    bins=[-0.1, 0.9, 1.9, 100],
                    labels=['0 Events', '1 Event', '2+ Events']
                )
                
                df_plot = df_clean[['Event_Category', 'Crop_Yield_MT_per_HA']].dropna()
                
                fig_events = px.box(
                    df_plot,
                    x='Event_Category',
                    y='Crop_Yield_MT_per_HA',
                    title=f"Yield Distribution by Extreme Weather Events - {country}",
                    labels={'Event_Category': 'Extreme Weather Events', 'Crop_Yield_MT_per_HA': 'Yield (MT/HA)'},
                    color='Event_Category',
                    color_discrete_sequence=['lightgreen', 'orange', 'red']
                )
                fig_events.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_events, use_container_width=True)
                
                # Show mean yields
                mean_yields = df_plot.groupby('Event_Category')['Crop_Yield_MT_per_HA'].mean()
                st.caption(f"📊 Average yields: " + " | ".join([f"{cat}: {val:.2f} MT/HA" for cat, val in mean_yields.items()]))
            
            # Plot 4: Correlation heatmap
            st.subheader("Correlation Heatmap")
            corr_features = [
                'Average_Temperature_C', 'Total_Precipitation_mm', 'Extreme_Weather_Events',
                'Fertilizer_Use_KG_per_HA', 'Irrigation_Access_%', 'Crop_Yield_MT_per_HA'
            ]
            existing_features = [col for col in corr_features if col in df_clean.columns]
            
            if len(existing_features) > 2:
                corr_matrix = df_clean[existing_features].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    title=f"Feature Correlation Matrix - {country}"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation table
            st.subheader("Detailed Correlation Analysis")
            corr_table = compute_correlation_table(df_clean)
            if corr_table is not None:
                st.dataframe(corr_table, use_container_width=True)
                
                st.caption("""
                📊 **Interpretation Guide:**
                - Pearson r: Linear correlation (-1 to +1)
                - Spearman r: Monotonic correlation (robust to outliers)
                - P-value < 0.05: Statistically significant
                - |r| > 0.5: Strong correlation
                - |r| 0.3-0.5: Moderate correlation
                - |r| < 0.3: Weak correlation
                """)
    
    # ========================================================================
    # TAB 5: KEY INSIGHTS & FUTURE SCOPE
    # ========================================================================
    with tab5:
        if st.session_state['cleaned_df'] is None:
            st.warning("⚠️ Please load data in the 'Country Overview' tab first!")
        else:
            df_clean = st.session_state['cleaned_df']
            country = st.session_state['selected_country']
            
            st.header(f"💡 Key Insights & Future Scope - {country}")
            
            # Generate insights
            insights = generate_insights(df_clean)
            
            st.subheader("🔍 Key Findings")
            st.markdown("Evidence-based insights from the analysis:")
            
            for i, insight in enumerate(insights, 1):
                st.markdown(f"**{i}.** {insight}")
            
            st.markdown("---")
            st.subheader("📋 Practical Recommendations")
            
            recommendations = [
                "**Climate Adaptation**: Implement crop varieties resistant to temperature extremes and water stress.",
                "**Irrigation Investment**: Expand irrigation infrastructure in regions with declining precipitation.",
                "**Fertilizer Optimization**: Balance fertilizer use to maximize yield while minimizing environmental impact.",
                "**Early Warning Systems**: Develop systems to predict and prepare for extreme weather events."
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            st.markdown("---")
            st.subheader("🚀 Future Scope")
            
            st.markdown("""
            This analysis can be extended in several directions:
            
            **1. Regional Deep-Dive Analysis**
            - Perform detailed analysis for each region within the country
            - Identify region-specific vulnerabilities and opportunities
            - Compare adaptation strategies across regions
            
            **2. Time-Lagged Analysis**
            - Study delayed effects of climate change on agriculture
            - Analyze multi-season impacts (e.g., winter temperature affecting spring yields)
            - Model temporal causality relationships
            
            **3. Machine Learning & Prediction**
            - Build predictive models for crop yield forecasting
            - Classify regions by climate risk levels
            - Develop early warning systems using ML algorithms
            
            **4. Economic Impact Modeling**
            - Quantify economic losses from climate change
            - Cost-benefit analysis of adaptation strategies
            - Policy impact assessment
            
            **5. Multi-Country Comparison**
            - Compare how different countries are affected
            - Learn from successful adaptation strategies globally
            - Identify global climate-agriculture patterns
            
            **6. Data Integration**
            - Incorporate satellite imagery data
            - Add soil quality and water availability data
            - Include socio-economic indicators
            """)
            
            st.markdown("---")
            st.subheader("💾 Export Insights")
            
            # Create insights text file
            insights_text = f"Climate Change Impact on Agriculture - {country}\n"
            insights_text += "=" * 60 + "\n\n"
            insights_text += "KEY FINDINGS:\n"
            insights_text += "-" * 60 + "\n"
            for i, insight in enumerate(insights, 1):
                # Remove markdown formatting
                clean_insight = insight.replace("**", "").replace("🌡️", "").replace("🌾", "").replace("🔗", "").replace("⚠️", "").replace("🏆", "").replace("💧", "")
                insights_text += f"{i}. {clean_insight}\n\n"
            
            insights_text += "\nRECOMMENDATIONS:\n"
            insights_text += "-" * 60 + "\n"
            for rec in recommendations:
                clean_rec = rec.replace("**", "")
                insights_text += f"{clean_rec}\n\n"
            
            st.download_button(
                label="📥 Download Insights Report (TXT)",
                data=insights_text,
                file_name=f"{country}_climate_agriculture_insights.txt",
                mime="text/plain",
                use_container_width=True
            )

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
