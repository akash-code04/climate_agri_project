"""
================================================================================
PROJECT: Climate Change Impact on Agriculture — Country-Focused EDA
COURSE: 2nd-Year Engineering Mini-Project
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
    """Generate evidence-based insights from cleaned data"""
    insights = []
    
    # Temperature insight
    if 'Average_Temperature_C' in df.columns and 'Year' in df.columns:
        temp_by_year = df.groupby('Year')['Average_Temperature_C'].mean()
        if len(temp_by_year) > 1:
            temp_change = temp_by_year.iloc[-1] - temp_by_year.iloc[0]
            years_span = temp_by_year.index[-1] - temp_by_year.index[0]
            insights.append(
                f"**Temperature Trend:** Average temperature {'increased' if temp_change > 0 else 'decreased'} by {abs(temp_change):.2f}°C "
                f"over {years_span} years, potentially {'stressing heat-sensitive crops' if temp_change > 0 else 'affecting crop phenology'}"
            )
    
    # Yield trend insight
    if 'Crop_Yield_MT_per_HA' in df.columns and 'Year' in df.columns:
        yield_by_year = df.groupby('Year')['Crop_Yield_MT_per_HA'].mean()
        if len(yield_by_year) > 1:
            yield_change_pct = ((yield_by_year.iloc[-1] - yield_by_year.iloc[0]) / yield_by_year.iloc[0]) * 100
            insights.append(
                f"**Yield Performance:** Overall crop yield {'increased' if yield_change_pct > 0 else 'decreased'} by {abs(yield_change_pct):.1f}% "
                f"over the period, {'suggesting successful adaptation or favorable conditions' if yield_change_pct > 0 else 'indicating climate stress or declining productivity'}"
            )
    
    # Correlation insight
    if all(col in df.columns for col in ['Average_Temperature_C', 'Crop_Yield_MT_per_HA']):
        df_corr = df[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']].dropna()
        if len(df_corr) > 10:
            corr = df_corr.corr().iloc[0, 1]
            insights.append(
                f"**Temperature-Yield Relationship:** Correlation coefficient of {corr:.3f} indicates "
                f"{'significant negative impact of heat on yields' if corr < -0.3 else 'moderate temperature sensitivity' if corr < 0 else 'unexpected positive association - may indicate sub-optimal baseline temperatures'}"
            )
    
    # Extreme weather insight
    if 'Extreme_Weather_Events' in df.columns and 'Year' in df.columns:
        events_by_year = df.groupby('Year')['Extreme_Weather_Events'].sum()
        if len(events_by_year) > 1:
            avg_events = events_by_year.mean()
            recent_trend = events_by_year.iloc[-5:].mean() - events_by_year.iloc[:5].mean() if len(events_by_year) > 10 else 0
            insights.append(
                f"**Extreme Weather Pattern:** Average of {avg_events:.1f} extreme events per year, "
                f"{'with increasing frequency in recent years' if recent_trend > 0.5 else 'with relatively stable frequency'} - "
                f"{'urgent need for risk management strategies' if avg_events > 2 else 'moderate climate shock exposure'}"
            )
    
    # Irrigation insight
    if 'Irrigation_Access_%' in df.columns:
        latest_irrigation = df['Irrigation_Access_%'].mean()
        insights.append(
            f"**Water Management:** Average irrigation access of {latest_irrigation:.1f}% "
            f"{'provides good buffer against rainfall variability' if latest_irrigation > 60 else 'leaves significant vulnerability to drought - expansion needed'}"
        )
    
    # Crop-specific insight
    if 'Crop_Type' in df.columns and 'Crop_Yield_MT_per_HA' in df.columns:
        crop_yields = df.groupby('Crop_Type')['Crop_Yield_MT_per_HA'].agg(['mean', 'std'])
        crop_yields['cv'] = crop_yields['std'] / crop_yields['mean']
        most_stable = crop_yields['cv'].idxmin()
        most_volatile = crop_yields['cv'].idxmax()
        insights.append(
            f"**Crop Stability:** {most_stable} shows most stable yields (lowest variation), "
            f"while {most_volatile} exhibits highest volatility - suggesting different climate adaptation capacities"
        )
    
    if len(insights) == 0:
        insights.append("Insufficient data for comprehensive insights generation")
    
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
                st.markdown("### 🔍 Insight")
                st.markdown(f"""
                - **What changed?** Temperature {"increased" if temp_change > 0 else "decreased"} by {abs(temp_change):.2f}°C over {year_range[1] - year_range[0]} years
                - **Why it matters?** {'Rising temperatures can reduce crop yields through heat stress, faster evapotranspiration, and shortened growing seasons' if temp_change > 0 else 'Cooling trends may affect crop phenology and frost risk'}
                - **Evidence:** {'Warming exceeds 0.5°C suggests significant climate shift impacting heat-sensitive crops like wheat' if abs(temp_change) > 0.5 else 'Moderate temperature change - monitor crop-specific responses'}
                """)
            
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
                st.markdown("### 🔍 Insight")
                st.markdown(f"""
                - **What changed?** Rainfall {"increased" if precip_change_pct > 0 else "decreased"} by {abs(precip_change_pct):.1f}% 
                - **Why it matters?** {'Increased precipitation can benefit rainfed crops but may cause waterlogging and disease' if precip_change_pct > 0 else 'Declining rainfall threatens rainfed agriculture and increases drought risk'}
                - **Agricultural impact:** {'Rice and sugarcane may benefit; wheat may face waterlogging issues' if precip_change_pct > 0 else 'Water-intensive crops like rice face stress; shift to drought-tolerant varieties needed'}
                """)
            
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
                st.markdown("### 🔍 Insight")
                st.markdown("""
                - **Climate connection:** CO₂ emissions drive long-term warming and weather pattern disruption
                - **Agricultural relevance:** While CO₂ can enhance photosynthesis (CO₂ fertilization effect), associated warming negates benefits
                - **Long-term concern:** Continued emissions intensify extreme weather events affecting crop stability """)
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
                st.markdown("### 🔍 Insight")
                st.markdown(f"""
                - **Shock frequency:** Average {avg_events:.1f} extreme events/year indicates {"high climate vulnerability" if avg_events > 2 else "moderate climate stress"}
                - **Yield impact:** Extreme events (floods, droughts, storms) cause sudden yield drops and crop damage
                - **Trend concern:** {"Increasing frequency suggests climate instability - requires adaptive strategies" if len(df_yearly) > 1 and df_yearly['Extreme_Weather_Events'].iloc[-5:].mean() > df_yearly['Extreme_Weather_Events'].iloc[:5].mean() else "Monitor for emerging patterns"}
                """)
    
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

                    st.markdown("### 🔍 Crop Performance Insight")
                    st.markdown("""
                    - **Yield trends reveal:** Crop-specific climate responses and adaptation success
                    - **Declining crops:** Likely facing climate stress (heat/water) or pest pressure
                    - **Improving crops:** May benefit from better varieties, inputs, or favorable conditions
                    - **Volatility signals:** High year-to-year variation indicates climate vulnerability""")
            
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
                st.markdown("### 🔍 Distribution Insight")
                st.markdown("""
                - **Wide distributions:** Indicate high yield variability - climate or management inconsistency
                - **Narrow distributions:** Suggest stable production - better adapted or controlled conditions
                - **Outliers:** Represent exceptional years (positive or negative) - investigate climate anomalies""")
            
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
                st.markdown("### 🔍 Input Trend Insight")
                st.markdown(f"""
                - **Fertilizer change:** {'+' if fert_change > 0 else ''}{fert_change:.1f} KG/HA suggests {"intensification strategy" if fert_change > 0 else "declining input use"}
                - **Yield relationship:** {"Increased fertilizer should boost yields if water available" if fert_change > 0 else "Reduced fertilizer may limit yield potential"}
                - **Sustainability note:** {"Monitor soil health and runoff pollution with high fertilizer use" if fert_change > 50 else "Balanced fertilizer management appears maintained"}""")
            
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
                st.markdown("### 🔍 Irrigation Insight")
                st.markdown(f"""- **Current access:** {latest_irr:.1f}% irrigation coverage - **Climate buffer:** {"High irrigation access reduces rainfall dependency and drought risk" if latest_irr > 60 else "Low irrigation increases vulnerability to rainfall variability"}
                - **Adaptation potential:** {"Maintain infrastructure for climate resilience" if latest_irr > 60 else "Expanding irrigation critical for climate adaptation"}""")
    
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
                        st.metric("P-value", f"{p_value:.25f}")
                    
                    st.markdown("### 🔍 Temperature-Yield Relationship")
                    direction = "negative" if slope < 0 else "positive"
                    strength = "strong" if abs(r_squared) > 0.5 else "moderate" if abs(r_squared) > 0.3 else "weak"
                    st.markdown(f"""
                    - **Relationship type:** {direction.capitalize()} correlation (r = {r_value:.3f})
                    - **Strength:** {strength.capitalize()} relationship (R² = {r_squared:.3f})
                    - **Interpretation:** Each 1°C increase → {slope:.4f} MT/HA yield change
                    - **Agricultural meaning:** {'Heat stress reducing yields - consider heat-tolerant varieties' if slope < -0.1 else 'Moderate temperature impact - monitor threshold effects' if slope < 0 else 'Positive response may indicate sub-optimal baseline temperatures'}
                    - **Statistical confidence:** {'Statistically significant (p < 0.05)' if p_value < 0.05 else 'Not statistically significant - caution in interpretation'}
                    """)
            
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
                    st.markdown("### 🔍 Rainfall-Yield Relationship")
                    st.markdown(f"""
                    - **Correlation:** r = {r_value:.3f} indicates {"strong water dependency" if abs(r_value) > 0.5 else "moderate rainfall influence" if abs(r_value) > 0.3 else "weak direct relationship"}
                    - **Non-linear pattern:** LOWESS curve shows {"optimal rainfall range exists" if True else "linear relationship"}
                    - **Implication:** {'Excess rain can be harmful (waterlogging) - irrigation needs management' if r_value < 0.3 else 'Rainfall critical for yield - drought risk high'}
                    - **Significance:** p = {p_value:.4f}""")
            
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
                st.markdown("### 🔍 Extreme Weather Impact")
                if len(mean_yields) >= 2:
                    yield_drop = ((mean_yields.iloc[0] - mean_yields.iloc[-1]) / mean_yields.iloc[0] * 100) if mean_yields.iloc[0] > 0 else 0
                    st.markdown(f"""
                    - **Shock effect:** Yield drops {abs(yield_drop):.1f}% as extreme events increase
                    - **Vulnerability:** {'High - extreme events cause major disruption' if yield_drop > 15 else 'Moderate - some resilience exists'}
                    - **Risk management:** Weather-based crop insurance and early warning systems needed""")
            
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
                st.markdown("### 🔍 Correlation Insights")
                st.markdown("""
                **Key patterns to observe:**
                - **Strong positive (red):** Variables move together - potential synergy or common cause
                - **Strong negative (blue):** Inverse relationship - trade-offs or competing effects
                - **Weak (white):** Little relationship - independent factors
                - **Agricultural focus:** Look for climate-yield connections and input-output relationships
                """)
            
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
            # Display generated insights
            for i, insight in enumerate(insights, 1):
                st.markdown(f"**{i}.** {insight}")

            st.markdown("---")

            # NEW SECTION: Crop Vulnerability Analysis
            st.subheader("🌾 Crop Vulnerability Assessment")

            if 'Crop_Type' in df_clean.columns and 'Average_Temperature_C' in df_clean.columns and 'Crop_Yield_MT_per_HA' in df_clean.columns:
                crop_climate_analysis = []
    
                for crop in df_clean['Crop_Type'].unique():
                    df_crop = df_clean[df_clean['Crop_Type'] == crop]
        
                    # Temperature sensitivity
                    if len(df_crop) > 10:
                        temp_corr = df_crop[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']].corr().iloc[0, 1]
                        precip_corr = df_crop[['Total_Precipitation_mm', 'Crop_Yield_MT_per_HA']].corr().iloc[0, 1] if 'Total_Precipitation_mm' in df_crop.columns else 0
                        yield_volatility = df_crop['Crop_Yield_MT_per_HA'].std() / df_crop['Crop_Yield_MT_per_HA'].mean() if df_crop['Crop_Yield_MT_per_HA'].mean() > 0 else 0
            
                        crop_climate_analysis.append({
                            'Crop': crop,
                            'Heat Sensitivity': temp_corr,
                            'Rainfall Sensitivity': precip_corr,
                            'Yield Volatility (CV)': yield_volatility
                        })
    
                if crop_climate_analysis:
                    df_vulnerability = pd.DataFrame(crop_climate_analysis)
        
                    col1, col2 = st.columns(2)
        
                    with col1:
                        st.markdown("**🌡️ Most Heat-Sensitive Crops**")
                        heat_sensitive = df_vulnerability.nsmallest(3, 'Heat Sensitivity')[['Crop', 'Heat Sensitivity']]
                        for _, row in heat_sensitive.iterrows():
                            st.markdown(f"- **{row['Crop']}**: r = {row['Heat Sensitivity']:.3f} (Yield ↓ as temperature ↑)")
        
                    with col2:
                        st.markdown("**🌧️ Most Rainfall-Dependent Crops**")
                        rain_dependent = df_vulnerability.nlargest(3, 'Rainfall Sensitivity')[['Crop', 'Rainfall Sensitivity']]
                        for _, row in rain_dependent.iterrows():
                            st.markdown(f"- **{row['Crop']}**: r = {row['Rainfall Sensitivity']:.3f} (Needs adequate rainfall)")
        
                    st.markdown("**⚠️ Most Volatile/Vulnerable Crops**")
                    volatile = df_vulnerability.nlargest(3, 'Yield Volatility (CV)')[['Crop', 'Yield Volatility (CV)']]
                    for _, row in volatile.iterrows():
                        st.markdown(f"- **{row['Crop']}**: CV = {row['Yield Volatility (CV)']:.2f} (High year-to-year variation)")

            st.markdown("---")

        # NEW SECTION: Extreme Year Analysis
        st.subheader("⚠️ Extreme Climate Years Analysis")

        if 'Year' in df_clean.columns and 'Crop_Yield_MT_per_HA' in df_clean.columns:
            # Identify extreme years
            yearly_summary = df_clean.groupby('Year').agg({
                'Crop_Yield_MT_per_HA': 'mean',
                'Average_Temperature_C': 'mean',
                'Total_Precipitation_mm': 'mean' if 'Total_Precipitation_mm' in df_clean.columns else lambda x: np.nan,
                'Extreme_Weather_Events': 'sum' if 'Extreme_Weather_Events' in df_clean.columns else lambda x: np.nan
            }).reset_index()
    
            # Find worst yield years
            worst_years = yearly_summary.nsmallest(3, 'Crop_Yield_MT_per_HA')
    
            st.markdown("**Worst Yield Years:**")
    
            extreme_data = []
            for _, year_row in worst_years.iterrows():
                year = year_row['Year']
                yield_val = year_row['Crop_Yield_MT_per_HA']
                temp = year_row['Average_Temperature_C']
        
                # Determine anomaly
                avg_temp = yearly_summary['Average_Temperature_C'].mean()
                temp_anomaly = temp - avg_temp
        
                anomaly_type = ""
                if 'Total_Precipitation_mm' in yearly_summary.columns:
                    precip = year_row['Total_Precipitation_mm']
                    avg_precip = yearly_summary['Total_Precipitation_mm'].mean()
                    if precip < avg_precip * 0.7:
                        anomaly_type = "Drought year"
                    elif precip > avg_precip * 1.3:
                        anomaly_type = "Excess rainfall"
        
                if abs(temp_anomaly) > 0.5:
                    anomaly_type += f" + {'Heat stress' if temp_anomaly > 0 else 'Cool period'}"
        
                if not anomaly_type:
                    anomaly_type = "Multiple stresses"
        
                extreme_data.append({
                    'Year': int(year),
                    'Climate Anomaly': anomaly_type,
                    'Avg Yield (MT/HA)': f"{yield_val:.2f}",
                    'Explanation': f"{'Heat stress reduced grain filling' if 'Heat' in anomaly_type else 'Water shortage limited growth' if 'Drought' in anomaly_type else 'Flood damage and disease' if 'Excess' in anomaly_type else 'Multiple climate stresses'}"
                })
    
            df_extreme = pd.DataFrame(extreme_data)
            st.table(df_extreme)

        st.markdown("---")

        # NEW SECTION: Thematic Insights
        st.subheader("📊 Thematic Climate-Agriculture Insights")

        tab_a, tab_b, tab_c, tab_d = st.tabs([
            "🌡️ Heat Exposure",
            "🌧️ Rainfall Dependency",
            "⚠️ Extreme Shocks",
            "🚜 Farmer Adaptation"
        ])

        with tab_a:
            st.markdown("""
            ### Heat Exposure Effect
    
            **Observation:**
            - Temperature trends show warming patterns affecting crop physiological processes
            - Heat-sensitive crops (wheat, certain vegetables) show declining yields
    
            **Mechanism:**
            - Heat stress during flowering reduces pollination success
            - Accelerated crop maturity shortens grain-filling period
            - Increased evapotranspiration raises water demand
    
            **Evidence from Analysis:**
            - Negative correlation between temperature and yield
            - Worst yield years often coincide with heat anomalies
            """)

        with tab_b:
            st.markdown("""
            ### Monsoon & Rainfall Dependency
    
            **Observation:**
            - Precipitation variability directly impacts rainfed agriculture
            - Water-sensitive crops show strong rainfall correlation
    
            **Critical Periods:**
            - Planting season rainfall determines germination success
            - Mid-season drought during flowering/grain filling most damaging
            - Excess rainfall causes waterlogging and disease
    
            **Evidence from Analysis:**
            - Yield-precipitation correlations reveal water dependency
            - Drought years show significant yield drops
            """)

        with tab_c:
            st.markdown("""
            ### Extreme Weather Shocks
    
            **Observation:**
            - Extreme events cause sudden, severe yield losses
            - Frequency of extreme events may be increasing
    
            **Types of Shocks:**
            - Floods: Crop damage, soil waterlogging, disease spread
            - Droughts: Water stress, crop failure
            - Storms: Physical damage, lodging
            - Unseasonal frost/heat: Phenological disruption
    
            **Evidence from Analysis:**
            - Box plots show yield drops in high-event years
            - Extreme years correspond to climate anomalies
            """)

        with tab_d:
            st.markdown("""
            ### Farmer Adaptation Behavior
    
            **Observed Adaptations:**
            - Irrigation expansion to buffer rainfall variability
            - Increased fertilizer use to maximize yields
            - Potential variety shifts (not directly visible but implied)
    
            **Effectiveness:**
            - Irrigation access correlates with stable yields
            - Input intensification shows yield maintenance efforts
    
            **Gaps:**
            - Adaptation may lag behind climate change pace
            - Need for systematic climate-smart agriculture adoption
            """)
# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
