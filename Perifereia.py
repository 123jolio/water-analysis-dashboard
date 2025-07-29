import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import numpy as np
import os
import glob
from functools import reduce
from scipy.cluster import hierarchy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

# For mutual information analysis
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

import pingouin as pg
from scipy import signal # NEW: for spectral analysis

# For advanced curve fitting
from scipy.optimize import curve_fit

# For signal processing (if not already imported)
# from scipy.signal import find_peaks # This was already imported above 'from scipy import signal'

# For LOWESS smoothing (optional but recommended)
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    print("statsmodels not available - LOWESS smoothing will be disabled")

# For Granger causality (optional but recommended)
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    print("statsmodels not available - Granger causality tests will be disabled")

# For ITS Analysis
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from datetime import datetime, date
import scipy.stats as stats

import warnings
import datetime # NEW: for st.slider default dates
import networkx as nx # NEW: Import for network graph

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output, use cautiously in production

# --- Helper Functions ---

def add_plot_download_button(df, file_name_prefix):
    """
    Adds download buttons for both CSV and Excel for a given plot's DataFrame.
    """
    if df is None or df.empty:
        return

    # Prepare data for download
    csv_data = df.to_csv(index=False).encode('utf-8')

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='plot_data')
    excel_data = output.getvalue()

    # Create columns for download buttons
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label=f"📥 Download Data as CSV",
            data=csv_data,
            file_name=f"{file_name_prefix}_data.csv",
            mime="text/csv",
            key=f"csv_{file_name_prefix}"
        )

    with col2:
        st.download_button(
            label=f"📥 Download Data as Excel",
            data=excel_data,
            file_name=f"{file_name_prefix}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"excel_{file_name_prefix}"
        )

def create_download_button(df, selected_columns, sheet_name, file_name_prefix):
    """
    Generates a Streamlit download button for the selected data in an Excel format.
    """
    if df is None or df.empty or not selected_columns:
        return  # Don't show button if there's no data or no selection

    # Ensure 'Date' column is always included if it exists in the dataframe
    columns_to_export = selected_columns[:] # Make a copy
    if 'Date' in df.columns and 'Date' not in columns_to_export:
        columns_to_export.insert(0, 'Date')

    # Filter the dataframe for the selected columns
    export_df = df[columns_to_export]

    # Create an in-memory Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Use a user-friendly sheet name, cleaning up any invalid characters
        safe_sheet_name = re.sub(r'[\\/*?:\[\]]', '_', sheet_name)
        export_df.to_excel(writer, index=False, sheet_name=safe_sheet_name[:31]) # Sheet names have a 31-char limit
    excel_data = output.getvalue()

    # Create the download button with a unique key
    st.download_button(
        label=f"📥 Download Selected {file_name_prefix.replace('_', ' ').title()} Data",
        data=excel_data,
        file_name=f"{file_name_prefix}_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{file_name_prefix}"
    )


def create_properly_merged_dataframe(available_data):
    """
    Create a properly merged dataframe ensuring Date column is handled correctly
    and aggregating duplicate dates by taking the mean for numeric columns.
    """
    dataframes_to_merge = []
    
    for source, df_to_merge in available_data.items():
        if df_to_merge is not None and not df_to_merge.empty:
            # Create a copy to avoid modifying original data
            df_copy = df_to_merge.copy()
            
            # Ensure Date column exists and is properly named
            if 'Date' not in df_copy.columns:
                date_cols = [col for col in df_copy.columns if 'date' in col.lower() or 'ημερομηνία' in col.lower()]
                if date_cols:
                    df_copy = df_copy.rename(columns={date_cols[0]: 'Date'})
                else:
                    st.warning(f"No date column found in {source} data. Skipping this source for merging.")
                    continue
            
            # Convert Date to datetime - this is crucial!
            df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
            df_copy = df_copy.dropna(subset=['Date'])
            
            if df_copy.empty:
                st.warning(f"No valid dates found in {source} data after datetime conversion. Skipping this source.")
                continue
            
            # --- NEW FIX: AGGREGATE DUPLICATE DATES BY MEAN ---
            # Identify numeric columns for aggregation
            numeric_cols_for_agg = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            if 'Date' in numeric_cols_for_agg:
                numeric_cols_for_agg.remove('Date') # Ensure 'Date' is not aggregated numerically

            if not df_copy.empty:
                # Separate numeric and non-numeric columns for aggregation strategy
                non_numeric_cols = [col for col in df_copy.columns if col not in numeric_cols_for_agg and col != 'Date']

                aggregated_parts = []
                if numeric_cols_for_agg:
                    aggregated_numeric = df_copy.groupby('Date')[numeric_cols_for_agg].mean()
                    aggregated_parts.append(aggregated_numeric)

                if non_numeric_cols:
                    # For non-numeric, just take the first value for each date
                    # For object dtypes (e.g., strings), .first() is usually safe.
                    # For other non-numeric types, ensure a sensible aggregation is chosen.
                    aggregated_non_numeric = df_copy.groupby('Date')[non_numeric_cols].first()
                    aggregated_parts.append(aggregated_non_numeric)
                
                if aggregated_parts:
                    df_copy_aggregated = pd.concat(aggregated_parts, axis=1).reset_index()
                else:
                    st.warning(f"No usable numeric or non-numeric data columns found in {source} for aggregation after date processing.")
                    continue # Skip this source if no data to process after all

                if df_copy_aggregated.empty:
                    st.warning(f"No valid data after aggregating by date for {source}. Skipping this source.")
                    continue
                df_copy = df_copy_aggregated # Use the aggregated DataFrame

            # --- END NEW FIX ---

            # Sort by date to ensure proper ordering (redundant if grouped, but harmless)
            df_copy = df_copy.sort_values('Date').reset_index(drop=True)
            
            # Add source suffix to parameter columns (not Date!)
            rename_dict = {}
            for col in df_copy.columns:
                if col != 'Date':
                    new_name = f"{col}_{source}"
                    rename_dict[col] = new_name
            
            df_copy = df_copy.rename(columns=rename_dict)
            dataframes_to_merge.append(df_copy)
            
            st.info(f"✓ {source}: {len(df_copy)} records, Date range: {df_copy['Date'].min().strftime('%Y-%m-%d')} to {df_copy['Date'].max().strftime('%Y-%m-%d')}")
    
    # Merge all dataframes
    if not dataframes_to_merge:
        return None
    
    if len(dataframes_to_merge) == 1:
        merged_data = dataframes_to_merge[0]
    else:
        # Use reduce for merging multiple dataframes on Date
        merged_data = reduce(
            lambda left, right: pd.merge(left, right, on='Date', how='outer'), 
            dataframes_to_merge
        )
    
    # Final sort by date and reset index
    merged_data = merged_data.sort_values(by='Date').reset_index(drop=True)
    
    # Verify Date column is still datetime
    if not pd.api.types.is_datetime64_any_dtype(merged_data['Date']):
        merged_data['Date'] = pd.to_datetime(merged_data['Date'], errors='coerce')
    
    # Ensure all Dates are just dates (no time component) for consistent daily frequency.
    merged_data['Date'] = merged_data['Date'].dt.normalize() # Sets time to 00:00:00
    
    return merged_data

def plot_time_series_fixed(data, param, source_info=""):
    """
    Fixed time series plotting function with proper date handling and 7-day MA.
    """
    if data is None or data.empty:
        return None
    
    if param not in data.columns or 'Date' not in data.columns:
        st.error(f"Required columns not found. Available: {list(data.columns)}")
        return None
    
    # Clean the data
    plot_data = data[['Date', param]].copy()
    plot_data = plot_data.dropna(subset=[param])
    
    if plot_data.empty:
        st.warning(f"No valid data found for {param}")
        return None
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(plot_data['Date']):
        plot_data['Date'] = pd.to_datetime(plot_data['Date'], errors='coerce')
        plot_data = plot_data.dropna(subset=['Date'])
    
    if plot_data.empty:
        st.warning(f"No valid dates found for {param}")
        return None
    
    # Sort by date
    plot_data = plot_data.sort_values('Date')
    
    # Create the plot
    fig = go.Figure()
    
    # Main data line
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data[param],
        mode='lines+markers',
        name=param,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
    ))
    
    # Add 7-day Moving Average
    if len(plot_data) > 7:
        ma = plot_data[param].rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=ma,
            mode='lines',
            name='7-day MA',
            line=dict(color='red', dash='dash', width=2)
        ))

    # Update layout with proper date formatting
    fig.update_layout(
        title=f'Time Series: {param} {source_info}',
        xaxis_title='Date',
        yaxis_title=param,
        template="plotly_white",
        hovermode='x unified',
        title_x=0.5,
        height=400,
        xaxis=dict(
            type='date',  # Explicitly set as date axis
            tickformat='%Y-%m-%d'
        )
    )
    
    return fig

def plot_dual_axis_time_series_fixed(data, param1, param2):
    """
    Fixed dual-axis time series with proper date handling
    """
    if data is None or data.empty:
        return None
    
    if param1 not in data.columns or param2 not in data.columns or 'Date' not in data.columns:
        return None
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])
    
    fig = go.Figure()
    
    # Add first parameter
    valid_data1 = data[['Date', param1]].dropna()
    if not valid_data1.empty:
        valid_data1 = valid_data1.sort_values('Date')
        fig.add_trace(go.Scatter(
            x=valid_data1['Date'],
            y=valid_data1[param1],
            name=param1.replace('_', ' '),
            line=dict(color='blue'),
            yaxis='y1',
            hovertemplate='Date: %{x}<br>' + param1 + ': %{y:.3f}<extra></extra>'
        ))
    
    # Add second parameter on secondary y-axis
    valid_data2 = data[['Date', param2]].dropna()
    if not valid_data2.empty:
        valid_data2 = valid_data2.sort_values('Date')
        fig.add_trace(go.Scatter(
            x=valid_data2['Date'],
            y=valid_data2[param2],
            name=param2.replace('_', ' '),
            line=dict(color='red'),
            yaxis='y2',
            hovertemplate='Date: %{x}<br>' + param2 + ': %{y:.3f}<extra></extra>'
        ))
    
    # Update layout for dual y-axis with proper date handling
    fig.update_layout(
        title=f'Time Series Comparison: {param1.replace("_", " ")} vs {param2.replace("_", " ")}',
        template="plotly_white",
        hovermode='x unified',
        height=500,
        xaxis=dict(
            title='Date',
            type='date',  # Explicitly set as date axis
            tickformat='%Y-%m-%d'
        ),
        yaxis=dict(
            title=dict(text=param1.replace('_', ' '), font=dict(color='blue')),
            tickfont=dict(color='blue'),
            side='left'
        ),
        yaxis2=dict(
            title=dict(text=param2.replace('_', ' '), font=dict(color='red')),
            tickfont=dict(color='red'),
            side='right',
            overlaying='y'
        ),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Water & Climate Analysis Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Parameter Standards & Definitions ---
WATER_QUALITY_STANDARDS = {
    # Water Quality parameters
    'pH': {'min': 6.5, 'max': 9.5, 'unit': '', 'max_change_rate': 0.5},
    'ΘΟΛΟΤΗΤΑ': {'min': 0, 'max': 5.0, 'unit': 'NTU', 'max_change_rate': 2.0},
    'ΑΓΩΓΙΜΟΤΗΤΑ': {'min': 0, 'max': 2500, 'unit': 'μS/cm', 'max_change_rate': 100},
    'ΧΛΩΡΙΟΥΧΑ': {'min': 0, 'max': 250, 'unit': 'mg/L', 'max_change_rate': 50},
    'ΣΚΛΗΡΟΤΗΤΑ': {'min': 0, 'max': 500, 'unit': 'mg/L CaCO3', 'max_change_rate': 50},
    'ΘΕΡΜΟΚΡΑΣΙΑ': {'min': None, 'max': None, 'unit': '°C', 'max_change_rate': 5},
    'TOC': {'min': None, 'max': None, 'unit': '', 'max_change_rate': 2},
    'ΟΣΜΗ': {'min': None, 'max': None, 'unit': '', 'max_change_rate': 1},
    'ΧΡΩΜΑ': {'min': None, 'max': None, 'unit': '', 'max_change_rate': 5},
    'ΝΙΤΡΙΚΑ (ΝΟ3-)': {'min': 0, 'max': 50, 'unit': 'mg/L', 'max_change_rate': 10},
    'ΝΙΤΡΩΔΗ (ΝΟ2-)': {'min': 0, 'max': 0.5, 'unit': 'mg/L', 'max_change_rate': 0.1},
    'ΑΜΜΩΝΙΑΚΑ (ΝΗ4+)': {'min': 0, 'max': 0.5, 'unit': 'mg/L', 'max_change_rate': 0.1},
    'ΣΙΔΗΡΟΣ': {'min': 0, 'max': 0.2, 'unit': 'mg/L', 'max_change_rate': 0.05},
    'ΜΑΓΓΑΝΙΟ': {'min': 0, 'max': 0.05, 'unit': 'mg/L', 'max_change_rate': 0.01},
    'ΑΡΓΙΛΙΟ': {'min': 0, 'max': 0.2, 'unit': 'mg/L', 'max_change_rate': 0.05},
    'ΘΕΙΙΚΑ': {'min': 0, 'max': 250, 'unit': 'mg/L', 'max_change_rate': 50},
    'ΦΩΣΦΟΡΟΣ (P2O5)': {'min': 0, 'max': 5, 'unit': 'mg/L', 'max_change_rate': 1},
    'ΜΑΓΝΗΣΙΟ': {'min': None, 'max': None, 'unit': 'mg/L', 'max_change_rate': 10},
    'ΑΣΒΕΣΤΙΟ': {'min': None, 'max': None, 'unit': 'mg/L', 'max_change_rate': 20},
    'ΚΥΑΝΙΟΥΧΑ': {'min': 0, 'max': 0.05, 'unit': 'mg/L', 'max_change_rate': 0.01},
    'ΦΘΟΡΙΟΥΧΑ': {'min': 0, 'max': 1.5, 'unit': 'mg/L', 'max_change_rate': 0.3},
    # Reservoir Level
    'Στάθμη': {'min': None, 'max': None, 'unit': 'm', 'max_change_rate': 2},
    # Climate data
    'Μέση Θερμοκρασία': {'min': None, 'max': None, 'unit': '°C', 'max_change_rate': 5},
    'Μέγιστη Θερμοκρασία': {'min': None, 'max': None, 'unit': '°C', 'max_change_rate': 5},
    'Ελάχιστη Θερμοκρασία': {'min': None, 'max': None, 'unit': '°C', 'max_change_rate': 5},
    'Βροχόπτωση': {'min': None, 'max': None, 'unit': 'mm', 'max_change_rate': 50},
    'Μέση Ταχύτητα Ανέμου': {'min': None, 'max': None, 'unit': 'km/h', 'max_change_rate': 20},
    'Μέγιστη Ταχύτητα Ανέμου': {'min': None, 'max': None, 'unit': 'km/h', 'max_change_rate': 30},
    # Satellite Data
    'temperature': {'min': None, 'max': None, 'unit': '°C', 'max_change_rate': 5},
    'modelled_elevation': {'min': None, 'max': None, 'unit': 'm', 'max_change_rate': 1},
    'Chl a_S2lwa': {'min': 0, 'max': 100, 'unit': 'mg/m³', 'max_change_rate': 20},
    'Chl a_GEE': {'min': 0, 'max': 100, 'unit': 'mg/m³', 'max_change_rate': 20},
    'color_Se2lwa': {'min': None, 'max': None, 'unit': '', 'max_change_rate': 10},
    'TSM': {'min': 0, 'max': 100, 'unit': 'g/m³', 'max_change_rate': 20},
}

# --- Path Configuration ---
# Update these paths to your local data paths or remove if using file uploaders only
WATER_QUALITY_PATH = r"C:\Users\ilioumbas\OneDrive - ΕΥΑΘ ΑΕ\EYATH_1\7.1_AKTOR_GADOURA\article\data from python\from perifereia"
CLIMATE_DATA_PATH = r"C:\Users\ilioumbas\OneDrive - ΕΥΑΘ ΑΕ\EYATH_1\7.1_AKTOR_GADOURA\article\data from python\group_1\Ρόδος_meteo"
RESERVOIR_LEVEL_PATH = r"C:\Users\ilioumbas\OneDrive - ΕΥΑΘ ΑΕ\EYATH_1\7.1_AKTOR_GADOURA\article\data from python\stathmi\1. Α-ΣΥ_ΣΤΑΘΜΗ ΤΑΜΙΕΥΤΗΡΑ.xlsx"
SATELLITE_DATA_PATH = r"C:\Users\ilioumbas\OneDrive - ΕΥΑΘ ΑΕ\EYATH_1\7.1_AKTOR_GADOURA\article\data from python\group_1\AREA_BGR_2.xlsx"


# --- Session State Initialization ---
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'insights' not in st.session_state:
    st.session_state.insights = []

# Ensure analysis_cache itself is a dictionary first (NEW FIX)
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Now ensure specific keys within analysis_cache are initialized (NEW FIX)
if 'lag_matrix' not in st.session_state.analysis_cache:
    st.session_state.analysis_cache['lag_matrix'] = pd.DataFrame()
if 'corr_at_lag_matrix' not in st.session_state.analysis_cache:
    st.session_state.analysis_cache['corr_at_lag_matrix'] = pd.DataFrame()
if 'lag_matrix_warnings' not in st.session_state.analysis_cache:
    st.session_state.analysis_cache['lag_matrix_warnings'] = []


# --- Original Data Processing Functions ---
@st.cache_data
def process_xlsx_files(files):
    if not files: return None
    all_dfs = []
    for file in files:
        is_path = isinstance(file, str)
        file_name = os.path.basename(file) if is_path else file.name
        year_match = re.search(r'(\d{4})', file_name)
        year = int(year_match.group(1)) if year_match else None
        if year is None:
            st.warning(f"Could not extract year from filename: {file_name}. Skipping.")
            continue
        try:
            engine = 'xlrd' if file_name.lower().endswith('.xls') else 'openpyxl'
            xls_sheets = pd.read_excel(file, engine=engine, sheet_name=None)
        except Exception as e:
            st.error(f"Could not read Excel file {file_name}. Error: {e}")
            continue
        for sheet_name, df in xls_sheets.items():
            month_match = re.search(r'^(\d{1,2})', str(sheet_name))
            month = int(month_match.group(1)) if month_match else None
            if month is None: continue
            df['Month'], df['Year'] = month, year
            all_dfs.append(df)
    if not all_dfs: return None
    combined_df = pd.concat(all_dfs, ignore_index=True).rename(columns={'ΗΜΕΡΟΜΗΝΙΑ': 'Parameter'})
    combined_df['Parameter'] = combined_df['Parameter'].str.strip() # FIX: Added .str for strip()
    parameters_of_interest = list(WATER_QUALITY_STANDARDS.keys())
    cleaned_df = combined_df[combined_df['Parameter'].isin(parameters_of_interest)].copy()
    id_vars = ['Parameter', 'Year', 'Month']
    day_columns = [col for col in cleaned_df.columns if isinstance(col, int) and 1 <= col <= 31]
    day_columns.extend([str(i) for i in range(1, 32) if str(i) in cleaned_df.columns])
    value_vars = list(set(day_columns))
    melted_df = cleaned_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Day', value_name='Value')
    melted_df['Day'] = pd.to_numeric(melted_df['Day'])
    melted_df['Date'] = pd.to_datetime(melted_df['Year'].astype(str) + '-' + melted_df['Month'].astype(str) + '-' + melted_df['Day'].astype(str), errors='coerce')
    melted_df.dropna(subset=['Date'], inplace=True)
    melted_df['Value'] = pd.to_numeric(melted_df['Value'].astype(str).replace(',', '.'), errors='coerce')
    melted_df.dropna(subset=['Value'], inplace=True)
    final_df = melted_df.pivot_table(index='Date', columns='Parameter', values='Value').reset_index()
    return final_df.sort_values(by='Date')

@st.cache_data
def process_txt_files(files):
    if not files: return None
    all_data = []
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    for file in files:
        is_path = isinstance(file, str)
        file_name = os.path.basename(file) if is_path else file.name
        try:
            content = open(file, 'r', encoding='utf-8', errors='ignore').read() if is_path else file.getvalue().decode('utf-8', errors='ignore')
        except Exception as e:
            st.warning(f"Could not read file {file_name}. Error: {e}")
            continue
        lines = content.splitlines()
        header_match = re.search(r'SUMMARY for (\w{3})\. (\d{4})', content)
        if not header_match: continue
        month_abbr, year_str = header_match.groups()
        month = month_map.get(month_abbr.upper())
        year = int(year_str)
        if not month: continue
        data_start_index = next((i for i, line in enumerate(lines) if '----' in line), -1)
        if data_start_index == -1: continue
        for line in lines[data_start_index + 1:]:
            if '---' in line or len(line.strip()) < 10: continue
            parts = line.split()
            if parts and parts[0].isdigit():
                try:
                    day = int(parts[0])
                    if 1 <= day <= 31 and len(parts) >= 13:
                        all_data.append({'Date': pd.to_datetime(f"{year}-{month}-{day}", errors='coerce'), 'Μέση Θερμοκρασία': float(parts[1]), 'Μέγιστη Θερμοκρασία': float(parts[2]), 'Ελάχιστη Θερμοκρασία': float(parts[4]), 'Βροχόπτωση': float(parts[8]), 'Μέση Ταχύτητα Ανέμου': float(parts[9]), 'Μέγιστη Ταχύτητα Ανέμου': float(parts[10]), 'Επικρατούσα Διεύθυνση Ανέμου': parts[12]})
                except (ValueError, IndexError): continue
    if not all_data: return None
    return pd.DataFrame(all_data).dropna(subset=['Date']).sort_values(by='Date')

@st.cache_data
def process_level_file(file):
    if not file: return None
    try:
        is_path = isinstance(file, str)
        file_name = os.path.basename(file) if is_path else file.name
        engine = 'xlrd' if file_name.lower().endswith('.xls') else 'openpyxl'
        df = pd.read_excel(file, engine=engine)
        date_col, level_col = None, None
        for col in df.columns:
            col_str = str(col).lower()
            if 'date' in col_str or 'ημερομηνία' in col_str: date_col = col
            if 'στάθμη' in col_str or 'level' in col_str: level_col = col
        if date_col and level_col:
            df = df[[date_col, level_col]].copy()
            df.columns = ['Date', 'Στάθμη']
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Στάθμη'] = pd.to_numeric(df['Στάθμη'], errors='coerce')
            return df.dropna().sort_values(by='Date')
        else:
            st.warning(f"Could not find 'Date' and 'Level' (Στάθμη) columns in {file_name}.")
            return None
    except Exception as e:
        st.error(f"Failed to process reservoir level file {file_name}. Error: {e}")
        return None

@st.cache_data
def process_satellite_file(file):
    if not file: return None
    try:
        is_path = isinstance(file, str)
        file_name = os.path.basename(file) if is_path else file.name
        engine = 'xlrd' if file_name.lower().endswith('.xls') else 'openpyxl'
        raw_df = pd.read_excel(file, engine=engine, header=None)

        header_row_index = -1
        for i, row in raw_df.iterrows():
            if any('date' in str(cell).lower() for cell in row.values):
                header_row_index = i
                break
        
        if header_row_index == -1:
            st.error(f"Could not find a header row with 'date' in satellite file: {file_name}")
            return None

        header = raw_df.iloc[header_row_index]
        data = raw_df.iloc[header_row_index + 1:]
        
        all_param_dfs = []
        
        # Find all columns that are dates and their indices
        date_columns = {i: col for i, col in enumerate(header) if pd.notna(col) and 'date' in str(col).lower()}
        date_indices = sorted(date_columns.keys())

        if not date_indices:
            st.error(f"No columns with 'date' found in the header of satellite file: {file_name}")
            return None

        # Iterate through date columns to find associated parameter columns
        for i, date_idx in enumerate(date_indices):
            # Determine the range of columns for this date's parameters
            start_idx = date_idx + 1
            end_idx = date_indices[i+1] if i + 1 < len(date_indices) else len(header)
            
            # Extract parameters for the current date column
            for param_idx in range(start_idx, end_idx):
                param_name = header.iloc[param_idx]
                if pd.notna(param_name) and str(param_name).strip():
                    # We have a valid parameter
                    param_df = data[[date_idx, param_idx]].copy()
                    param_df.columns = ['Date', param_name]
                    param_df.dropna(how='all', inplace=True)
                    param_df['Date'] = pd.to_datetime(param_df['Date'], errors='coerce')
                    param_df[param_name] = pd.to_numeric(param_df[param_name], errors='coerce')
                    param_df.dropna(subset=['Date', param_name], inplace=True)
                    
                    if not param_df.empty:
                        all_param_dfs.append(param_df.set_index('Date'))

        if not all_param_dfs:
            st.warning(f"No valid satellite data could be extracted from {file_name}.")
            return None
            
        # Merge all the individual parameter dataframes
        final_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), all_param_dfs)
        
        # Handle duplicate columns by averaging them
        final_df = final_df.groupby(final_df.columns, axis=1).mean()

        return final_df.reset_index().sort_values(by='Date')

    except Exception as e:
        st.error(f"An unexpected error occurred while processing satellite file {file_name}: {e}")
        return None

# --- Advanced Analytics Functions ---

# MODIFIED: calculate_lagged_correlations with robust column selection and squeezing
def calculate_lagged_correlations(df_source, param1, param2, max_lag=30, freq='D', interpolation_method='linear'):
    """
    Calculate correlations with time lags after resampling and interpolating.

    Args:
        df_source (pd.DataFrame): DataFrame containing 'Date', param1, and param2 columns.
                                  Assumes 'Date' is already datetime and normalized (time to 00:00:00).
        param1 (str): Name of the first parameter column.
        param2 (str): Name of the second parameter column.
        max_lag (int): Maximum lag to consider in days (positive and negative).
        freq (str): Resampling frequency (e.g., 'D' for daily, 'H' for hourly).
        interpolation_method (str): Method for interpolation (e.g., 'linear', 'ffill', 'bfill').

    Returns:
        pd.DataFrame: DataFrame with 'lag' and 'correlation' columns, or empty DataFrame if not enough data.
        list: List of warning messages generated during calculation for this pair.
    """
    correlations = []
    warnings_list = [] # List to collect warnings for this specific pair calculation

    # Ensure required columns exist and handle potential missingness for these specific columns
    if param1 not in df_source.columns or param2 not in df_source.columns:
        warnings_list.append(f"Parameters '{param1}' or '{param2}' not found in the provided DataFrame for lagged correlation.")
        return pd.DataFrame(), warnings_list

    df_temp = df_source[['Date', param1, param2]].copy()
    
    # --- FIX 1 (Part of Solution 1): Ensure single column selection if unexpected DataFrame is returned ---
    # This addresses the case where df_temp[param1] or df_temp[param2] might
    # unexpectedly return a DataFrame (e.g., due to duplicate column names in underlying data)
    # Check if the selection returns a DataFrame with multiple columns, and if so, take the first one.
    if isinstance(df_temp[param1], pd.DataFrame):
        df_temp[param1] = df_temp[param1].iloc[:, 0]
    if isinstance(df_temp[param2], pd.DataFrame):
        df_temp[param2] = df_temp[param2].iloc[:, 0]
    
    # Ensure columns are truly Series after potential iloc[:,0]
    # .squeeze() will convert a single-column DataFrame to a Series, if it somehow remained a DataFrame.
    df_temp[param1] = df_temp[param1].squeeze()
    df_temp[param2] = df_temp[param2].squeeze()
    # --- END FIX 1 ---

    df_temp = df_temp.set_index('Date')

    # Apply resampling based on the 'freq' parameter passed to this function
    df_resampled = df_temp.asfreq(freq).interpolate(method=interpolation_method)
    df_resampled = df_resampled.dropna(subset=[param1, param2]) # Drop NaNs for both params after resampling/interpolation

    if df_resampled.empty or df_resampled.shape[0] < 2: # Need at least 2 data points for meaningful correlation
        warnings_list.append(f"Not enough common, interpolated data points ({df_resampled.shape[0]} valid records) for parameters '{param1}' and '{param2}' to calculate cross-correlation after resampling and dropping NaNs using frequency '{freq}'.")
        return pd.DataFrame(), warnings_list

    for lag in range(-max_lag, max_lag + 1):
        series1 = df_resampled[param1]
        series2 = df_resampled[param2] # Get the original series (not shifted yet for direct reference)
        series2_shifted = series2.shift(lag) # Now shift it

        # --- NEW FIX for "If using all scalar values, you must pass an index" ---
        aligned_data = pd.DataFrame() # Initialize as empty
        try:
            # Only attempt to create DataFrame if both series are not empty AND have any non-NaN values
            if not series1.empty and not series2_shifted.empty and series1.notna().any() and series2_shifted.notna().any():
                temp_df_for_alignment = pd.DataFrame({
                    'param1_val': series1, # These are already Series
                    'param2_shifted_val': series2_shifted # These are already Series
                }, index=df_resampled.index) # Explicitly pass the index here to avoid scalar value error
                
                aligned_data = temp_df_for_alignment.dropna() # Drop rows where either value is NaN after alignment
            else:
                # If series are empty or all NaN, then aligned_data remains empty
                aligned_data = pd.DataFrame() 
            
        except ValueError as ve: # Catch potential ValueError during DataFrame creation if series are malformed
            warnings_list.append(f"ValueError during DataFrame creation for '{param1}' vs '{param2}' at lag {lag}: {ve}. Skipping correlation for this lag.")
            aligned_data = pd.DataFrame() 
        except Exception as e: # Catch any other unexpected errors during DataFrame creation
            warnings_list.append(f"Unexpected error during DataFrame creation for '{param1}' vs '{param2}' at lag {lag}: {e}. Skipping correlation for this lag.")
            aligned_data = pd.DataFrame() 
        # --- END NEW FIX ---
        
        if aligned_data.empty or aligned_data.shape[0] < 2: # Check again after all alignments and drops
            corr = np.nan # Cannot compute correlation, set to NaN
        else:
            try:
                # pandas .corr() will generally handle series with constant values (std dev of 0) by returning NaN
                # However, an explicit check for std dev can be added if needed for specific cases
                if aligned_data['param1_val'].std() == 0 or aligned_data['param2_shifted_val'].std() == 0:
                    corr = np.nan 
                    warnings_list.append(f"Correlation for '{param1}' vs '{param2}' at lag {lag} is undefined due to one or both series having a constant value (zero standard deviation).")
                else:
                    corr = aligned_data['param1_val'].corr(aligned_data['param2_shifted_val'])
            except Exception as e:
                corr = np.nan # Catch any remaining unusual errors from .corr()
                warnings_list.append(f"Error calculating .corr() for '{param1}' vs '{param2}' at lag {lag}: {e}")
        
        if pd.notna(corr):
            correlations.append({'lag': lag, 'correlation': corr})
    
    return pd.DataFrame(correlations), warnings_list # Return the list of warnings


def detect_anomalies(data, contamination=0.05):
    """Detect multivariate anomalies using Isolation Forest"""
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if numeric_data.shape[0] < 10 or numeric_data.shape[1] < 2:
        return pd.DataFrame()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    anomalies = clf.fit_predict(scaled_data)
    
    anomaly_df = data.loc[numeric_data.index].copy()
    anomaly_df['anomaly'] = anomalies
    anomaly_df = anomaly_df[anomaly_df['anomaly'] == -1]
    
    return anomaly_df.drop('anomaly', axis=1)

def analyze_feature_importance(data, target_param, max_features=10):
    """Analyze which parameters most influence a target variable"""
    # Prepare data
    numeric_cols = [col for col in data.columns if col != 'Date' and col != target_param]
    df_clean = data[[target_param] + numeric_cols].dropna()
    
    if df_clean.shape[0] < 30:  # Need sufficient data
        return pd.DataFrame()
    
    X = df_clean[numeric_cols]
    y = df_clean[target_param]
    
    # Train model
    rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(max_features)
    
    return importance_df

def generate_insights(data_dict):
    """Generate automated insights from all available data"""
    insights = []
    
    # Analyze each dataset
    for source_name, df in data_dict.items():
        if df is None or df.empty:
            continue
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Trend analysis
        for col in numeric_cols:
            if col == 'Date' or df[col].isna().all():
                continue
                
            clean_data = df[['Date', col]].dropna()
            if len(clean_data) < 10:
                continue
                
            # Calculate trend
            x = np.arange(len(clean_data))
            y = clean_data[col].values
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                if p_value < 0.05 and abs(r_value) > 0.3:  # Significant trend
                    direction = "increasing" if slope > 0 else "decreasing"
                    insights.append({
                        'type': 'trend',
                        'icon': '📈' if slope > 0 else '📉',
                        'message': f"{col} in {source_name} shows a {direction} trend (R²={r_value**2:.2f})",
                        'severity': 'info'
                    })
            except:
                pass
        
        # Check for recent anomalies
        if len(df) > 30:
            recent_data = df.tail(30)
            for col in numeric_cols:
                if col == 'Date' or recent_data[col].isna().all():
                    continue
                    
                mean = recent_data[col].mean()
                std = recent_data[col].std()
                
                if std > 0:
                    last_value = recent_data[col].iloc[-1]
                    z_score = abs((last_value - mean) / std)
                    
                    if z_score > 3:
                        insights.append({
                            'type': 'anomaly',
                            'icon': '⚠️',
                            'message': f"Unusual value detected for {col} in {source_name}: {last_value:.2f} (Z-score: {z_score:.1f})",
                            'severity': 'warning'
                        })
    
    # Cross-dataset insights
    if len(data_dict) >= 2:
        # Look for correlated parameters across datasets
        all_params = []
        for source, df in data_dict.items():
            if df is not None:
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != 'Date':
                        all_params.append((source, col))
        
        # Parameter groupings
        param_groups = {
            'temperature': ['ΘΕΡΜΟΚΡΑΣΙΑ', 'Μέση Θερμοκρασία', 'temperature'],
            'chlorophyll': ['Chl a_S2lwa', 'Chl a_GEE'],
            'turbidity': ['ΘΟΛΟΤΗΤΑ', 'TSM']
        }
        
        for group_name, group_params in param_groups.items():
            found_params = [(s, p) for s, p in all_params if any(gp in p for gp in group_params)]
            if len(found_params) > 1:
                insights.append({
                    'type': 'correlation',
                    'icon': '🔗',
                    'message': f"Multiple {group_name} measurements available for cross-validation",
                    'severity': 'info'
                })
    
    return insights

def create_alert_system(data_dict):
    """Generate alerts based on thresholds and trends"""
    alerts = []
    
    for source_name, df in data_dict.items():
        if df is None or df.empty:
            continue
            
        for param, limits in WATER_QUALITY_STANDARDS.items():
            if param not in df.columns:
                continue
                
            recent_data = df[param].tail(7).dropna()
            if len(recent_data) == 0:
                continue
            
            # Check absolute thresholds
            if limits.get('max') is not None and (recent_data > limits['max']).any():
                max_value = recent_data.max()
                alerts.append({
                    'level': 'critical',
                    'source': source_name,
                    'parameter': param,
                    'message': f"{param} exceeded maximum threshold ({limits['max']} {limits.get('unit', '')})",
                    'value': f"{max_value:.2f} {limits.get('unit', '')}",
                    'timestamp': df['Date'].iloc[-1]
                })
            
            if limits.get('min') is not None and (recent_data < limits['min']).any():
                min_value = recent_data.min()
                alerts.append({
                    'level': 'warning',
                    'source': source_name,
                    'parameter': param,
                    'message': f"{param} below minimum threshold ({limits['min']} {limits.get('unit', '')})",
                    'value': f"{min_value:.2f} {limits.get('unit', '')}",
                    'timestamp': df['Date'].iloc[-1]
                })
            
            # Check rate of change
            if len(recent_data) > 1 and limits.get('max_change_rate'):
                daily_changes = recent_data.diff().abs()
                max_change = daily_changes.max()
                
                if max_change > limits['max_change_rate']:
                    alerts.append({
                        'level': 'warning',
                        'source': source_name,
                        'parameter': param,
                        'message': f"{param} changing rapidly",
                        'value': f"{max_change:.2f} {limits.get('unit', '')}/day",
                        'timestamp': df['Date'].iloc[-1]
                    })
    
    return sorted(alerts, key=lambda x: (x['level'] == 'critical', x['timestamp']), reverse=True)

@st.cache_data
def calculate_nmi_matrix(df):
    """Calculates the Normalized Mutual Information matrix for a dataframe."""
    df_binned = pd.DataFrame()
    
    # Bin continuous data into discrete categories
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Using 10 bins is a reasonable default
            df_binned[col] = pd.cut(df[col], bins=10, labels=False, duplicates='drop')
        else:
            df_binned[col] = df[col] # Keep categorical columns as is

    nmi_matrix = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    for i, col1 in enumerate(df_binned.columns):
        for j, col2 in enumerate(df_binned.columns):
            if i > j: continue # Matrix is symmetric
            
            # Drop NaNs for the pair before calculating
            valid_data = df_binned[[col1, col2]].dropna()
            if len(valid_data) < 2:
                score = np.nan
            else:
                score = normalized_mutual_info_score(valid_data[col1], valid_data[col2])
            
            nmi_matrix.loc[col1, col2] = score
            nmi_matrix.loc[col2, col1] = score
            
    np.fill_diagonal(nmi_matrix.values, 1.0) # A variable has perfect NMI with itself
    return nmi_matrix

# MODIFIED: calculate_lag_matrix to use the new calculate_lagged_correlations with frequency
@st.cache_data
def calculate_lag_matrix(df, numeric_cols, max_lag, freq='D'): # ADDED freq parameter
    """Calculates a matrix of optimal lags and their correlations using resampled data."""
    lag_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)
    corr_at_lag_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)
    
    all_collected_warnings = [] # List to collect all warnings from sub-calls

    for i, param1 in enumerate(numeric_cols):
        for j, param2 in enumerate(numeric_cols):
            # Pass the freq to calculate_lagged_correlations
            lag_corr_df, sub_warnings = calculate_lagged_correlations(df, param1, param2, max_lag=max_lag, freq=freq) # PASSED freq
            all_collected_warnings.extend(sub_warnings) # Aggregate warnings
            
            if not lag_corr_df.empty:
                # Find the lag with the maximum *absolute* correlation
                optimal_lag_row = lag_corr_df.loc[lag_corr_df['correlation'].abs().idxmax()]
                optimal_lag = optimal_lag_row['lag']
                corr_at_lag = optimal_lag_row['correlation']
                
                lag_matrix.loc[param1, param2] = optimal_lag
                corr_at_lag_matrix.loc[param1, param2] = corr_at_lag

            else:
                lag_matrix.loc[param1, param2] = np.nan
                corr_at_lag_matrix.loc[param1, param2] = np.nan

    np.fill_diagonal(lag_matrix.values, 0)
    np.fill_diagonal(corr_at_lag_matrix.values, 1.0) 
    
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            # Ensure symmetry for corr_at_lag_matrix for display, especially if some cells are NaN
            if pd.isna(corr_at_lag_matrix.iloc[j, i]):
                    corr_at_lag_matrix.iloc[j, i] = corr_at_lag_matrix.iloc[i, j] 
    
    return lag_matrix, corr_at_lag_matrix, all_collected_warnings

def hierarchical_clustering_order(matrix_df):
    """
    Performs hierarchical clustering on a correlation/similarity matrix and returns the optimal ordering.
    This helps group similar parameters together for better visualization.
    """
    # Handle any NaN values by filling with 0
    matrix_filled = matrix_df.fillna(0)
    
    # Convert correlation to distance (1 - abs(correlation))
    # We use absolute value because both positive and negative correlations indicate relationships
    distance_matrix = 1 - np.abs(matrix_filled.values)
    
    # Ensure the distance matrix is symmetric and has 0 on diagonal
    # This is critical for scipy.cluster.hierarchy functions
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Check if the matrix is valid for clustering (at least two non-NaN values)
    if distance_matrix.shape[0] < 2 or np.all(np.isnan(distance_matrix)) or np.all(distance_matrix == 0):
        # If all distances are 0 (e.g., all correlations are 1 or -1), clustering might fail.
        # Return original order or a sorted list of indices.
        return list(range(matrix_df.shape[0])) # Return original order if not enough data or trivial distances
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
    
    # Get the optimal ordering
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
    optimal_order = dendro['leaves']
    
    return optimal_order

def plot_parameter_advanced(data, param, show_anomalies=False):
    """Enhanced parameter plotting with anomaly detection"""
    fig = go.Figure()
    
    # Main data line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[param],
        mode='lines+markers',
        name=param,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Add moving average
    if len(data) > 7:
        ma = data[param].rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=ma,
            mode='lines',
            name='7-day MA',
            line=dict(color='orange', dash='dash', width=2)
        ))
    
    # Add threshold lines
    limits = WATER_QUALITY_STANDARDS.get(param, {})
    if limits.get('min') is not None:
        fig.add_hline(y=limits['min'], line_dash="dot", line_color="green",
                                     annotation_text=f"Min: {limits['min']}")
    if limits.get('max') is not None:
        fig.add_hline(y=limits['max'], line_dash="dot", line_color="red",
                                     annotation_text=f"Max: {limits['max']}")
    
    # Add anomalies if requested
    if show_anomalies and len(data) > 30:
        anomaly_data = detect_anomalies(data[['Date', param]])
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['Date'],
                y=anomaly_data[param],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Time Profile for {param}',
        xaxis_title='Date',
        yaxis_title=f'{param} ({limits.get("unit", "")})',
        template="plotly_white",
        hovermode='x unified',
        title_x=0.5
    )
    
    return fig

def plot_correlation_scatter(data, param1, param2, show_nonlinear_trend=False):
    """Creates a scatter plot to visualize the correlation between two parameters."""
    
    # Drop rows where either parameter is NaN to ensure a clean plot
    # Also keep the 'Date' column for hover information
    df_plot = data[['Date', param1, param2]].dropna()
    
    if df_plot.empty:
        return None # Return None if no overlapping data

    # Use a different trendline based on the user's choice
    if show_nonlinear_trend:
        title = f'Scatter Plot with LOWESS Trendline: {get_param_info(param1)} vs. {get_param_info(param2)}'
        trendline_type = "lowess"
    else:
        title = f'Scatter Plot: {get_param_info(param1)} vs. {get_param_info(param2)}'
        trendline_type = None

    fig = px.scatter(
        df_plot,
        x=param1,
        y=param2,
        title=title,
        template="plotly_white",
        trendline=trendline_type,
        trendline_color_override="red",
        hover_data=['Date'] # Add date to hover
    )
    
    fig.update_layout(title_x=0.5, xaxis_title=get_param_info(param1), yaxis_title=get_param_info(param2))
    return fig

# NEW: Function for Spectral Coherence Analysis
def plot_coherence_and_phase(df_source, param1, param2, freq_sampling='D', nperseg=256):
    """
    Calculates and plots the coherence and phase between two time series.
    Requires data to be uniformly sampled (or resampled and interpolated).
    """
    
    # Select relevant columns, set Date as index, and ensure numeric types
    if param1 not in df_source.columns or param2 not in df_source.columns:
        st.error(f"Error: One or both parameters '{param1}' or '{param2}' not found for spectral analysis.")
        return

    df_plot = df_source[['Date', param1, param2]].copy()
    df_plot['Date'] = pd.to_datetime(df_plot['Date']) # Ensure datetime
    df_plot = df_plot.set_index('Date')
    df_plot[param1] = pd.to_numeric(df_plot[param1], errors='coerce')
    df_plot[param2] = pd.to_numeric(df_plot[param2], errors='coerce')
    
    # Resample to common frequency and interpolate for spectral analysis
    # Use .asfreq() to ensure a continuous range, then interpolate
    df_plot_resampled = df_plot.asfreq(freq_sampling).interpolate(method='linear')
    df_plot_resampled = df_plot_resampled.dropna(subset=[param1, param2]) # Drop any NaNs left after interpolation

    if df_plot_resampled.empty or df_plot_resampled.shape[0] < nperseg:
        st.warning(f"Not enough common, interpolated data points ({df_plot_resampled.shape[0]}) for spectral coherence analysis. Need at least {nperseg} points (nperseg) for meaningful results. Try a larger date range.")
        return

    # Convert frequency string to sampling rate (samples per unit time, e.g., 1 sample per day for 'D')
    fs_val = 1.0 # Default: 1 sample per fundamental time unit (e.g., 1 sample per day if freq_sampling is 'D')
    
    # Correct fs based on resampling unit:
    if freq_sampling == 'D':
        fs_val = 1.0 # 1 sample per day, so freq is cycles/day
    elif freq_sampling == 'H':
        fs_val = 1.0 # 1 sample per hour, so freq is cycles/hour. Need to convert for display.
    elif freq_sampling == 'W':
        fs_val = 1.0 # 1 sample per week, so freq is cycles/week.
    elif freq_sampling == 'M':
        fs_val = 1.0 # 1 sample per month, so freq is cycles/month.

    # Calculate Coherence and Phase Angle
    try:
        f, Cxy = signal.coherence(df_plot_resampled[param1], df_plot_resampled[param2], fs=fs_val, nperseg=nperseg)
        f_csd, Pxy = signal.csd(df_plot_resampled[param1], df_plot_resampled[param2], fs=fs_val, nperseg=nperseg)
        phase = np.angle(Pxy, deg=True) # Phase angle in degrees
    except Exception as e:
        st.error(f"Error calculating spectral coherence: {e}. This might occur if data is too sparse or too short after resampling for the chosen nperseg (e.g., nperseg > number of valid data points).")
        return

    # Adjust frequency labels for clarity based on selected sampling frequency
    xaxis_title_freq = f'Frequency (cycles/{freq_sampling.lower()})'
    if freq_sampling == 'D':
        xaxis_title_freq = 'Frequency (cycles/day)'
    elif freq_sampling == 'H':
        xaxis_title_freq = 'Frequency (cycles/hour)'  
        # If you still want cycles/day here, you'd multiply f by 24 and adjust x_range accordingly
    
    # Filter out frequencies of 0 (DC component) or very high frequencies if desired
    valid_indices = (f > 0) # Exclude DC component (frequency 0)

    # Plotting Coherence
    fig_coherence = go.Figure()
    fig_coherence.add_trace(go.Scatter(x=f[valid_indices], y=Cxy[valid_indices], mode='lines', name='Coherence'))
    fig_coherence.update_layout(
        title=f'Coherence between {get_param_info(param1)} and {get_param_info(param2)}',
        xaxis_title=xaxis_title_freq,
        yaxis_title='Coherence',
        title_x=0.5,
        hovermode='x unified',
        xaxis_range=[0, fs_val/2] # Max frequency is Nyquist frequency (fs_val / 2)
    )
    st.plotly_chart(fig_coherence, use_container_width=True)

    # Plotting Phase Angle
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=f_csd[valid_indices], y=phase[valid_indices], mode='lines', name='Phase Angle'))
    fig_phase.update_layout(
        title=f'Phase Angle between {get_param_info(param1)} and {get_param_info(param2)}',
        xaxis_title=xaxis_title_freq, # Keep consistent frequency unit
        yaxis_title='Phase Angle (Degrees)',
        title_x=0.5,
        hovermode='x unified',
        xaxis_range=[0, fs_val/2],
        yaxis_range=[-180, 180] # Phase angle typically -180 to 180 degrees
    )
    st.plotly_chart(fig_phase, use_container_width=True)

    st.markdown("---")
    st.write("### Interpreting Coherence and Phase:")
    st.write(" - **Coherence:** A value closer to 1 indicates a strong linear relationship at that specific frequency (cycle). Values near 0 indicate no linear relationship at that frequency.")
    st.write(" - **Phase Angle:** If coherence is high at a particular frequency, the phase angle tells you the lag *for that cycle*.")
    st.write("    - **Positive phase angle:** The second parameter (`param2`) **lags** the first parameter (`param1`). (i.g., `param1` leads `param2`)")
    st.write("    - **Negative phase angle:** The second parameter (`param2`) **leads** the first parameter (`param1`). (i.g., `param2` leads `param1`)")
    st.write(f"    - **To convert phase angle (degrees) to lag in units of {freq_sampling.lower()}:** `Lag = (Phase Angle / 360) / Frequency (cycles/{freq_sampling.lower()})`")
    st.write(f"    - **Note on Frequency Units:** The frequency axis above is in cycles per **{freq_sampling.lower()}** based on your 'Data Sampling Frequency' selection.")
    st.write(f"    - To convert period: `Period ({freq_sampling.lower()}s/cycle) = 1 / Frequency (cycles/{freq_sampling.lower()})`")
    if freq_sampling != 'D':
        # Add a note on conversion to 'days' if freq_sampling is not daily.
        # This mapping is for conversion Factor_to_Days = Num_Original_Units_in_a_Day
        # e.g., for 'H', 24 hours in a day, so freq_cycles_per_hour * 24 = freq_cycles_per_day
        # fs_map stores samples/fundamental unit, so fs_map['D'] is 1 sample per day.
        # fs_map['H'] is 1 sample per hour, but we are using fs_val=1.0 regardless, so f_cycles/hour.
        # Let's clarify the interpretation rather than doing complex conversions inside the text.
        st.info(f"    To convert frequency from cycles per **{freq_sampling.lower()}** to cycles per **day**, you need to know how many {freq_sampling.lower()}s are in a day. For example, if 'H' (hourly), then divide frequency by 24 (or multiply period by 24) to get values per day.")


# ADD THESE FUNCTIONS TO YOUR CODE AFTER THE EXISTING FUNCTIONS #
# (After plot_coherence_and_phase function and before the UI Layout section that starts with "# --- UI Layout ---")
# ============= START OF MISSING FUNCTIONS =============
def get_param_info(param_name):
    """
    Helper function to format parameter names from 'param_Source' to 'param (Source)'.
    This function should be defined once, preferably at the top with other helper functions.
    """
    if isinstance(param_name, str) and '_' in param_name:
        parts = param_name.rsplit('_', 1)
        # Check if the last part is a recognized source to avoid splitting "Chl a_S2lwa" incorrectly
        # This list should match the keys in your `data_dict` and `available_data`.
        if parts[-1] in ["Water Quality", "Climate", "Reservoir", "Satellite"]:
            param = parts[0]
            source = parts[1]
            return f"{param} ({source})"
    return param_name


def create_correlation_summary_table(corr_matrix, lag_matrix=None, top_n=20, threshold=0.3):
    """
    Creates a sorted summary table of correlations with optional lag information
        Parameters:
    - corr_matrix: correlation matrix DataFrame
    - lag_matrix: optional lag matrix DataFrame
    - top_n: number of top correlations to show
    - threshold: minimum absolute correlation to include
    """
    # Extract upper triangle to avoid duplicates
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) 
    
    # Create list of correlations
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if mask[i, j]: # Ensure we only consider each unique pair once
                param1 = corr_matrix.columns[i]
                param2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                                        
                if abs(corr_val) >= threshold and not pd.isna(corr_val):
                    correlation_data = {
                        'Parameter 1': get_param_info(param1),
                        'Parameter 2': get_param_info(param2),
                        'Correlation': corr_val,
                        'Abs Correlation': abs(corr_val),
                        'Relationship': 'Positive' if corr_val > 0 else 'Negative'
                    }
                                            
                    # Add lag information if available
                    if lag_matrix is not None and param1 in lag_matrix.index and param2 in lag_matrix.columns:
                        lag_val = lag_matrix.loc[param1, param2]
                        # Dynamically determine the lag unit for display based on context
                        # This table is used by both raw and averaged tabs, so we need a generic unit.
                        # The calling add_correlation_summaries will pass a specific lag_matrix.
                        # It's safer to keep "periods" here, and let the tab's context explain "what a period is".
                        if "Avg" in str(lag_matrix.name) or "Averaged" in str(lag_matrix.name): # Simple check for averaged context
                             lag_unit = "periods"
                        else:
                            lag_unit = "days" # Default for raw data context
                        
                        correlation_data[f'Optimal Lag ({lag_unit})'] = lag_val 
                        if not pd.isna(lag_val):
                            if lag_val > 0:
                                correlation_data['Lead/Lag'] = f"{get_param_info(param1)} leads {get_param_info(param2)} by {abs(lag_val):.0f} {lag_unit}" 
                            elif lag_val < 0:
                                correlation_data['Lead/Lag'] = f"{get_param_info(param2)} leads {get_param_info(param1)} by {abs(lag_val):.0f} {lag_unit}" 
                            else:
                                correlation_data['Lead/Lag'] = "Synchronous"
                        else:
                            correlation_data['Lead/Lag'] = "N/A"
                                            
                    correlations.append(correlation_data)
    
    # Convert to DataFrame and sort
    summary_df = pd.DataFrame(correlations)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Abs Correlation', ascending=False).head(top_n)
        summary_df = summary_df.drop('Abs Correlation', axis=1) # Drop helper column for display
        return summary_df
    return pd.DataFrame() # Return empty if no correlations meet criteria


def create_network_graph(corr_matrix, threshold=0.5, layout='spring'):
    """
    Creates an interactive network graph of correlations
    """
    # Create graph
    G = nx.Graph()
        # Add nodes
    for param in corr_matrix.columns:
        G.add_node(param)
        # Add edges for correlations above threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold and not pd.isna(corr_val):
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                                     weight=abs(corr_val), correlation=corr_val)
        # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai': # Corrected spelling
        pos = nx.kamada_kawai_layout(G)
    else: # Fallback to spring if an invalid layout is chosen
        pos = nx.spring_layout(G, k=3, iterations=50)

    # Ensure positions exist for all nodes, especially if some nodes have no edges above threshold
    if not pos: # If graph is empty or no edges, layout might not generate positions
        return None # Indicate no graph can be created

    edge_x = []
    edge_y = []
    # No direct use of edge_colors, edge_widths, edge_hovertext arrays.
    # Instead, individual traces for positive/negative edges are created below.

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [get_param_info(node) for node in G.nodes()] # Use readable names for nodes
    node_hovertext = [f"Parameter: {get_param_info(node)}<br>Connections: {G.degree(node)}" for node in G.nodes()]

    # Need separate traces for positive and negative edges to color them and enable legend
    positive_edges_traces = []
    negative_edges_traces = []
    
    # Track if legend entries have been added for each type to avoid duplicates
    added_pos_legend = False
    added_neg_legend = False

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        corr = edge[2]['correlation']
        weight = edge[2]['weight']

        hover_text = f"{get_param_info(edge[0])} - {get_param_info(edge[1])}<br>Correlation: {corr:.2f}"

        if corr > 0:
            positive_edges_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=weight*5, color='blue'), # Thicker for stronger positive
                mode='lines',
                name='Positive Correlation',
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=not added_pos_legend # Only show legend once for positive
            ))
            added_pos_legend = True
        else: # corr < 0
            negative_edges_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=weight*5, color='red'), # Thicker for stronger negative
                mode='lines',
                name='Negative Correlation',
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=not added_neg_legend # Only show legend once for negative
            ))
            added_neg_legend = True


    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=20,
            color='lightblue', # Base node color
            line=dict(width=2, color='darkblue')
        ),
        hoverinfo='text',
        hovertext=node_hovertext,
        showlegend=False # Nodes don't need a legend entry
    )
        
    fig = go.Figure(data=positive_edges_traces + negative_edges_traces + [node_trace])
    fig.update_layout(
        title=f"Correlation Network (|r| ≥ {threshold})", # Adjusted title
        showlegend=True, # Show legend for positive/negative edges
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig


def create_correlation_by_category(corr_matrix, categories):
    """
    Groups correlations by parameter categories
        categories = {
        'Water Quality': ['pH', 'ΘΟΛΟΤΗΤΑ', 'ΑΓΩΓΙΜΟΤΗΤΑ', ...],
        'Climate': ['temperature', 'Μέση Θερμοκρασία', ...],
        'Nutrients': ['ΝΙΤΡΙΚΑ', 'ΦΩΣΦΟΡΟΣ', ...],
        ...
    }
    """
    category_correlations = {}
    
    for cat1, params1 in categories.items():
        for cat2, params2 in categories.items():
            if cat1 <= cat2:  # Avoid duplicates (A-B is same as B-A)
                # Get correlations between categories
                # Filter params to only include those actually in the corr_matrix columns
                cat_params1_in_matrix = [p for p in params1 if p in corr_matrix.columns]
                cat_params2_in_matrix = [p for p in params2 if p in corr_matrix.columns]
                                        
                if cat_params1_in_matrix and cat_params2_in_matrix:
                    sub_corr = corr_matrix.loc[cat_params1_in_matrix, cat_params2_in_matrix]
                    
                    # Ensure sub_corr is not empty before calculating mean/max
                    if not sub_corr.empty:
                        # Handle cases where sub_corr might contain only NaNs, which mean().mean() would return NaN
                        avg_corr = sub_corr.abs().mean().mean()
                        max_corr = sub_corr.abs().max().max()

                        # Fallback if all values are NaN
                        if pd.isna(avg_corr):
                            avg_corr = 0
                        if pd.isna(max_corr):
                            max_corr = 0
                            
                        category_correlations[f"{cat1} - {cat2}"] = {
                            'Average |r|': avg_corr,
                            'Max |r|': max_corr,
                            'N Parameters': f"{len(cat_params1_in_matrix)} x {len(cat_params2_in_matrix)}"
                        }
    
    summary_df = pd.DataFrame(category_correlations).T
    if not summary_df.empty:
        return summary_df.sort_values('Average |r|', ascending=False)
    return pd.DataFrame()


def create_correlation_dendrogram(corr_matrix):
    """
    Creates a dendrogram showing hierarchical clustering of parameters
    """
    if corr_matrix.shape[0] < 2:
        return None # Not enough parameters for a dendrogram

    # Calculate distance matrix (handle NaNs by filling with 0, then ensure symmetry)
    distance_matrix = 1 - np.abs(corr_matrix.fillna(0).values)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Check if distance matrix is valid for clustering (not all zeros/identical)
    if np.all(distance_matrix == 0):
        st.info("All parameters are perfectly correlated (or anti-correlated). Dendrogram cannot be generated.")
        return None
    
    # Perform hierarchical clustering
    try:
        linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
    except Exception as e:
        st.warning(f"Could not perform hierarchical clustering for dendrogram: {e}. Data might be too uniform or sparse.")
        return None
        
    # Create dendrogram
    fig = go.Figure()
        
    dendro = hierarchy.dendrogram(linkage_matrix, labels=corr_matrix.columns.map(get_param_info).tolist(), no_plot=True) # Use readable names
        
    # Create the dendrogram traces
    for i, d in enumerate(dendro['dcoord']):
        x = dendro['icoord'][i]
        y = d
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', showlegend=False, 
                                     line=dict(color='black', width=1)))
        # Add labels for the leaves (parameters)
    fig.update_layout(
        title="Parameter Clustering Dendrogram",
        xaxis=dict(tickmode='array', tickvals=[dendro['icoord'][j][1] for j in range(len(dendro['icoord'])) if dendro['dcoord'][j][-1] == 0], # Positions of leaves
                     ticktext=[dendro['ivl'][k] for k in dendro['leaves']], # Labels for leaves
                     tickangle=-45),
        yaxis_title="Distance",
        height=500,
        margin=dict(t=50, b=150) # Adjust margin for labels
    )
    
    return fig


def add_correlation_summaries(corr_matrix, lag_matrix=None, suffix=""): # Added suffix for clarity
    """Add various summary visualizations to the correlation tab"""

    st.markdown("---")
    st.subheader(f"📊 Correlation Analysis Summary {suffix}")
    
    if corr_matrix.empty:
        st.info(f"No correlation matrix available for summary {suffix}. Please calculate the relevant Lag Matrix first.")
        return

    # Summary tabs
    summary_tabs = st.tabs([f"📋 Ranked List {suffix}", f"🕸️ Network View {suffix}", f"📊 By Category {suffix}", f"🌳 Dendrogram {suffix}"])

    with summary_tabs[0]:
        st.write(f"### Strongest Correlations (Ranked) {suffix}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            threshold = st.slider(f"Minimum absolute correlation threshold {suffix}:", 0.0, 1.0, 0.3, 0.05, key=f"corr_threshold{suffix}")
        with col2:
            top_n = st.number_input(f"Show top N {suffix}:", min_value=10, max_value=100, value=20, step=5, key=f"top_n_corr{suffix}")
        
        summary_df = create_correlation_summary_table(corr_matrix, lag_matrix, top_n=top_n, threshold=threshold)
        
        if not summary_df.empty:
            # Color code the correlation values
            def color_correlation(val):
                if isinstance(val, (int, float)):
                    if val > 0.7:
                        return 'background-color: darkgreen; color: white'
                    elif val > 0.5:
                        return 'background-color: lightgreen'
                    elif val < -0.7:
                        return 'background-color: darkred; color: white'
                    elif val < -0.5:
                        return 'background-color: lightcoral'
                return ''
            
            styled_df = summary_df.style.applymap(color_correlation, subset=['Correlation'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download Correlation Summary {suffix}",
                data=csv,
                file_name=f"correlation_summary{suffix}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No correlations found above threshold {threshold} or not enough data {suffix}.")

    with summary_tabs[1]:
        st.write(f"### Correlation Network Graph {suffix}")
        st.info("Blue edges show positive correlations, red edges show negative correlations. Edge thickness represents absolute correlation strength.")
        
        col1, col2 = st.columns(2)
        with col1:
            network_threshold = st.slider(f"Network correlation threshold {suffix}:", 0.3, 0.9, 0.5, 0.05, key=f"network_threshold{suffix}")
        with col2:
            layout_type = st.selectbox(f"Layout {suffix}:", ["spring", "circular", "kamada_kawai"], key=f"network_layout{suffix}")
        
        # Ensure there's enough data for a meaningful network graph
        if corr_matrix.shape[0] < 2:
            st.info(f"Not enough parameters in the correlation matrix to generate a network graph {suffix}.")
        else:
            if st.button(f"Generate Network Graph {suffix}", key=f"gen_network{suffix}"): # Added a button to trigger network generation
                network_fig = create_network_graph(corr_matrix, threshold=network_threshold, layout=layout_type)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True)
                else:
                    st.warning(f"Could not generate network graph {suffix}. Check data or reduce threshold.")


    with summary_tabs[2]:
        st.write(f"### Correlations by Parameter Category {suffix}")
        
        # Define categories based on your parameters (ensure these match your data's naming conventions)
        categories = {
            'Water Quality': ['pH', 'ΘΟΛΟΤΗΤΑ', 'ΑΓΩΓΙΜΟΤΗΤΑ', 'ΧΛΩΡΙΟΥΧΑ', 'ΣΚΛΗΡΟΤΗΤΑ', 'TOC', 'ΟΣΜΗ', 'ΧΡΩΜΑ', 'ΝΙΤΡΙΚΑ (ΝΟ3-)', 'ΝΙΤΡΩΔΗ (ΝΟ2-)', 'ΑΜΜΩΝΙΑΚΑ (ΝΗ4+)', 'ΣΙΔΗΡΟΣ', 'ΜΑΓΓΑΝΙΟ', 'ΑΡΓΙΛΙΟ', 'ΘΕΙΙΚΑ', 'ΦΩΣΦΟΡΟΣ (P2O5)', 'ΜΑΓΝΗΣΙΟ', 'ΑΣΒΕΣΤΙΟ', 'ΚΥΑΝΙΟΥΧΑ', 'ΦΘΟΡΙΟΥΧΑ'],
            'Reservoir': ['Στάθμη'],
            'Climate': ['Μέση Θερμοκρασία', 'Μέγιστη Θερμοκρασία', 'Ελάχιστη Θερμοκρασία', 'Βροχόπτωση', 'Μέση Ταχύτητα Ανέμου', 'Μέγιστη Ταχύτητα Ανέμου'],
            'Satellite': ['temperature', 'modelled_elevation', 'Chl a_S2lwa', 'Chl a_GEE', 'color_Se2lwa', 'TSM'],
        }
        
        # Add source suffixes to categories (this needs to match how create_properly_merged_dataframe names columns)
        expanded_categories = {}
        for cat, params in categories.items():
            expanded_params = []
            for param in params:
                # Iterate through sources in available_data to find matching columns
                # For averaged data, we need to check if the suffixed parameter exists in the *current* corr_matrix columns
                for source_name_key in ["Water Quality", "Climate", "Reservoir", "Satellite"]: # Use fixed source names
                    full_param_name = f"{param}_{source_name_key}"
                    if full_param_name in corr_matrix.columns:
                        expanded_params.append(full_param_name)
                    # Also include original param name if it exists directly (e.g. if not suffixed)
                    elif param in corr_matrix.columns and param not in expanded_params:
                        expanded_params.append(param)
            if expanded_params:
                expanded_categories[cat] = list(set(expanded_params)) # Use set to avoid duplicates

        if not expanded_categories:
            st.info(f"No defined categories match available parameters in the merged data {suffix}.")
        else:
            category_summary = create_correlation_by_category(corr_matrix, expanded_categories)
            if not category_summary.empty:
                st.dataframe(category_summary, use_container_width=True)
            else:
                st.info(f"No inter-category correlations found or categories too sparse {suffix}.")
            
            # Heatmap of category correlations
            if len(expanded_categories) > 1:
                st.write(f"### Inter-category Correlation Strength {suffix}")
                cat_names = list(expanded_categories.keys())
                cat_matrix = pd.DataFrame(index=cat_names, columns=cat_names, dtype=float)
                                        
                for cat1 in cat_names:
                    for cat2 in cat_names:
                        params1 = [p for p in expanded_categories[cat1] if p in corr_matrix.columns]
                        params2 = [p for p in expanded_categories[cat2] if p in corr_matrix.columns]
                        if params1 and params2:
                            sub_corr = corr_matrix.loc[params1, params2]
                            # Calculate average absolute correlation between category pairs
                            avg_abs_corr = sub_corr.abs().mean().mean()
                            cat_matrix.loc[cat1, cat2] = avg_abs_corr if not pd.isna(avg_abs_corr) else 0.0
                            
                fig_cat = go.Figure(data=go.Heatmap(
                    z=cat_matrix.values,
                    x=[cat for cat in cat_matrix.columns], # Use original category names
                    y=[cat for cat in cat_matrix.index],   # Use original category names
                    colorscale='Blues',
                    text=np.round(cat_matrix.values, 2),
                    texttemplate='%{text:.2f}'
                ))
                fig_cat.update_layout(title=f"Average Absolute Correlation Between Categories {suffix}", height=400)
                st.plotly_chart(fig_cat, use_container_width=True)


    with summary_tabs[3]:
        st.write(f"### Hierarchical Clustering of Parameters {suffix}")
        st.info("Parameters that cluster together tend to have similar correlation patterns across the entire dataset. This visualization groups them based on their similarity.")
        
        if corr_matrix.shape[0] < 2:
            st.info(f"Not enough parameters in the correlation matrix to generate a dendrogram {suffix}.")
        elif st.button(f"Generate Dendrogram {suffix}", key=f"gen_dendro{suffix}"):
            dendro_fig = create_correlation_dendrogram(corr_matrix)
            if dendro_fig:
                st.plotly_chart(dendro_fig, use_container_width=True)
            else:
                st.warning(f"Could not generate dendrogram {suffix}. This may happen if all parameters are perfectly correlated/anti-correlated or have no variation.")

# ============= END OF MISSING FUNCTIONS =============

# ============= ITS ANALYSIS FUNCTIONS =============
def perform_its_analysis(data, parameter, interruption_date, pre_trend_period=None):
    """
    Perform Interrupted Time Series Analysis
    
    Parameters:
    - data: DataFrame with Date and parameter columns
    - parameter: string name of parameter to analyze
    - interruption_date: datetime object representing the interruption point
    - pre_trend_period: optional tuple of (start_date, end_date) for pre-trend analysis
    
    Returns:
    - Dictionary with analysis results including model, statistics, and predictions
    """
    
    # Prepare data
    df = data[['Date', parameter]].copy()
    df = df.dropna()
    df = df.sort_values('Date')
    
    if len(df) < 20:
        return {'error': 'Insufficient data points for ITS analysis (minimum 20 required)'}
    
    # Create time variables
    df['time'] = range(len(df))
    df['interruption'] = (df['Date'] >= interruption_date).astype(int)
    df['time_since_interruption'] = np.where(
        df['interruption'] == 1, 
        df['time'] - df[df['interruption'] == 1]['time'].min(), 
        0
    )
    
    # Check if we have data both before and after interruption
    pre_count = (df['interruption'] == 0).sum()
    post_count = (df['interruption'] == 1).sum()
    
    if pre_count < 10 or post_count < 5:
        return {'error': f'Insufficient data before ({pre_count}) or after ({post_count}) interruption'}
    
    # Basic ITS model: Y = β0 + β1*time + β2*interruption + β3*time_since_interruption + ε
    X = df[['time', 'interruption', 'time_since_interruption']]
    X = sm.add_constant(X)  # Add intercept
    y = df[parameter]
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Get predictions
    df['predicted'] = model.predict(X)
    df['residuals'] = y - df['predicted']
    
    # Calculate counterfactual (what would have happened without interruption)
    X_counterfactual = df[['time', 'interruption', 'time_since_interruption']].copy()
    X_counterfactual['interruption'] = 0
    X_counterfactual['time_since_interruption'] = 0
    X_counterfactual = sm.add_constant(X_counterfactual)
    df['counterfactual'] = model.predict(X_counterfactual)
    
    # Diagnostic tests
    durbin_watson_stat = durbin_watson(df['residuals'])
    
    # Breusch-Pagan test for heteroscedasticity
    try:
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(df['residuals'], X)
    except:
        bp_stat, bp_pvalue = np.nan, np.nan
    
    # Calculate effect sizes
    interruption_effect = model.params['interruption']  # Immediate level change
    trend_change = model.params['time_since_interruption']  # Slope change
    
    # Calculate confidence intervals
    conf_int = model.conf_int()
    
    # Pre-trend analysis if requested
    pre_trend_results = None
    if pre_trend_period and len(df[df['Date'] < interruption_date]) > 10:
        pre_data = df[df['Date'] < interruption_date].copy()
        if pre_trend_period[0]:
            pre_data = pre_data[pre_data['Date'] >= pre_trend_period[0]]
        if pre_trend_period[1]:
            pre_data = pre_data[pre_data['Date'] <= pre_trend_period[1]]
        
        if len(pre_data) > 5:
            X_pre = sm.add_constant(pre_data['time'])
            y_pre = pre_data[parameter]
            pre_model = sm.OLS(y_pre, X_pre).fit()
            pre_trend_results = {
                'slope': pre_model.params['time'],
                'p_value': pre_model.pvalues['time'],
                'r_squared': pre_model.rsquared
            }
    
    return {
        'model': model,
        'data': df,
        'interruption_date': interruption_date,
        'pre_count': pre_count,
        'post_count': post_count,
        'interruption_effect': interruption_effect,
        'trend_change': trend_change,
        'conf_intervals': conf_int,
        'durbin_watson': durbin_watson_stat,
        'bp_test': {'statistic': bp_stat, 'p_value': bp_pvalue},
        'pre_trend': pre_trend_results,
        'effect_size_immediate': abs(interruption_effect) / df[parameter].std(),
        'effect_size_trend': abs(trend_change) / df[parameter].std()
    }

def plot_its_results(its_results, parameter_name):
    """
    Create visualization for ITS analysis results
    """
    if 'error' in its_results:
        return None
    
    df = its_results['data'].copy()  # Create a copy to avoid modifying original data
    interruption_date = its_results['interruption_date']
    
    # Ensure Date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    fig = go.Figure()
    
    # Original data points
    pre_data = df[df['interruption'] == 0]
    post_data = df[df['interruption'] == 1]
    
    # Pre-interruption data
    fig.add_trace(go.Scatter(
        x=pre_data['Date'],
        y=pre_data[parameter_name],
        mode='markers',
        name='Pre-interruption',
        marker=dict(color='blue', size=6),
        hovertemplate='Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
    ))
    
    # Post-interruption data
    fig.add_trace(go.Scatter(
        x=post_data['Date'],
        y=post_data[parameter_name],
        mode='markers',
        name='Post-interruption',
        marker=dict(color='red', size=6),
        hovertemplate='Date: %{x}<br>Value: %{y:.3f}<extra></extra>'
    ))
    
    # Fitted model line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['predicted'],
        mode='lines',
        name='ITS Model Fit',
        line=dict(color='green', width=3),
        hovertemplate='Date: %{x}<br>Predicted: %{y:.3f}<extra></extra>'
    ))
    
    # Counterfactual line (what would have happened without interruption)
    fig.add_trace(go.Scatter(
        x=post_data['Date'],
        y=post_data['counterfactual'],
        mode='lines',
        name='Counterfactual (No interruption)',
        line=dict(color='orange', width=2, dash='dash'),
        hovertemplate='Date: %{x}<br>Counterfactual: %{y:.3f}<extra></extra>'
    ))
    
    # Add vertical line for interruption
    # Ensure interruption_date is in the right format
    if isinstance(interruption_date, str):
        interruption_date = pd.to_datetime(interruption_date)
    
    # Use the datetime object directly with Plotly
    fig.add_shape(
        type="line",
        x0=interruption_date,
        y0=0,
        x1=interruption_date,
        y1=1,
        yref="paper",
        line=dict(color="red", width=3, dash="solid")
    )
    
    # Add annotation for the interruption
    fig.add_annotation(
        x=interruption_date,
        y=1,
        yref="paper",
        text="Wildfire Event",
        showarrow=False,
        font=dict(color="red", size=14),
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    # Add confidence intervals
    conf_int = its_results['model'].get_prediction().conf_int()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=conf_int[:, 1],
        mode='lines',
        line=dict(color='rgba(0,100,80,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=conf_int[:, 0],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(0,100,80,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Interrupted Time Series Analysis: {get_param_info(parameter_name)}',
        xaxis_title='Date',
        yaxis_title=f'{get_param_info(parameter_name)}',
        template="plotly_white",
        hovermode='x unified',
        height=600,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def create_its_diagnostics_plots(its_results, parameter_name):
    """
    Create diagnostic plots for ITS analysis
    """
    if 'error' in its_results:
        return None, None
    
    df = its_results['data']
    
    # Residuals plot
    fig_residuals = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals vs Fitted",
            "Residuals vs Time", 
            "Q-Q Plot of Residuals",
            "Residual Histogram"
        ),
        vertical_spacing=0.12
    )
    
    # 1. Residuals vs Fitted
    fig_residuals.add_trace(
        go.Scatter(
            x=df['predicted'],
            y=df['residuals'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.6),
            showlegend=False
        ),
        row=1, col=1
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Residuals vs Time
    fig_residuals.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['residuals'],
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.6),
            showlegend=False
        ),
        row=1, col=2
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. Q-Q Plot
    residuals_sorted = np.sort(df['residuals'])
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))
    
    fig_residuals.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=residuals_sorted,
            mode='markers',
            marker=dict(size=5, color='purple', opacity=0.6),
            showlegend=False
        ),
        row=2, col=1
    )
    # Add reference line
    min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
    max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
    fig_residuals.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='red'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Histogram of residuals
    fig_residuals.add_trace(
        go.Histogram(
            x=df['residuals'],
            nbinsx=20,
            marker_color='orange',
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update axis labels
    fig_residuals.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig_residuals.update_yaxes(title_text="Residuals", row=1, col=1)
    fig_residuals.update_xaxes(title_text="Date", row=1, col=2)
    fig_residuals.update_yaxes(title_text="Residuals", row=1, col=2)
    fig_residuals.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig_residuals.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    fig_residuals.update_xaxes(title_text="Residuals", row=2, col=2)
    fig_residuals.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig_residuals.update_layout(
        title=f"ITS Model Diagnostics: {get_param_info(parameter_name)}",
        height=700
    )
    
    # Model summary statistics
    model = its_results['model']
    
    summary_stats = {
        'R-squared': model.rsquared,
        'Adj. R-squared': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic,
        'Durbin-Watson': its_results['durbin_watson'],
        'BP Test p-value': its_results['bp_test']['p_value']
    }
    
    return fig_residuals, summary_stats

# ============= END OF ITS ANALYSIS FUNCTIONS =============

# --- UI Layout ---
with st.sidebar:
    st.image("https://www.freepnglogos.com/uploads/water-drop-png/water-drop-png-index-of-images-24.png", width=100)
    st.title("💧 Advanced Analytics")
    st.markdown("AI-Powered Water Quality Analysis")
    
    # File uploaders
    st.header("📁 Data Sources")
    
    with st.expander("1. Water Quality Data", expanded=True):
        if WATER_QUALITY_PATH and os.path.exists(WATER_QUALITY_PATH):
            xlsx_files = glob.glob(os.path.join(WATER_QUALITY_PATH, '*.xls*'))
            st.success(f"✓ {len(xlsx_files)} files loaded")
        else:
            xlsx_files = st.file_uploader("Upload XLSX/XLS", type=["xlsx", "xls"],
                                         accept_multiple_files=True, key="wq_uploader")

    with st.expander("2. Climate Data"):
        if CLIMATE_DATA_PATH and os.path.exists(CLIMATE_DATA_PATH):
            txt_files = glob.glob(os.path.join(CLIMATE_DATA_PATH, '*.txt'))
            st.success(f"✓ {len(txt_files)} files loaded")
        else:
            txt_files = st.file_uploader("Upload TXT", type="txt",
                                         accept_multiple_files=True, key="climate_uploader")
    
    with st.expander("3. Reservoir Level"):
        if RESERVOIR_LEVEL_PATH and os.path.exists(RESERVOIR_LEVEL_PATH):
            level_file = RESERVOIR_LEVEL_PATH
            st.success("✓ File loaded")
        else:
            level_file = st.file_uploader("Upload XLS/XLSX", type=["xls", "xlsx"],
                                         key="level_uploader")
    
    with st.expander("4. Satellite Data"):
        if SATELLITE_DATA_PATH and os.path.exists(SATELLITE_DATA_PATH):
            satellite_file = SATELLITE_DATA_PATH
            st.success("✓ File loaded")
        else:
            satellite_file = st.file_uploader("Upload XLS/XLSX", type=["xls", "xlsx"],
                                               key="satellite_uploader")
    
    st.markdown("---")
    
    # Analysis Settings
    st.header("⚙️ Analysis Settings")
    anomaly_threshold = st.slider("Anomaly Detection Sensitivity", 0.01, 0.20, 0.05, 0.01)
    lag_days = st.slider("Max Lag Days for Correlation", 7, 60, 30, 7)
    
    # Export button
    if st.button("📊 Export Analysis Report", type="primary"):
        st.info("Report export functionality coming soon!")

# Main content area
st.title("🌊 Advanced Water & Climate Analysis Dashboard")
st.markdown("*AI-powered insights for environmental monitoring*")

# Initialize file variables if not set by uploaders
if 'xlsx_files' not in locals():
    xlsx_files = []
if 'txt_files' not in locals():
    txt_files = []
if 'level_file' not in locals():
    level_file = None
if 'satellite_file' not in locals():
    satellite_file = None

# Process data
if not any([xlsx_files, txt_files, level_file, satellite_file]):
    st.info("👋 Welcome! Please upload data files using the sidebar to begin analysis.")
    
    # Show demo info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Features", "8+ Analysis Types", "Advanced ML")
    with col2:
        st.metric("Parameters", "40+ Monitored", "Real-time alerts")
    with col3:
        st.metric("Insights", "Automated", "AI-powered")
else:
    # Process all data
    data_xlsx = process_xlsx_files(xlsx_files)
    data_txt = process_txt_files(txt_files)
    data_level = process_level_file(level_file)
    data_satellite = process_satellite_file(satellite_file)
    
    # Create data dictionary
    data_dict = {
        "Water Quality": data_xlsx,
        "Climate": data_txt,
        "Reservoir": data_level,
        "Satellite": data_satellite
    }
    
    # Generate insights and alerts
    available_data = {k: v for k, v in data_dict.items() if v is not None}
    if available_data:
        st.session_state.insights = generate_insights(available_data)
        st.session_state.alerts = create_alert_system(available_data)
    
    # Alert Dashboard
    if st.session_state.alerts:
        st.error(f"🚨 {len([a for a in st.session_state.alerts if a['level'] == 'critical'])} Critical Alerts")
        st.warning(f"⚠️ {len([a for a in st.session_state.alerts if a['level'] == 'warning'])} Warnings")
        
        with st.expander("View Active Alerts", expanded=True):
            for alert in st.session_state.alerts[:5]:  # Show top 5
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if alert['level'] == 'critical':
                        st.error("🚨 CRITICAL")
                    else:
                        st.warning("⚠️ WARNING")
                with col2:
                    st.write(f"**{alert['parameter']}** - {alert['message']}")
                    st.caption(f"Source: {alert['source']} | Value: {alert['value']}")
                with col3:
                    st.caption(alert['timestamp'].strftime('%Y-%m-%d'))
    
    # Quick Insights Panel
    if st.session_state.insights:
        st.subheader("🔍 Quick Insights")
        insight_cols = st.columns(3)
        for idx, insight in enumerate(st.session_state.insights[:6]):
            with insight_cols[idx % 3]:
                st.info(f"{insight['icon']} {insight['message']}")
    
    # Main Analysis Tabs
    tab_names = ["📊 Overview", "💧 Water Quality", "☀️ Climate", "🌊 Reservoir", "🛰️ Satellite", 
                 "🔗 Raw Data Correlations", "📈 Averaged Data Correlations", "🔀 Non-Linear Correlations", 
                 "🔬 Advanced Correlations", "🔥 ITS Analysis", "🤖 ML Analysis", "📊 Predictions"] # Added ITS Analysis tab
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_params = sum(len(df.columns) - 1 for df in available_data.values() if df is not None)
            st.metric("Total Parameters", total_params)
        
        with col2:
            total_records = sum(len(df) for df in available_data.values() if df is not None)
            st.metric("Total Records", f"{total_records:,}")
        
        with col3:
            date_range = []
            for df in available_data.values():
                if df is not None and 'Date' in df.columns and not df.empty:
                    date_range.extend([df['Date'].min(), df['Date'].max()])
            if date_range:
                days = (max(date_range) - min(date_range)).days
                st.metric("Date Range", f"{days} days")
        
        with col4:
            quality_score = 100 - (len(st.session_state.alerts) * 5)  # Simple score
            st.metric("Quality Score", f"{max(0, quality_score)}%",
                                     f"{len(st.session_state.alerts)} issues")
        
        # Data availability heatmap
        st.subheader("Data Availability Matrix")
        availability_data = []
        for source, df in available_data.items():
            if df is not None and not df.empty:
                params = [col for col in df.columns if col != 'Date']
                for param in params[:10]:  # Limit for visualization
                    completeness = (df[param].notna().sum() / len(df)) * 100
                    availability_data.append({
                        'Source': source,
                        'Parameter': param,
                        'Completeness': completeness
                    })
        
        if availability_data:
            avail_df = pd.DataFrame(availability_data)
            pivot_df = avail_df.pivot(index='Parameter', columns='Source', values='Completeness')
            
            fig = px.imshow(pivot_df,
                             labels=dict(x="Data Source", y="Parameter", color="Completeness (%)"),
                             color_continuous_scale="RdYlGn",
                             title="Data Completeness Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    
    # Water Quality Tab
   # Water Quality Tab
with tabs[1]:
    st.header("💧 Water Quality Analysis")
    if data_xlsx is not None and not data_xlsx.empty:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get a list of parameters that have at least one non-null value
            params = [c for c in data_xlsx.columns if c != 'Date' and data_xlsx[c].notna().any()]
            selected_params = st.multiselect(
                "Select parameters to visualize:", 
                params,
                default=params[:2] if params else [],
                key="wq_params_select"
            )
        
        with col2:
            show_anomalies = st.checkbox("Show Anomalies", value=True, key="wq_anom")
        
        # Add a separator
        st.markdown("---")

        # Loop through each selected parameter to create its plot and download buttons
        for param in selected_params:
            with st.container(): # Use a container to group plot and its buttons
                st.subheader(f"Time Profile for {param}")
                
                # Generate the plot
                fig = plot_parameter_advanced(data_xlsx, param, show_anomalies)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Add download buttons for this specific plot's data ---
                    plot_data_df = data_xlsx[['Date', param]].dropna(subset=[param])
                    add_plot_download_button(plot_data_df, f"water_quality_{param.replace(' ', '_')}")

                else:
                    st.warning(f"Could not generate plot for {param}.")

                # Parameter statistics expander
                with st.expander(f"📊 View Statistics for {param}"):
                    param_data = data_xlsx[param].dropna()
                    if not param_data.empty:
                        stat_cols = st.columns(4)
                        stat_cols[0].metric("Mean", f"{param_data.mean():.2f}")
                        stat_cols[1].metric("Std Dev", f"{param_data.std():.2f}")
                        stat_cols[2].metric("Min", f"{param_data.min():.2f}")
                        stat_cols[3].metric("Max", f"{param_data.max():.2f}")
                    else:
                        st.info("No statistical data available.")
                
                st.markdown("---") # Add a separator between plots
    else:
        st.info("No water quality data has been uploaded. Please add data in the sidebar to begin analysis.")
    
    # Climate Tab
    # Climate Tab
    # Climate Tab
    with tabs[2]:
        st.header("Climate Analysis")
        if data_txt is not None and not data_txt.empty:
            # Climate parameter selection
            climate_params = [c for c in data_txt.columns if c != 'Date' and c != 'Επικρατούσα Διεύθυνση Ανέμου' and data_txt[c].notna().any()]
            selected_climate = st.multiselect("Select climate parameters:", climate_params,
                                                 default=climate_params[:1] if climate_params else [])
            
            show_anomalies_climate = st.checkbox("Show Anomalies", value=True, key="climate_anom")
            
            # Add the download button
            create_download_button(
                df=data_txt,
                selected_columns=selected_climate,
                sheet_name="Climate_Data",
                file_name_prefix="climate"
            )
            
            for param in selected_climate:
                fig = plot_parameter_advanced(data_txt, param, show_anomalies_climate)
                st.plotly_chart(fig, use_container_width=True)
            
            # --- Seasonal Rainfall Analysis ---
            st.markdown("---")
            st.subheader("🌦️ Seasonal Rainfall Analysis")

            if 'Βροχόπτωση' in data_txt.columns:
                # Create a copy for analysis
                rain_df = data_txt[['Date', 'Βροχόπτωση']].copy()
                rain_df['Date'] = pd.to_datetime(rain_df['Date'])
                rain_df.dropna(subset=['Βροχόπτωση'], inplace=True)
                rain_df = rain_df.sort_values('Date')

                # Define a function to assign season
                def get_season(month):
                    if month in [12, 1, 2]: return 'Winter'
                    elif month in [3, 4, 5]: return 'Spring'
                    elif month in [6, 7, 8]: return 'Summer'
                    else: return 'Autumn'

                rain_df['Season'] = rain_df['Date'].dt.month.apply(get_season)
                rain_df['Year'] = rain_df['Date'].dt.year

                # --- 1. Summary Table of Total Rainfall per Season ---
                st.write("**Total Rainfall per Season (mm)**")
                seasonal_totals = rain_df.groupby(['Year', 'Season'])['Βροχόπτωση'].sum().unstack(fill_value=0)
                season_order = [s for s in ['Winter', 'Spring', 'Summer', 'Autumn'] if s in seasonal_totals.columns]
                seasonal_totals = seasonal_totals[season_order]
                st.dataframe(seasonal_totals.style.format("{:.1f}").background_gradient(cmap='Blues'))

                # --- 2. Plot of Daily Cumulative Rainfall (Calendar Seasons) ---
                st.write("**Overall Daily Cumulative Rainfall**")
                rain_df['Cumulative Rainfall'] = rain_df['Βροχόπτωση'].cumsum()

                fig_cumulative_rain = px.line(
                    rain_df, x='Date', y='Cumulative Rainfall', color='Season',
                    title='Daily Accumulation of Rainfall (Colored by Calendar Season)',
                    labels={'Cumulative Rainfall': 'Cumulative Rainfall (mm)'},
                    template='plotly_white',
                    color_discrete_map={
                        "Winter": "blue", "Spring": "green",
                        "Summer": "red", "Autumn": "goldenrod"
                    }
                )
                st.plotly_chart(fig_cumulative_rain, use_container_width=True)

                # --- 3. NEW: Hydrological Season Accumulation ---
                st.markdown("---")
                st.subheader("💧 Hydrological Season Rainfall Accumulation")
                st.info("This plot shows the rainfall accumulation that resets for each rainy season, which is defined as running from September 1st to August 31st.")

                # Define and apply the hydrological year
                def get_hydrological_year(date):
                    if date.month >= 9:
                        return f"{date.year}-{date.year + 1}"
                    else:
                        return f"{date.year - 1}-{date.year}"

                rain_df['Hydrological Year'] = rain_df['Date'].apply(get_hydrological_year)

                # Calculate cumulative sum *within* each hydrological year group
                rain_df['Season Accumulation'] = rain_df.groupby('Hydrological Year')['Βροχόπτωση'].cumsum()

                # Create the resetting accumulation plot
                fig_hydro_season = px.line(
                    rain_df,
                    x='Date',
                    y='Season Accumulation',
                    color='Hydrological Year',
                    title='Rainfall Accumulation per Rainy Season',
                    labels={'Season Accumulation': 'Cumulative Rainfall (mm)', 'Date': 'Date'},
                    template='plotly_white'
                )
                fig_hydro_season.update_layout(hovermode='x unified')
                st.plotly_chart(fig_hydro_season, use_container_width=True)

            else:
                st.info("Rainfall data ('Βροχόπτωση') not available for seasonal analysis.")
        else:
            st.info("No climate data available")
    
    # Reservoir Tab
    with tabs[3]:
        st.header("Reservoir Level Analysis")
        if data_level is not None and not data_level.empty:
            show_anomalies_level = st.checkbox("Show Anomalies", value=True, key="level_anom")
            
            # Add the download button for the entire reservoir dataset
            create_download_button(
                df=data_level,
                selected_columns=data_level.columns.tolist(), # Select all columns
                sheet_name="Reservoir_Level",
                file_name_prefix="reservoir"
            )

            fig = plot_parameter_advanced(data_level, 'Στάθμη', show_anomalies_level)
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume calculation (if applicable)
            st.subheader("Reservoir Statistics")
            col1, col2, col3 = st.columns(3)
            
            current_level = data_level['Στάθμη'].iloc[-1]
            avg_level = data_level['Στάθμη'].mean()
            change_rate = data_level['Στάθμη'].diff().mean()
            
            col1.metric("Current Level", f"{current_level:.2f} m",
                                         f"{current_level - avg_level:.2f} from avg")
            col2.metric("Average Level", f"{avg_level:.2f} m")
            col3.metric("Daily Change Rate", f"{change_rate:.3f} m/day")
        else:
            st.info("No reservoir data available")
    
    # Satellite Tab
    with tabs[4]:
        st.header("Satellite Data Analysis")
        if data_satellite is not None and not data_satellite.empty:
            sat_params = [c for c in data_satellite.columns if c != 'Date' and data_satellite[c].notna().any()]
            selected_sat = st.multiselect("Select satellite parameters:", sat_params,
                                       default=sat_params[:2] if sat_params else [])
            
            show_anomalies_sat = st.checkbox("Show Anomalies", value=True, key="sat_anom")
            
            # Add the download button
            create_download_button(
                df=data_satellite,
                selected_columns=selected_sat,
                sheet_name="Satellite_Data",
                file_name_prefix="satellite"
            )
            
            for param in selected_sat:
                fig = plot_parameter_advanced(data_satellite, param, show_anomalies_sat)
                st.plotly_chart(fig, use_container_width=True)
            
            # Chlorophyll comparison if both available
            if 'Chl a_S2lwa' in data_satellite.columns and 'Chl a_GEE' in data_satellite.columns:
                st.subheader("Chlorophyll-a Method Comparison")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data_satellite['Chl a_S2lwa'],
                                         y=data_satellite['Chl a_GEE'],
                                         mode='markers', name='Data Points'))
                
                # Add 1:1 line
                min_val = min(data_satellite['Chl a_S2lwa'].min(), data_satellite['Chl a_GEE'].min())
                max_val = max(data_satellite['Chl a_S2lwa'].max(), data_satellite['Chl a_GEE'].max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                         mode='lines', name='1:1 Line', line=dict(dash='dash')))
                
                fig.update_layout(title="S2lwa vs GEE Chlorophyll-a Comparison",
                                         xaxis_title="Chl a_S2lwa (mg/m³)",
                                         yaxis_title="Chl a_GEE (mg/m³)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No satellite data available")
    
    # --- Original Raw Data Correlations Tab (complete with multi-series) ---
    with tabs[5]:
        st.header("🔗 Raw Data Correlations")
        st.info("This tab presents correlation analyses and time series plots based on the **raw daily merged data**.")

        merged_data = create_properly_merged_dataframe(available_data)

        if merged_data is not None and not merged_data.empty:
            numeric_cols = [col for col in merged_data.columns
                            if col != 'Date' and pd.api.types.is_numeric_dtype(merged_data[col])]

            if not numeric_cols:
                st.warning("No numeric data available in the raw merged dataset for correlation analysis.")
            else:
                # --- Multi-Series Plotter ---
                st.markdown("---")
                st.subheader("Multi-Series Time Comparison (Raw Data)")
                st.info("Select multiple parameters to plot them on a single timeline for a general overview.")

                selected_params_multi = st.multiselect(
                    "Select parameters to plot:",
                    options=numeric_cols,
                    format_func=get_param_info,
                    default=numeric_cols[:2] if len(numeric_cols) > 1 else numeric_cols,
                    key="multi_select_raw_tab"
                )

                col_multi_1, col_multi_2 = st.columns(2)
                with col_multi_1:
                    use_first_difference_multi = st.checkbox(
                        "Analyze First Difference (Rate of Change)",
                        key="use_first_diff_raw_tab",
                        help="If checked, the plot will show the daily change instead of the absolute values."
                    )
                with col_multi_2:
                    normalize_data_multi_checked = False
                    if len(selected_params_multi) > 1:
                        normalize_data_multi_checked = st.checkbox(
                            "Normalize data for comparison (0-1 scale)",
                            key="normalize_data_raw",
                            help="Scale all parameters to a 0-1 range for easier visual comparison of trends."
                        )

                if not selected_params_multi:
                    st.warning("Please select at least one parameter to display the multi-series chart.")
                else:
                    data_to_plot_multi = merged_data.copy()
                    plot_params_multi = selected_params_multi
                    title_suffix_multi = "(Raw Data)"

                    if use_first_difference_multi:
                        title_suffix_multi = "(Raw Data First Difference)"
                        new_diff_cols_multi = []
                        for param in selected_params_multi:
                            diff_col_name = f"{param} (Difference)"
                            data_to_plot_multi[diff_col_name] = data_to_plot_multi[param].diff()
                            new_diff_cols_multi.append(diff_col_name)
                        plot_params_multi = new_diff_cols_multi
                    
                    # Create the plot
                    fig_multi = go.Figure()
                    colors = px.colors.qualitative.Plotly

                    for i, param in enumerate(plot_params_multi):
                        param_data = data_to_plot_multi[['Date', param]].dropna()
                        if not param_data.empty:
                            y_values = param_data[param]
                            display_name = get_param_info(param)

                            if normalize_data_multi_checked:
                                param_min, param_max = y_values.min(), y_values.max()
                                if param_max > param_min:
                                    y_values = (y_values - param_min) / (param_max - param_min)
                                    display_name = f"{display_name} (normalized)"
                            
                            fig_multi.add_trace(go.Scatter(
                                x=param_data['Date'], y=y_values, name=display_name,
                                line=dict(color=colors[i % len(colors)]), mode='lines',
                                connectgaps=False
                            ))
                    
                    fig_multi.update_layout(
                        title=f"Multi-Series Comparison {title_suffix_multi}",
                        xaxis_title="Date",
                        yaxis_title="Normalized Values (0-1)" if normalize_data_multi_checked else "Values",
                        template="plotly_white", hovermode='x unified', height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_multi, use_container_width=True)


                # --- DETAILED PARAMETER COMPARISON (on raw data) ---
                st.markdown("---")
                st.subheader("Detailed Parameter Comparison (Raw Data)")
                st.info("Select two parameters to perform a detailed analysis, including cross-correlation and scatter plots.")

                col1, col2 = st.columns(2)
                with col1:
                    x_axis_param_raw = st.selectbox(
                        "Select first parameter:",
                        options=numeric_cols, format_func=get_param_info, key="scatter_x_fixed_raw_tab"
                    )
                with col2:
                    y_axis_param_raw = st.selectbox(
                        "Select second parameter:",
                        options=numeric_cols,
                        index=min(1, len(numeric_cols) - 1) if len(numeric_cols) > 1 else 0,
                        format_func=get_param_info, key="scatter_y_fixed_raw_tab"
                    )

                if x_axis_param_raw and y_axis_param_raw:
                    # Dual-axis comparison of the selected pair
                    st.subheader(f"Dual-Axis Comparison: {get_param_info(x_axis_param_raw)} vs {get_param_info(y_axis_param_raw)}")
                    fig_dual = plot_dual_axis_time_series_fixed(merged_data, x_axis_param_raw, y_axis_param_raw)
                    if fig_dual:
                        st.plotly_chart(fig_dual, use_container_width=True)

                    # Cross-Correlation Plot for the selected pair
                    if x_axis_param_raw != y_axis_param_raw:
                        st.subheader("Cross-Correlation Plot")
                        if st.button("Calculate Cross-Correlation", key="calc_cross_corr_window_raw_tab"):
                            with st.spinner("Calculating cross-correlation for raw data..."):
                                cross_corr_results, cross_corr_warnings = calculate_lagged_correlations(merged_data, x_axis_param_raw, y_axis_param_raw, max_lag=lag_days, freq='D')
                                # ... (rest of the cross-correlation plotting logic) ...

                    # Scatter plot analysis for the selected pair
                    if x_axis_param_raw != y_axis_param_raw:
                        st.subheader("Scatter Plot Analysis")
                        fig_scatter = plot_correlation_scatter(merged_data, x_axis_param_raw, y_axis_param_raw, show_nonlinear_trend=True)
                        if fig_scatter:
                            st.plotly_chart(fig_scatter, use_container_width=True)

                # --- Overall Correlation and Lag Matrices ---
                st.markdown("---")
                st.subheader("Overall Correlation Matrix (All Raw Data)")
                # ... (the rest of the code for the correlation matrices and summaries remains the same) ...

        else:
            st.error("❌ Failed to merge data or no data available for raw data correlation analysis.")
            st.info("Please check that your data files contain proper date columns and numeric data.")

    # --- NEW Averaged Data Correlations Tab ---
    with tabs[6]: # This is the new tab
        st.header("📈 Averaged Data Correlations")
        st.info("This tab allows you to perform correlation analyses on data averaged over specified periods, helping to reveal longer-term relationships and reduce noise.")

        merged_data = create_properly_merged_dataframe(available_data) # Ensure merged_data is available

        if merged_data is not None and not merged_data.empty:
            # Get all numeric columns from the original merged data for initial checks
            numeric_cols_all = [col for col in merged_data.columns
                                if col != 'Date' and pd.api.types.is_numeric_dtype(merged_data[col])]

            if not numeric_cols_all:
                st.warning("No numeric data available in the merged dataset for averaged correlation analysis. Please upload data with numeric parameters.")
            else:
                # Averaging Period Selection
                st.subheader("Data Averaging Configuration")
                col_avg_1, col_avg_2 = st.columns([1, 2])
                with col_avg_1:
                    selected_averaging_period = st.selectbox(
                        "Select Data Averaging Period:",
                        options=['3-Day Average', '7-Day Average', '15-Day Average', '30-Day Average'],
                        index=1, # Default to 7-Day Average
                        key="averaged_analysis_avg_period",
                        help="Choose an averaging period to smooth data. This will affect all analyses in this tab."
                    )

                # Determine resampling frequency string for pandas based on selection
                resample_freq_map = {
                    '3-Day Average': '3D',
                    '7-Day Average': '7D',
                    '15-Day Average': '15D',
                    '30-Day Average': '30D'
                }
                resample_freq = resample_freq_map.get(selected_averaging_period, '7D') # Default to '7D' if key not found

                # Apply resampling/averaging to the entire merged_data
                data_for_averaged_analysis = merged_data.copy()
                data_for_averaged_analysis = data_for_averaged_analysis.set_index('Date')
                data_for_averaged_analysis = data_for_averaged_analysis.resample(resample_freq).mean(numeric_only=True).reset_index()
                data_for_averaged_analysis['Date'] = pd.to_datetime(data_for_averaged_analysis['Date'])

                with col_avg_2:
                    st.info(f"Data will be averaged over {selected_averaging_period.replace('Average', 'day periods')}. Total records after averaging: {len(data_for_averaged_analysis)}.")
                    st.caption("Note: Averaging introduces NaNs for periods without sufficient underlying data. Periods with all NaNs will also be dropped if `dropna` is implicitly applied later, or explicitly if needed.")

                # Update numeric_cols based on the averaged data (only keep columns that are still numeric and have data)
                numeric_cols_averaged = [col for col in data_for_averaged_analysis.columns
                                         if col != 'Date' and pd.api.types.is_numeric_dtype(data_for_averaged_analysis[col]) and data_for_averaged_analysis[col].notna().any()]

                if not numeric_cols_averaged:
                    st.warning("No numeric parameters with valid data available after selected averaging. Please adjust averaging period or check raw data for completeness.")
                else:
                    # --- LAGGED CORRELATION MATRIX (on averaged data) ---
                    st.markdown("---")
                    st.subheader("Lagged Correlation Matrix (Averaged Data)")
                    st.info(f"This matrix shows the optimal time delay (in {selected_averaging_period.lower().replace(' average', '-day average')} periods) between parameter pairs. A positive value (red) means the Y-axis parameter leads the X-axis parameter.")

                    use_clustering_lag_avg = st.checkbox("Apply hierarchical clustering to organize parameters in Lag Matrix", value=True, key="use_clustering_lag_avg_corr_tab")

                    if st.button(f"Calculate Lag Matrix (Averaged Data)", key="lag_matrix_button_avg_corr_tab"):
                        with st.spinner(f"Calculating optimal lags for all parameter pairs using {selected_averaging_period.lower()} data..."):
                            if len(numeric_cols_averaged) >= 2:
                                # Call calculate_lag_matrix with the selected averaged data and its frequency
                                lag_matrix_avg, corr_at_lag_matrix_avg, lag_matrix_warnings_avg = calculate_lag_matrix(data_for_averaged_analysis, numeric_cols_averaged, lag_days, freq=resample_freq)

                                st.session_state.analysis_cache['lag_matrix_avg'] = lag_matrix_avg
                                st.session_state.analysis_cache['corr_at_lag_matrix_avg'] = corr_at_lag_matrix_avg
                                st.session_state.analysis_cache['lag_matrix_warnings_avg'] = lag_matrix_warnings_avg
                            else:
                                st.warning("Not enough numeric columns (at least 2) with valid data to calculate a lag matrix on averaged data.")
                                st.session_state.analysis_cache['lag_matrix_avg'] = pd.DataFrame()
                                st.session_state.analysis_cache['corr_at_lag_matrix_avg'] = pd.DataFrame()
                                st.session_state.analysis_cache['lag_matrix_warnings_avg'] = ["Not enough numeric columns available in averaged data to calculate full lagged correlation matrix."]

                    if 'lag_matrix_avg' in st.session_state.analysis_cache and not st.session_state.analysis_cache['lag_matrix_avg'].empty:
                        lag_matrix_avg = st.session_state.analysis_cache['lag_matrix_avg']
                        corr_at_lag_matrix_avg = st.session_state.analysis_cache['corr_at_lag_matrix_avg']
                        lag_matrix_warnings_avg = st.session_state.analysis_cache.get('lag_matrix_warnings_avg', [])

                        if lag_matrix_warnings_avg:
                            with st.expander("Show Warnings/Notes from Averaged Lag Matrix Calculation"):
                                for warning_msg in lag_matrix_warnings_avg:
                                    st.info(f"- {warning_msg}")

                        # Create custom text for the heatmap cells, reflecting averaging periods
                        hover_text_avg = []
                        for yi, y_param in enumerate(lag_matrix_avg.index):
                            hover_text_avg.append([])
                            for xi, x_param in enumerate(lag_matrix_avg.columns):
                                lag_val = lag_matrix_avg.iloc[yi, xi]
                                corr_val = corr_at_lag_matrix_avg.iloc[xi, yi] # Use symmetric value for corr
                                hover_text_avg[-1].append(f'Y: {get_param_info(y_param)}<br>X: {get_param_info(x_param)}<br>Lag: {lag_val:.0f} periods<br>Corr at Lag: {corr_val:.2f}')

                        fig_lag_avg = go.Figure(data=go.Heatmap(
                            z=lag_matrix_avg.values,
                            x=[get_param_info(col) for col in lag_matrix_avg.columns],
                            y=[get_param_info(col) for col in lag_matrix_avg.index],
                            colorscale='RdBu',
                            zmid=0,
                            text=np.round(lag_matrix_avg.values, 0),
                            texttemplate='%{text}P', # 'P' for periods
                            hovertext=hover_text_avg,
                            hoverinfo='text'
                        ))
                        fig_lag_avg.update_layout(
                            title=f"Optimal Time Lag Matrix (in {selected_averaging_period.replace('Average', 'Average Periods')})",
                            height=800,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_lag_avg, use_container_width=True)

                        fig_corr_at_lag_avg = go.Figure(data=go.Heatmap(
                            z=corr_at_lag_matrix_avg.values,
                            x=[get_param_info(col) for col in corr_at_lag_matrix_avg.columns],
                            y=[get_param_info(col) for col in corr_at_lag_matrix_avg.index],
                            colorscale='RdBu',
                            zmid=0,
                            text=np.round(corr_at_lag_matrix_avg.values, 2),
                            texttemplate='%{text:.2f}',
                            hovertext=hover_text_avg,
                            hoverinfo='text'
                        ))
                        fig_corr_at_lag_avg.update_layout(
                            title=f"Correlation at Optimal Time Lag ({selected_averaging_period} Data)",
                            height=800,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_corr_at_lag_avg, use_container_width=True)
                    else:
                        st.info("No lagged correlation matrix available for averaged data. Click 'Calculate Lag Matrix (Averaged Data)' above.")


                    # --- DETAILED PARAMETER COMPARISON (on averaged data) ---
                    st.markdown("---")
                    st.subheader("Detailed Parameter Comparison (on Averaged Data)")

                    col1, col2 = st.columns(2)

                    with col1:
                        x_axis_param_avg = st.selectbox(
                            "Select first parameter:",
                            options=numeric_cols_averaged,
                            format_func=get_param_info,
                            key="scatter_x_fixed_avg_tab" # Unique key
                        )

                    with col2:
                        y_axis_param_avg = st.selectbox(
                            "Select second parameter:",
                            options=numeric_cols_averaged,
                            index=min(1, len(numeric_cols_averaged)-1) if len(numeric_cols_averaged) > 1 else 0,
                            format_func=get_param_info,
                            key="scatter_y_fixed_avg_tab" # Unique key
                        )

                    # --- NEW: Checkbox for First Difference ---
                    use_first_difference = st.checkbox(
                        "Analyze First Difference",
                        key="use_first_diff_avg_tab",
                        help="If checked, all plots and calculations below will be based on the change from one period to the next (period-over-period change)."
                    )


                    # Add the Date Range Selection (still useful for filtering the averaged data further)
                    st.subheader("Time Window for Detailed Analysis (of averaged data)")

                    if not data_for_averaged_analysis.empty and 'Date' in data_for_averaged_analysis.columns:
                        start_date_overall_avg_tab = data_for_averaged_analysis['Date'].min().date()
                        end_date_overall_avg_tab = data_for_averaged_analysis['Date'].max().date()
                    else:
                        end_date_overall_avg_tab = datetime.date.today()
                        start_date_overall_avg_tab = end_date_overall_avg_tab - datetime.timedelta(days=365)
                        st.warning("Could not determine overall date range for slider on averaged data. Using a default range.")

                    selected_start_date_avg_tab, selected_end_date_avg_tab = st.slider(
                        "Filter averaged data by date range:",
                        min_value=start_date_overall_avg_tab,
                        max_value=end_date_overall_avg_tab,
                        value=(start_date_overall_avg_tab, end_date_overall_avg_tab),
                        format="YYYY-MM-DD",
                        key="detailed_analysis_date_range_avg_tab" # Unique key
                    )

                    # Filter `data_for_averaged_analysis` by date range
                    current_filtered_averaged_data_tab = data_for_averaged_analysis[
                        (data_for_averaged_analysis['Date'].dt.date >= selected_start_date_avg_tab) &
                        (data_for_averaged_analysis['Date'].dt.date <= selected_end_date_avg_tab)
                    ]

                    # --- NEW: Logic to handle first difference ---
                    data_to_plot = current_filtered_averaged_data_tab.copy()
                    plot_x_param = x_axis_param_avg
                    plot_y_param = y_axis_param_avg
                    analysis_mode_label = f"({selected_averaging_period} Data)"

                    if use_first_difference:
                        st.info("ℹ️ Analyzing the first difference (period-over-period change) of the selected parameters.")
                        # Create new column names for the differenced data
                        plot_x_param = f"{x_axis_param_avg} (Difference)"
                        plot_y_param = f"{y_axis_param_avg} (Difference)"
                        analysis_mode_label = f"({selected_averaging_period} First Difference)"

                        # Calculate the difference and create new columns
                        data_to_plot[plot_x_param] = data_to_plot[x_axis_param_avg].diff()
                        data_to_plot[plot_y_param] = data_to_plot[y_axis_param_avg].diff()

                        # Important: Remove the first row which will have NaN after differencing
                        data_to_plot.dropna(subset=[plot_x_param, plot_y_param], inplace=True)
                    # --- END NEW LOGIC ---

                    if data_to_plot.empty:
                        st.warning("No averaged data available for the selected date range and parameters for detailed analysis. Adjust filters or select different parameters.")
                    else:
                        st.info(f"Analyzing {selected_averaging_period.lower()} data from {selected_start_date_avg_tab.strftime('%Y-%m-%d')} to {selected_end_date_avg_tab.strftime('%Y-%m-%d')} ({len(current_filtered_averaged_data_tab)} records).")

                        # Analysis options (these can still be toggled)
                        col3, col4 = st.columns(2)
                        with col3:
                            show_scatter_avg_tab = st.checkbox("Show Scatter Plot", value=True, key="show_scatter_avg_tab")
                        with col4:
                            show_time_series_avg_tab = st.checkbox("Show Time Series", value=True, key="show_ts_avg_tab")

                        if x_axis_param_avg and y_axis_param_avg:

                            # Show individual time series (use data_to_plot and new param names)
                            if show_time_series_avg_tab:
                                st.subheader(f"Individual Time Series {analysis_mode_label}")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    fig1 = plot_time_series_fixed(data_to_plot, plot_x_param, f"({get_param_info(plot_x_param)})")
                                    if fig1: st.plotly_chart(fig1, use_container_width=True)
                                    else: st.error(f"Could not plot {plot_x_param} for this range on averaged data.")
                                with col_b:
                                    fig2 = plot_time_series_fixed(data_to_plot, plot_y_param, f"({get_param_info(plot_y_param)})")
                                    if fig2: st.plotly_chart(fig2, use_container_width=True)
                                    else: st.error(f"Could not plot {plot_y_param} for this range on averaged data.")

                            # Dual-axis comparison (use data_to_plot and new param names)
                            st.subheader(f"Dual-Axis Comparison {analysis_mode_label}")
                            fig_dual = plot_dual_axis_time_series_fixed(data_to_plot, plot_x_param, plot_y_param)
                            if fig_dual: st.plotly_chart(fig_dual, use_container_width=True)

                            # Cross-Correlation Plot for Selected Parameters (use data_to_plot and new param names)
                            st.subheader(f"Cross-Correlation Plot {analysis_mode_label}")
                            st.info(f"Shows the correlation between {get_param_info(plot_x_param)} and {get_param_info(plot_y_param)} at various time lags. Lag units are {selected_averaging_period.replace('Average', 'day average').lower()} periods.")

                            if x_axis_param_avg != y_axis_param_avg:
                                if st.button(f"Calculate Cross-Correlation for this window {analysis_mode_label}", key="calc_cross_corr_window_avg_tab"): # Unique key
                                    with st.spinner(f"Calculating cross-correlation for {analysis_mode_label} data..."):
                                        cross_corr_results_avg, cross_corr_warnings_avg = calculate_lagged_correlations(data_to_plot, plot_x_param, plot_y_param, max_lag=lag_days, freq=resample_freq) # Pass resample_freq

                                        if cross_corr_warnings_avg:
                                            for warning_msg in cross_corr_warnings_avg:
                                                st.info(f"Note for this pair: {warning_msg}")

                                        if not cross_corr_results_avg.empty:
                                            fig_cross_corr_avg = px.bar(
                                                cross_corr_results_avg,
                                                x='lag',
                                                y='correlation',
                                                title=f'Cross-Correlation: {get_param_info(plot_x_param)} vs {get_param_info(plot_y_param)} {analysis_mode_label}',
                                                labels={'lag': f'Lag ({selected_averaging_period.replace("Average", "day average")} Periods)', 'correlation': 'Correlation Coefficient'},
                                                template="plotly_white",
                                                color_discrete_sequence=px.colors.qualitative.Plotly
                                            )
                                            fig_cross_corr_avg.update_layout(title_x=0.5)
                                            st.plotly_chart(fig_cross_corr_avg, use_container_width=True)

                                            max_corr_row_avg = cross_corr_results_avg.loc[cross_corr_results_avg['correlation'].abs().idxmax()]
                                            st.metric(f"Optimal Lag for Max Correlation {analysis_mode_label}", f"{max_corr_row_avg['lag']:.0f} periods")
                                            st.metric(f"Correlation at Optimal Lag {analysis_mode_label}", f"{max_corr_row_avg['correlation']:.3f}")
                                        else:
                                            st.warning("No cross-correlation results could be generated for these parameters in the selected window. See notes above for details.")
                            else:
                                st.info("Select two different parameters to calculate their cross-correlation.")

                            # Scatter plot analysis (use data_to_plot and new param names)
                            if show_scatter_avg_tab and x_axis_param_avg != y_axis_param_avg:
                                st.subheader(f"Scatter Plot Analysis {analysis_mode_label}")

                                scatter_data_avg = data_to_plot[['Date', plot_x_param, plot_y_param]].copy()
                                scatter_data_avg = scatter_data_avg.dropna(subset=[plot_x_param, plot_y_param])

                                if scatter_data_avg.empty:
                                    st.error("No overlapping data found between selected parameters for scatter plot in this window.")
                                else:
                                    col_a, col_b, col_c = st.columns(3)
                                    correlation_avg = scatter_data_avg[plot_x_param].corr(scatter_data_avg[plot_y_param])
                                    total_points_avg = len(scatter_data_avg)

                                    with col_a: st.metric(f"Data Points {analysis_mode_label}", total_points_avg)
                                    with col_b: st.metric(f"Pearson Correlation {analysis_mode_label}", f"{correlation_avg:.3f}")
                                    with col_c: st.metric(f"Data Completeness {analysis_mode_label}", f"{(total_points_avg / len(current_filtered_averaged_data_tab)) * 100:.1f}%")

                                    fig_scatter_avg = px.scatter(
                                        scatter_data_avg,
                                        x=plot_x_param,
                                        y=plot_y_param,
                                        title=f'Scatter Plot: {get_param_info(plot_x_param)} vs {get_param_info(plot_y_param)} {analysis_mode_label}',
                                        template="plotly_white",
                                        hover_data={'Date': True}
                                    )

                                    if len(scatter_data_avg) > 2:
                                        z_avg = np.polyfit(scatter_data_avg[plot_x_param], scatter_data_avg[plot_y_param], 1)
                                        p_avg = np.poly1d(z_avg)
                                        x_trend_avg = np.linspace(scatter_data_avg[plot_x_param].min(), scatter_data_avg[plot_x_param].max(), 100)
                                        y_trend_avg = p_avg(x_trend_avg)
                                        fig_scatter_avg.add_trace(go.Scatter(x=x_trend_avg, y=y_trend_avg, mode='lines', name=f'Linear Trend (r={correlation_avg:.3f})', line=dict(color='red', dash='dash')))

                                    fig_scatter_avg.update_layout(title_x=0.5, height=500, xaxis_title=get_param_info(plot_x_param), yaxis_title=get_param_info(plot_y_param))
                                    st.plotly_chart(fig_scatter_avg, use_container_width=True)

                                    with st.expander("Recent Data Preview"):
                                        st.dataframe(scatter_data_avg.tail(10))
                            elif show_scatter_avg_tab:
                                st.info("Select two different parameters to show a meaningful scatter plot.")
                        else: # If no parameters selected for detailed comparison
                            st.info("Please select two parameters above to perform detailed comparisons (time series, cross-correlation, scatter plot) on averaged data.")

                    # Overall Correlation Matrix for Averaged Data (using current selection)
                    st.markdown("---")
                    st.subheader(f"Pearson, Spearman, or Kendall Correlation Matrix ({selected_averaging_period} Data)")

                    correlation_method_avg = st.selectbox(
                        "Select Method:",
                        ('pearson', 'spearman', 'kendall'),
                        help="Choose correlation method for analysis on averaged data",
                        key="avg_corr_matrix_method" # Unique key
                    )

                    use_clustering_corr_avg = st.checkbox("Apply hierarchical clustering", value=True, key="avg_use_clustering_corr")

                    if len(numeric_cols_averaged) >= 2:
                        corr_matrix_avg = data_for_averaged_analysis[numeric_cols_averaged].corr(method=correlation_method_avg)

                        if use_clustering_corr_avg and not corr_matrix_avg.empty and corr_matrix_avg.shape[0] >= 2:
                            try:
                                optimal_order_avg = hierarchical_clustering_order(corr_matrix_avg)
                                ordered_params_avg = [numeric_cols_averaged[i] for i in optimal_order_avg]
                                corr_matrix_avg = corr_matrix_avg.loc[ordered_params_avg, ordered_params_avg]
                            except Exception as e:
                                st.warning(f"Could not apply clustering: {e}. Ensure enough data variation for clustering.")

                        fig_corr_avg = go.Figure(data=go.Heatmap(
                            z=corr_matrix_avg.values,
                            x=[get_param_info(col) for col in corr_matrix_avg.columns],
                            y=[get_param_info(row) for row in corr_matrix_avg.index],
                            colorscale='RdBu',
                            zmid=0,
                            text=np.round(corr_matrix_avg.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 8}
                        ))
                        fig_corr_avg.update_layout(
                            title=f"Correlation Matrix ({correlation_method_avg.capitalize()}) - {selected_averaging_period} Data",
                            height=600,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_corr_avg, use_container_width=True)
                    else:
                        st.info("Not enough numeric columns to compute a correlation matrix on averaged data.")

                    add_correlation_summaries(corr_matrix_avg, st.session_state.analysis_cache.get('lag_matrix_avg'), suffix=f"({selected_averaging_period} Data)")
                    
                    # Time-Varying Correlation Analysis Section
                    st.markdown("---")
                    st.subheader("📊 Time-Varying Correlation Analysis")
                    st.info("Analyze how correlations between parameters change over time using moving windows or custom time periods. This helps identify temporal patterns, seasonal effects, and structural changes in relationships.")
                    
                    # Create two columns for the two analysis types
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        time_corr_method = st.radio(
                            "Select analysis method:",
                            ["Moving Window", "Custom Time Periods"],
                            key="time_corr_method",
                            help="Moving Window: Analyze correlations in uniform sliding windows\nCustom Time Periods: Define specific date ranges for comparison"
                        )
                    
                    # Parameter selection for time-varying analysis
                    col_tv1, col_tv2 = st.columns(2)
                    with col_tv1:
                        tv_param1 = st.selectbox(
                            "Select first parameter:",
                            options=numeric_cols_averaged,
                            format_func=get_param_info,
                            key="tv_param1"
                        )
                    with col_tv2:
                        tv_param2 = st.selectbox(
                            "Select second parameter:",
                            options=numeric_cols_averaged,
                            index=min(1, len(numeric_cols_averaged)-1) if len(numeric_cols_averaged) > 1 else 0,
                            format_func=get_param_info,
                            key="tv_param2"
                        )
                    
                    if tv_param1 and tv_param2 and tv_param1 != tv_param2:
                        
                        if time_corr_method == "Moving Window":
                            st.subheader("🔄 Moving Window Correlation Analysis")
                            
                            col_mw1, col_mw2, col_mw3 = st.columns(3)
                            with col_mw1:
                                window_size = st.slider(
                                    f"Window size ({selected_averaging_period.replace('Average', 'periods')})",
                                    min_value=5,
                                    max_value=min(50, len(data_for_averaged_analysis)//3),
                                    value=20,
                                    key="tv_window_size",
                                    help="Size of the moving window for correlation calculation"
                                )
                            
                            with col_mw2:
                                step_size = st.slider(
                                    "Step size",
                                    min_value=1,
                                    max_value=window_size//2,
                                    value=max(1, window_size//5),
                                    key="tv_step_size",
                                    help="How many periods to slide the window each step"
                                )
                            
                            with col_mw3:
                                corr_method_tv = st.selectbox(
                                    "Correlation method:",
                                    ["pearson", "spearman", "kendall"],
                                    key="tv_corr_method"
                                )
                            
                            if st.button("Calculate Moving Window Correlations", key="calc_moving_window"):
                                with st.spinner("Calculating time-varying correlations..."):
                                    # Prepare data
                                    tv_data = data_for_averaged_analysis[['Date', tv_param1, tv_param2]].dropna()
                                    
                                    if len(tv_data) < window_size:
                                        st.error(f"Not enough data points ({len(tv_data)}) for the selected window size ({window_size})")
                                    else:
                                        results = []
                                        
                                        # Calculate correlations for each window
                                        for start_idx in range(0, len(tv_data) - window_size + 1, step_size):
                                            end_idx = start_idx + window_size
                                            window_data = tv_data.iloc[start_idx:end_idx]
                                            
                                            # Calculate correlation
                                            corr = window_data[tv_param1].corr(window_data[tv_param2], method=corr_method_tv)
                                            
                                            # Store results
                                            results.append({
                                                'start_date': window_data['Date'].iloc[0],
                                                'end_date': window_data['Date'].iloc[-1],
                                                'center_date': window_data['Date'].iloc[len(window_data)//2],
                                                'correlation': corr,
                                                'n_points': len(window_data),
                                                'param1_mean': window_data[tv_param1].mean(),
                                                'param2_mean': window_data[tv_param2].mean(),
                                                'param1_std': window_data[tv_param1].std(),
                                                'param2_std': window_data[tv_param2].std()
                                            })
                                        
                                        results_df = pd.DataFrame(results)
                                        
                                        # Store in session state
                                        st.session_state['tv_moving_window_results'] = results_df
                                        
                                        # Plot results
                                        fig_tv = go.Figure()
                                        
                                        # Add correlation line
                                        fig_tv.add_trace(go.Scatter(
                                            x=results_df['center_date'],
                                            y=results_df['correlation'],
                                            mode='lines+markers',
                                            name='Correlation',
                                            line=dict(width=3),
                                            marker=dict(size=6),
                                            hovertemplate='Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
                                        ))
                                        
                                        # Add confidence bands (approximate 95% CI for correlation)
                                        ci = 1.96 / np.sqrt(window_size - 3)  # Fisher z approximation
                                        fig_tv.add_trace(go.Scatter(
                                            x=results_df['center_date'],
                                            y=results_df['correlation'] + ci,
                                            mode='lines',
                                            line=dict(width=0),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                        fig_tv.add_trace(go.Scatter(
                                            x=results_df['center_date'],
                                            y=results_df['correlation'] - ci,
                                            mode='lines',
                                            fill='tonexty',
                                            fillcolor='rgba(0,100,80,0.2)',
                                            line=dict(width=0),
                                            name='95% CI',
                                            hoverinfo='skip'
                                        ))
                                        
                                        # Add zero line
                                        fig_tv.add_hline(y=0, line_dash="dash", line_color="gray")
                                        
                                        # Add significance threshold lines
                                        fig_tv.add_hline(y=0.7, line_dash="dot", line_color="green", 
                                                        annotation_text="Strong positive")
                                        fig_tv.add_hline(y=-0.7, line_dash="dot", line_color="red", 
                                                        annotation_text="Strong negative")
                                        
                                        fig_tv.update_layout(
                                            title=f'Time-Varying {corr_method_tv.capitalize()} Correlation: {get_param_info(tv_param1)} vs {get_param_info(tv_param2)}',
                                            xaxis_title='Time',
                                            yaxis_title='Correlation Coefficient',
                                            yaxis_range=[-1.1, 1.1],
                                            template="plotly_white",
                                            height=600,
                                            hovermode='x unified'
                                        )
                                        
                                        st.plotly_chart(fig_tv, use_container_width=True)
                                        
                                        # Summary statistics
                                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                                        with col_sum1:
                                            st.metric("Mean Correlation", f"{results_df['correlation'].mean():.3f}")
                                        with col_sum2:
                                            st.metric("Std Dev", f"{results_df['correlation'].std():.3f}")
                                        with col_sum3:
                                            st.metric("Min", f"{results_df['correlation'].min():.3f}")
                                        with col_sum4:
                                            st.metric("Max", f"{results_df['correlation'].max():.3f}")
                                        
                                        # Identify periods of interest
                                        st.subheader("🔍 Notable Periods")
                                        
                                        # Strongest correlations
                                        strongest_pos = results_df.nlargest(3, 'correlation')
                                        strongest_neg = results_df.nsmallest(3, 'correlation')
                                        
                                        col_period1, col_period2 = st.columns(2)
                                        with col_period1:
                                            st.write("**Strongest Positive Correlations:**")
                                            for _, row in strongest_pos.iterrows():
                                                st.write(f"- {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}: r = {row['correlation']:.3f}")
                                        
                                        with col_period2:
                                            st.write("**Strongest Negative Correlations:**")
                                            for _, row in strongest_neg.iterrows():
                                                if row['correlation'] < 0:
                                                    st.write(f"- {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}: r = {row['correlation']:.3f}")
                                        
                                        # Download results
                                        csv = results_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Time-Varying Correlation Results",
                                            data=csv,
                                            file_name=f"time_varying_correlation_{tv_param1}_{tv_param2}.csv",
                                            mime="text/csv"
                                        )
                        
                        else:  # Custom Time Periods
                            st.subheader("📅 Custom Time Period Correlation Analysis")
                            
                            # Number of periods
                            n_periods = st.number_input(
                                "Number of time periods to define:",
                                min_value=2,
                                max_value=10,
                                value=3,
                                key="tv_n_periods"
                            )
                            
                            # Define periods
                            st.write("Define your time periods:")
                            periods = []
                            
                            for i in range(n_periods):
                                col_start, col_end, col_label = st.columns([2, 2, 1])
                                
                                with col_start:
                                    start_date = st.date_input(
                                        f"Start date {i+1}:",
                                        value=data_for_averaged_analysis['Date'].min() + pd.Timedelta(days=i*365),
                                        min_value=data_for_averaged_analysis['Date'].min().date(),
                                        max_value=data_for_averaged_analysis['Date'].max().date(),
                                        key=f"tv_period_start_{i}"
                                    )
                                
                                with col_end:
                                    end_date = st.date_input(
                                        f"End date {i+1}:",
                                        value=min(start_date + pd.Timedelta(days=365), 
                                                 data_for_averaged_analysis['Date'].max().date()),
                                        min_value=start_date,
                                        max_value=data_for_averaged_analysis['Date'].max().date(),
                                        key=f"tv_period_end_{i}"
                                    )
                                
                                with col_label:
                                    label = st.text_input(
                                        f"Label {i+1}:",
                                        value=f"Period {i+1}",
                                        key=f"tv_period_label_{i}"
                                    )
                                
                                periods.append({
                                    'start': pd.to_datetime(start_date),
                                    'end': pd.to_datetime(end_date),
                                    'label': label
                                })
                            
                            # Correlation method selection
                            corr_method_custom = st.selectbox(
                                "Correlation method:",
                                ["pearson", "spearman", "kendall"],
                                key="tv_custom_corr_method"
                            )
                            
                            if st.button("Calculate Period Correlations", key="calc_period_corr"):
                                with st.spinner("Calculating correlations for custom periods..."):
                                    period_results = []
                                    
                                    for period in periods:
                                        # Filter data for this period
                                        mask = (data_for_averaged_analysis['Date'] >= period['start']) & \
                                               (data_for_averaged_analysis['Date'] <= period['end'])
                                        period_data = data_for_averaged_analysis[mask][[tv_param1, tv_param2]].dropna()
                                        
                                        if len(period_data) >= 3:  # Minimum for correlation
                                            # Calculate correlation
                                            corr = period_data[tv_param1].corr(period_data[tv_param2], method=corr_method_custom)
                                            
                                            # Calculate confidence interval
                                            n = len(period_data)
                                            if n > 3:
                                                # Fisher z transformation
                                                z = 0.5 * np.log((1 + corr) / (1 - corr))
                                                se = 1 / np.sqrt(n - 3)
                                                ci_lower = np.tanh(z - 1.96 * se)
                                                ci_upper = np.tanh(z + 1.96 * se)
                                            else:
                                                ci_lower, ci_upper = np.nan, np.nan
                                            
                                            # Calculate p-value (approximate)
                                            if corr_method_custom == "pearson":
                                                t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
                                                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                                            else:
                                                p_value = np.nan  # Would need specific implementations for spearman/kendall
                                            
                                            period_results.append({
                                                'Period': period['label'],
                                                'Start': period['start'],
                                                'End': period['end'],
                                                'Correlation': corr,
                                                'CI_Lower': ci_lower,
                                                'CI_Upper': ci_upper,
                                                'N': n,
                                                'P_value': p_value,
                                                'Significant': p_value < 0.05 if not np.isnan(p_value) else False
                                            })
                                        else:
                                            period_results.append({
                                                'Period': period['label'],
                                                'Start': period['start'],
                                                'End': period['end'],
                                                'Correlation': np.nan,
                                                'CI_Lower': np.nan,
                                                'CI_Upper': np.nan,
                                                'N': len(period_data),
                                                'P_value': np.nan,
                                                'Significant': False
                                            })
                                    
                                    results_df = pd.DataFrame(period_results)
                                    
                                    # Visualization
                                    fig_periods = go.Figure()
                                    
                                    # Add bars for correlations
                                    colors = ['green' if sig else 'gray' for sig in results_df['Significant']]
                                    
                                    fig_periods.add_trace(go.Bar(
                                        x=results_df['Period'],
                                        y=results_df['Correlation'],
                                        error_y=dict(
                                            type='data',
                                            symmetric=False,
                                            array=results_df['CI_Upper'] - results_df['Correlation'],
                                            arrayminus=results_df['Correlation'] - results_df['CI_Lower']
                                        ),
                                        marker_color=colors,
                                        text=[f"n={n}" for n in results_df['N']],
                                        textposition='outside'
                                    ))
                                    
                                    fig_periods.add_hline(y=0, line_dash="dash", line_color="black")
                                    
                                    fig_periods.update_layout(
                                        title=f'{corr_method_custom.capitalize()} Correlation by Time Period: {get_param_info(tv_param1)} vs {get_param_info(tv_param2)}',
                                        xaxis_title='Time Period',
                                        yaxis_title='Correlation Coefficient',
                                        yaxis_range=[-1.1, 1.1],
                                        template="plotly_white",
                                        height=500,
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_periods, use_container_width=True)
                                    
                                    # Results table
                                    st.subheader("📊 Detailed Results")
                                    
                                    # Format the dataframe for display
                                    display_df = results_df.copy()
                                    display_df['Period Info'] = display_df.apply(
                                        lambda row: f"{row['Start'].strftime('%Y-%m-%d')} to {row['End'].strftime('%Y-%m-%d')}",
                                        axis=1
                                    )
                                    display_df['Correlation (95% CI)'] = display_df.apply(
                                        lambda row: f"{row['Correlation']:.3f} ({row['CI_Lower']:.3f}, {row['CI_Upper']:.3f})" 
                                        if not np.isnan(row['Correlation']) else "N/A",
                                        axis=1
                                    )
                                    display_df['P-value'] = display_df['P_value'].apply(
                                        lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"
                                    )
                                    
                                    # Select columns to display
                                    display_cols = ['Period', 'Period Info', 'Correlation (95% CI)', 'N', 'P-value', 'Significant']
                                    st.dataframe(
                                        display_df[display_cols].style.applymap(
                                            lambda x: 'background-color: lightgreen' if x == True else 'background-color: lightcoral' if x == False else '',
                                            subset=['Significant']
                                        ),
                                        use_container_width=True
                                    )
                                    
                                    # Test for significant differences between periods
                                    if len(results_df) > 1 and results_df['Correlation'].notna().sum() > 1:
                                        st.subheader("🔬 Statistical Comparison Between Periods")
                                        
                                        # Perform pairwise comparisons using Fisher's z test
                                        comparisons = []
                                        for i in range(len(results_df)):
                                            for j in range(i+1, len(results_df)):
                                                if not np.isnan(results_df.iloc[i]['Correlation']) and \
                                                   not np.isnan(results_df.iloc[j]['Correlation']):
                                                    r1, n1 = results_df.iloc[i]['Correlation'], results_df.iloc[i]['N']
                                                    r2, n2 = results_df.iloc[j]['Correlation'], results_df.iloc[j]['N']
                                                    
                                                    # Fisher z transformation
                                                    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
                                                    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
                                                    
                                                    # Standard error of difference
                                                    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
                                                    
                                                    # Z statistic
                                                    z_stat = (z1 - z2) / se_diff
                                                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                                                    
                                                    comparisons.append({
                                                        'Comparison': f"{results_df.iloc[i]['Period']} vs {results_df.iloc[j]['Period']}",
                                                        'Difference': r1 - r2,
                                                        'Z-statistic': z_stat,
                                                        'P-value': p_value,
                                                        'Significant': p_value < 0.05
                                                    })
                                        
                                        if comparisons:
                                            comp_df = pd.DataFrame(comparisons)
                                            st.dataframe(comp_df, use_container_width=True)
                                            
                                            # Interpretation
                                            sig_diffs = comp_df[comp_df['Significant'] == True]
                                            if len(sig_diffs) > 0:
                                                st.success(f"✅ Found {len(sig_diffs)} significant difference(s) in correlation between periods")
                                            else:
                                                st.info("ℹ️ No significant differences in correlation found between periods")
                    
                    # Additional advanced time-varying analysis options
                    with st.expander("🔬 Advanced Time-Varying Analysis Options"):
                        st.subheader("Additional Statistical Approaches")
                        
                        st.markdown("""
                        ### 1. **Change Point Detection**
                        Automatically identify points in time where the correlation structure changes significantly.
                        
                        ### 2. **Seasonal Decomposition of Correlations**
                        Separate the correlation time series into trend, seasonal, and residual components.
                        
                        ### 3. **Wavelet Coherence Analysis**
                        Analyze correlation patterns at different time scales simultaneously.
                        
                        ### 4. **State-Space Models**
                        Model time-varying correlations using dynamic linear models.
                        
                        ### 5. **GARCH-DCC Models**
                        For volatility and dynamic conditional correlation modeling.
                        """)
                        
                        # Implement change point detection
                        if st.checkbox("Enable Change Point Detection", key="enable_cpd"):
                            if 'tv_moving_window_results' in st.session_state and not st.session_state['tv_moving_window_results'].empty:
                                st.subheader("🎯 Change Point Detection in Correlations")
                                
                                # Simple change point detection using cumulative sum
                                results_df = st.session_state['tv_moving_window_results']
                                correlations = results_df['correlation'].values
                                
                                # Calculate CUSUM
                                mean_corr = np.mean(correlations)
                                cusum = np.cumsum(correlations - mean_corr)
                                
                                # Find change point (maximum deviation from mean)
                                change_point_idx = np.argmax(np.abs(cusum))
                                change_point_date = results_df.iloc[change_point_idx]['center_date']
                                
                                # Plot CUSUM
                                fig_cusum = go.Figure()
                                fig_cusum.add_trace(go.Scatter(
                                    x=results_df['center_date'],
                                    y=cusum,
                                    mode='lines',
                                    name='CUSUM'
                                ))
                                fig_cusum.add_vline(
                                    x=change_point_date,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Change point: {change_point_date.strftime('%Y-%m-%d')}"
                                )
                                fig_cusum.update_layout(
                                    title="Cumulative Sum (CUSUM) Chart for Change Point Detection",
                                    xaxis_title="Date",
                                    yaxis_title="CUSUM",
                                    template="plotly_white",
                                    height=400
                                )
                                st.plotly_chart(fig_cusum, use_container_width=True)
                                
                                # Statistics before and after change point
                                before_mask = results_df['center_date'] < change_point_date
                                after_mask = ~before_mask
                                
                                col_cp1, col_cp2 = st.columns(2)
                                with col_cp1:
                                    st.metric(
                                        "Mean Correlation Before",
                                        f"{results_df[before_mask]['correlation'].mean():.3f}",
                                        f"n = {before_mask.sum()}"
                                    )
                                with col_cp2:
                                    st.metric(
                                        "Mean Correlation After",
                                        f"{results_df[after_mask]['correlation'].mean():.3f}",
                                        f"n = {after_mask.sum()}"
                                    )
                                
                                # Test for significance
                                if before_mask.sum() > 1 and after_mask.sum() > 1:
                                    t_stat, p_value = stats.ttest_ind(
                                        results_df[before_mask]['correlation'],
                                        results_df[after_mask]['correlation']
                                    )
                                    if p_value < 0.05:
                                        st.success(f"✅ Significant change detected (p = {p_value:.4f})")
                                    else:
                                        st.info(f"ℹ️ No significant change detected (p = {p_value:.4f})")
        else:
            st.error("❌ Failed to merge data or no data available for averaged correlation analysis.")
            st.info("Please check that your data files contain proper date columns and numeric data.")

    # --- NEW Non-Linear Correlations Tab ---
    with tabs[7]:  # This is the enhanced non-linear tab
        st.header("🔀 Non-Linear Correlations & Advanced Analysis")
        st.info("This tab explores non-linear relationships between parameters using advanced statistical methods including mutual information, polynomial regression, GAM smoothing, and more.")

        merged_data = create_properly_merged_dataframe(available_data)
        
        if merged_data is not None and not merged_data.empty:
            # Averaging Period Selection for Non-Linear Tab
            st.subheader("Data Configuration for Non-Linear Analysis")
            col_nl_avg_1, col_nl_avg_2, col_nl_avg_3 = st.columns([1, 1, 1])
            with col_nl_avg_1:
                selected_averaging_period_nl = st.selectbox(
                    "Select Data Averaging Period:",
                    options=['Raw Data', '3-Day Average', '7-Day Average', '15-Day Average', '30-Day Average'],
                    index=2,  # Default to 7-Day Average
                    key="nonlinear_analysis_avg_period",
                    help="Choose an averaging period to smooth data for non-linear analysis."
                )
            
            with col_nl_avg_2:
                min_data_points = st.number_input(
                    "Minimum data points required:",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    key="nl_min_data_points",
                    help="Minimum number of valid data points required for analysis"
                )
            
            with col_nl_avg_3:
                confidence_level = st.slider(
                    "Confidence Level:",
                    min_value=0.90,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    key="nl_confidence_level",
                    help="Confidence level for statistical tests"
                )
            
            # Process data based on selection
            if selected_averaging_period_nl == 'Raw Data':
                data_for_nonlinear_analysis = merged_data.copy()
                time_unit = "days"
            else:
                resample_freq_map_nl = {
                    '3-Day Average': '3D',
                    '7-Day Average': '7D',
                    '15-Day Average': '15D',
                    '30-Day Average': '30D'
                }
                resample_freq_nl = resample_freq_map_nl.get(selected_averaging_period_nl, '7D')
                time_unit = selected_averaging_period_nl.lower().replace(' average', '')
                
                data_for_nonlinear_analysis = merged_data.copy()
                data_for_nonlinear_analysis = data_for_nonlinear_analysis.set_index('Date')
                data_for_nonlinear_analysis = data_for_nonlinear_analysis.resample(resample_freq_nl).mean(numeric_only=True).reset_index()
                data_for_nonlinear_analysis['Date'] = pd.to_datetime(data_for_nonlinear_analysis['Date'])
            
            st.info(f"Analysis will use {selected_averaging_period_nl}. Total records: {len(data_for_nonlinear_analysis)}.")

            numeric_cols_nl = [col for col in data_for_nonlinear_analysis.columns 
                               if col != 'Date' and pd.api.types.is_numeric_dtype(data_for_nonlinear_analysis[col]) 
                               and data_for_nonlinear_analysis[col].notna().any()]

            if not numeric_cols_nl:
                st.warning("No numeric data available for non-linear correlation analysis.")
            else:
                # Create sub-tabs for different non-linear analyses
                nl_tabs = st.tabs(["📊 Overview", "🔍 Mutual Information", "📈 Non-Linear Regression", 
                                   "🌊 Phase-Space Analysis", "🎯 Causality Testing"])
                
                # Overview Tab
                with nl_tabs[0]:
                    st.subheader("Non-Linear Relationships Overview")
                    
                    # Quick comparison of linear vs non-linear correlations
                    col1, col2 = st.columns(2)
                    with col1:
                        param1_overview = st.selectbox(
                            "Select first parameter:",
                            options=numeric_cols_nl,
                            format_func=get_param_info,
                            key="nl_overview_param1"
                        )
                    with col2:
                        param2_overview = st.selectbox(
                            "Select second parameter:",
                            options=numeric_cols_nl,
                            index=min(1, len(numeric_cols_nl)-1) if len(numeric_cols_nl) > 1 else 0,
                            format_func=get_param_info,
                            key="nl_overview_param2"
                        )
                    
                    if param1_overview and param2_overview and param1_overview != param2_overview:
                        # Prepare data
                        overview_data = data_for_nonlinear_analysis[[param1_overview, param2_overview]].dropna()
                        
                        if len(overview_data) >= min_data_points:
                            # Calculate various correlation metrics
                            pearson_r = overview_data[param1_overview].corr(overview_data[param2_overview])
                            spearman_r = overview_data[param1_overview].corr(overview_data[param2_overview], method='spearman')
                            kendall_tau = overview_data[param1_overview].corr(overview_data[param2_overview], method='kendall')
                            
                            # Distance correlation
                            try:
                                dcorr = pg.distance_corr(overview_data[param1_overview], overview_data[param2_overview])[0]
                            except:
                                dcorr = np.nan
                            
                            # Display metrics
                            st.subheader("Correlation Metrics Comparison")
                            metrics_cols = st.columns(4)
                            with metrics_cols[0]:
                                st.metric("Pearson (Linear)", f"{pearson_r:.3f}")
                            with metrics_cols[1]:
                                st.metric("Spearman (Rank)", f"{spearman_r:.3f}")
                            with metrics_cols[2]:
                                st.metric("Kendall (Rank)", f"{kendall_tau:.3f}")
                            with metrics_cols[3]:
                                st.metric("Distance (Non-linear)", f"{dcorr:.3f}" if not np.isnan(dcorr) else "N/A")
                            
                            # Visualization with multiple regression types
                            st.subheader("Relationship Visualization")
                            
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=("Linear Regression", "Polynomial Regression", 
                                              "LOWESS Smoothing", "Binned Averages"),
                                vertical_spacing=0.12,
                                horizontal_spacing=0.1
                            )
                            
                            x_data = overview_data[param1_overview].values
                            y_data = overview_data[param2_overview].values
                            
                            # 1. Linear regression
                            fig.add_trace(
                                go.Scatter(x=x_data, y=y_data, mode='markers', 
                                         name='Data', showlegend=False,
                                         marker=dict(size=5, color='blue', opacity=0.5)),
                                row=1, col=1
                            )
                            z = np.polyfit(x_data, y_data, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_data.min(), x_data.max(), 100)
                            fig.add_trace(
                                go.Scatter(x=x_line, y=p(x_line), mode='lines',
                                         name='Linear fit', line=dict(color='red')),
                                row=1, col=1
                            )
                            
                            # 2. Polynomial regression (degree 3)
                            fig.add_trace(
                                go.Scatter(x=x_data, y=y_data, mode='markers',
                                         showlegend=False,
                                         marker=dict(size=5, color='blue', opacity=0.5)),
                                row=1, col=2
                            )
                            z_poly = np.polyfit(x_data, y_data, 3)
                            p_poly = np.poly1d(z_poly)
                            fig.add_trace(
                                go.Scatter(x=x_line, y=p_poly(x_line), mode='lines',
                                         name='Polynomial fit', line=dict(color='green')),
                                row=1, col=2
                            )
                            
                            # 3. LOWESS smoothing
                            fig.add_trace(
                                go.Scatter(x=x_data, y=y_data, mode='markers',
                                         showlegend=False,
                                         marker=dict(size=5, color='blue', opacity=0.5)),
                                row=2, col=1
                            )
                            try:
                                from statsmodels.nonparametric.smoothers_lowess import lowess
                                lowess_result = lowess(y_data, x_data, frac=0.3)
                                fig.add_trace(
                                    go.Scatter(x=lowess_result[:, 0], y=lowess_result[:, 1],
                                             mode='lines', name='LOWESS',
                                             line=dict(color='orange')),
                                    row=2, col=1
                                )
                            except:
                                st.warning("LOWESS smoothing unavailable (install statsmodels)")
                            
                            # 4. Binned averages
                            fig.add_trace(
                                go.Scatter(x=x_data, y=y_data, mode='markers',
                                         showlegend=False,
                                         marker=dict(size=5, color='blue', opacity=0.3)),
                                row=2, col=2
                            )
                            # Create bins
                            n_bins = min(20, len(overview_data) // 5)
                            bins = np.linspace(x_data.min(), x_data.max(), n_bins)
                            bin_indices = np.digitize(x_data, bins)
                            bin_means_x = []
                            bin_means_y = []
                            bin_std_y = []
                            for i in range(1, n_bins):
                                mask = bin_indices == i
                                if mask.sum() > 0:
                                    bin_means_x.append(x_data[mask].mean())
                                    bin_means_y.append(y_data[mask].mean())
                                    bin_std_y.append(y_data[mask].std())
                            
                            fig.add_trace(
                                go.Scatter(x=bin_means_x, y=bin_means_y,
                                         mode='lines+markers',
                                         name='Binned averages',
                                         line=dict(color='purple'),
                                         error_y=dict(type='data', array=bin_std_y, visible=True)),
                                row=2, col=2
                            )
                            
                            # Update layout
                            fig.update_xaxes(title_text=get_param_info(param1_overview), row=2, col=1)
                            fig.update_xaxes(title_text=get_param_info(param1_overview), row=2, col=2)
                            fig.update_yaxes(title_text=get_param_info(param2_overview), row=1, col=1)
                            fig.update_yaxes(title_text=get_param_info(param2_overview), row=2, col=1)
                            
                            fig.update_layout(
                                height=800,
                                title_text=f"Multiple Regression Approaches: {get_param_info(param1_overview)} vs {get_param_info(param2_overview)}",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Residual analysis
                            st.subheader("Residual Analysis")
                            st.info("Comparing residuals from linear vs polynomial fits can reveal non-linear patterns.")
                            
                            linear_residuals = y_data - p(x_data)
                            poly_residuals = y_data - p_poly(x_data)
                            
                            fig_resid = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=("Linear Model Residuals", "Polynomial Model Residuals")
                            )
                            
                            fig_resid.add_trace(
                                go.Scatter(x=x_data, y=linear_residuals, mode='markers',
                                         marker=dict(size=5, color='red', opacity=0.5)),
                                row=1, col=1
                            )
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
                            
                            fig_resid.add_trace(
                                go.Scatter(x=x_data, y=poly_residuals, mode='markers',
                                         marker=dict(size=5, color='green', opacity=0.5)),
                                row=1, col=2
                            )
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
                            
                            fig_resid.update_xaxes(title_text=get_param_info(param1_overview))
                            fig_resid.update_yaxes(title_text="Residuals", row=1, col=1)
                            fig_resid.update_layout(height=400, showlegend=False)
                            
                            st.plotly_chart(fig_resid, use_container_width=True)
                            
                            # Statistical tests
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.metric("Linear R²", f"{pearson_r**2:.3f}")
                                st.metric("Linear Residual Std", f"{linear_residuals.std():.3f}")
                            with col_stats2:
                                poly_r2 = 1 - (np.sum(poly_residuals**2) / np.sum((y_data - y_data.mean())**2))
                                st.metric("Polynomial R²", f"{poly_r2:.3f}")
                                st.metric("Polynomial Residual Std", f"{poly_residuals.std():.3f}")
                            
                            # Test for non-linearity
                            if poly_r2 - pearson_r**2 > 0.05:
                                st.success(f"🎯 Polynomial model shows significant improvement (ΔR² = {poly_r2 - pearson_r**2:.3f}), suggesting non-linear relationship.")
                            else:
                                st.info("📊 Linear model appears adequate for this relationship.")
                        
                        else:
                            st.warning(f"Need at least {min_data_points} data points for analysis. Current: {len(overview_data)}")
                    else:
                        st.info("Select two different parameters to analyze their relationship.")
                
                # Mutual Information Tab
                with nl_tabs[1]:
                    st.subheader("Mutual Information Analysis")
                    st.info("Mutual Information (MI) quantifies the amount of information one variable contains about another, capturing both linear and non-linear dependencies.")
                    
                    # Options for MI calculation
                    col_mi_1, col_mi_2 = st.columns(2)
                    with col_mi_1:
                        mi_method = st.selectbox(
                            "MI Estimation Method:",
                            ["histogram", "k-nearest neighbors"],
                            key="mi_method",
                            help="Histogram is faster but less accurate for continuous data. KNN is more accurate but slower."
                        )
                    with col_mi_2:
                        n_bins_mi = st.slider(
                            "Number of bins (for histogram method):",
                            min_value=5,
                            max_value=50,
                            value=15,
                            key="mi_n_bins"
                        )
                    
                    # Calculate full MI matrix
                    if st.button("Calculate Complete MI Matrix", key="calc_mi_matrix_full"):
                        with st.spinner("Calculating Mutual Information matrix..."):
                            if len(numeric_cols_nl) >= 2:
                                # Custom MI calculation with proper handling
                                from sklearn.feature_selection import mutual_info_regression
                                from sklearn.metrics import mutual_info_score
                                
                                mi_matrix = pd.DataFrame(index=numeric_cols_nl, columns=numeric_cols_nl, dtype=float)
                                
                                for i, col1 in enumerate(numeric_cols_nl):
                                    for j, col2 in enumerate(numeric_cols_nl):
                                        if i == j:
                                            mi_matrix.loc[col1, col2] = 1.0  # Perfect information with itself
                                        else:
                                            # Get clean data for this pair
                                            pair_data = data_for_nonlinear_analysis[[col1, col2]].dropna()
                                            if len(pair_data) < min_data_points:
                                                mi_matrix.loc[col1, col2] = np.nan
                                            else:
                                                if mi_method == "histogram":
                                                    # Discretize data for histogram method
                                                    col1_binned = pd.cut(pair_data[col1], bins=n_bins_mi, labels=False)
                                                    col2_binned = pd.cut(pair_data[col2], bins=n_bins_mi, labels=False)
                                                    mi_value = mutual_info_score(col1_binned, col2_binned)
                                                    # Normalize by the minimum entropy
                                                    h1 = -np.sum((np.bincount(col1_binned) / len(col1_binned)) * np.log(np.bincount(col1_binned) / len(col1_binned) + 1e-10))
                                                    h2 = -np.sum((np.bincount(col2_binned) / len(col2_binned)) * np.log(np.bincount(col2_binned) / len(col2_binned) + 1e-10))
                                                    mi_matrix.loc[col1, col2] = mi_value / min(h1, h2)
                                                else:
                                                    # KNN method
                                                    mi_value = mutual_info_regression(
                                                        pair_data[[col1]].values,
                                                        pair_data[col2].values,
                                                        n_neighbors=5,
                                                        random_state=42
                                                    )[0]
                                                    mi_matrix.loc[col1, col2] = mi_value
                                
                                st.session_state.analysis_cache['mi_matrix_full'] = mi_matrix
                            else:
                                st.warning("Not enough numeric columns for MI matrix.")
                    
                    # Display MI matrix if available
                    if 'mi_matrix_full' in st.session_state.analysis_cache:
                        mi_matrix = st.session_state.analysis_cache['mi_matrix_full']
                        
                        # Apply clustering
                        use_clustering_mi = st.checkbox("Apply hierarchical clustering", value=True, key="use_clustering_mi")
                        if use_clustering_mi and mi_matrix.shape[0] >= 2:
                            try:
                                optimal_order_mi = hierarchical_clustering_order(mi_matrix.fillna(0))
                                ordered_params_mi = [mi_matrix.columns[i] for i in optimal_order_mi]
                                mi_matrix = mi_matrix.loc[ordered_params_mi, ordered_params_mi]
                            except:
                                pass
                        
                        # Plot heatmap
                        fig_mi = go.Figure(data=go.Heatmap(
                            z=mi_matrix.values,
                            x=[get_param_info(col) for col in mi_matrix.columns],
                            y=[get_param_info(row) for row in mi_matrix.index],
                            colorscale='Viridis',
                            zmin=0,
                            text=np.round(mi_matrix.values, 2),
                            texttemplate='%{text:.2f}',
                            textfont={"size": 8},
                            hovertemplate='X: %{x}<br>Y: %{y}<br>MI: %{z:.3f}<extra></extra>'
                        ))
                        fig_mi.update_layout(
                            title=f"Mutual Information Matrix ({mi_method} method)",
                            height=600,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_mi, use_container_width=True)
                        
                        # Show top MI pairs
                        st.subheader("Strongest Information Dependencies")
                        mi_pairs = []
                        for i in range(len(mi_matrix.columns)):
                            for j in range(i+1, len(mi_matrix.columns)):
                                mi_val = mi_matrix.iloc[i, j]
                                if not pd.isna(mi_val):
                                    mi_pairs.append({
                                        'Parameter 1': get_param_info(mi_matrix.columns[i]),
                                        'Parameter 2': get_param_info(mi_matrix.columns[j]),
                                        'Mutual Information': mi_val
                                    })
                        
                        if mi_pairs:
                            mi_df = pd.DataFrame(mi_pairs).sort_values('Mutual Information', ascending=False).head(10)
                            st.dataframe(mi_df, use_container_width=True)
                    
                    # Detailed MI analysis for selected pair
                    st.markdown("---")
                    st.subheader("Detailed MI Analysis for Selected Parameters")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        mi_param1 = st.selectbox(
                            "Select first parameter:",
                            options=numeric_cols_nl,
                            format_func=get_param_info,
                            key="mi_detail_param1"
                        )
                    with col2:
                        mi_param2 = st.selectbox(
                            "Select second parameter:",
                            options=numeric_cols_nl,
                            index=min(1, len(numeric_cols_nl)-1) if len(numeric_cols_nl) > 1 else 0,
                            format_func=get_param_info,
                            key="mi_detail_param2"
                        )
                    
                    if mi_param1 and mi_param2 and mi_param1 != mi_param2:
                        if st.button("Analyze MI Dependency", key="analyze_mi_detail"):
                            pair_data = data_for_nonlinear_analysis[[mi_param1, mi_param2]].dropna()
                            
                            if len(pair_data) >= min_data_points:
                                # Calculate MI with different bin sizes
                                bin_sizes = range(5, min(50, len(pair_data)//5), 5)
                                mi_values = []
                                
                                for bins in bin_sizes:
                                    col1_binned = pd.cut(pair_data[mi_param1], bins=bins, labels=False)
                                    col2_binned = pd.cut(pair_data[mi_param2], bins=bins, labels=False)
                                    mi = mutual_info_score(col1_binned, col2_binned)
                                    mi_values.append(mi)
                                
                                # Plot MI vs bin size
                                fig_mi_bins = go.Figure()
                                fig_mi_bins.add_trace(go.Scatter(
                                    x=list(bin_sizes),
                                    y=mi_values,
                                    mode='lines+markers',
                                    name='MI vs Bin Size'
                                ))
                                fig_mi_bins.update_layout(
                                    title="Mutual Information vs Discretization Resolution",
                                    xaxis_title="Number of Bins",
                                    yaxis_title="Mutual Information",
                                    height=400
                                )
                                st.plotly_chart(fig_mi_bins, use_container_width=True)
                                
                                # 2D histogram showing information density
                                fig_2dhist = go.Figure(data=go.Histogram2d(
                                    x=pair_data[mi_param1],
                                    y=pair_data[mi_param2],
                                    nbinsx=n_bins_mi,
                                    nbinsy=n_bins_mi,
                                    colorscale='Viridis'
                                ))
                                fig_2dhist.update_layout(
                                    title=f"Joint Distribution: {get_param_info(mi_param1)} vs {get_param_info(mi_param2)}",
                                    xaxis_title=get_param_info(mi_param1),
                                    yaxis_title=get_param_info(mi_param2),
                                    height=500
                                )
                                st.plotly_chart(fig_2dhist, use_container_width=True)
                            else:
                                st.warning(f"Need at least {min_data_points} data points.")
                
                # Non-Linear Regression Tab
                with nl_tabs[2]:
                    st.subheader("Non-Linear Regression Analysis")
                    st.info("Compare different regression models to identify the best fit for your data.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        reg_x = st.selectbox(
                            "Independent variable (X):",
                            options=numeric_cols_nl,
                            format_func=get_param_info,
                            key="reg_x"
                        )
                    with col2:
                        reg_y = st.selectbox(
                            "Dependent variable (Y):",
                            options=numeric_cols_nl,
                            index=min(1, len(numeric_cols_nl)-1) if len(numeric_cols_nl) > 1 else 0,
                            format_func=get_param_info,
                            key="reg_y"
                        )
                    
                    if reg_x and reg_y and reg_x != reg_y:
                        # Model selection
                        models_to_fit = st.multiselect(
                            "Select models to compare:",
                            ["Linear", "Polynomial (2nd)", "Polynomial (3rd)", "Polynomial (4th)",
                             "Logarithmic", "Exponential", "Power", "Sigmoid"],
                            default=["Linear", "Polynomial (2nd)", "Polynomial (3rd)"],
                            key="models_to_fit"
                        )
                        
                        if st.button("Fit Models", key="fit_models_button"):
                            reg_data = data_for_nonlinear_analysis[[reg_x, reg_y]].dropna()
                            
                            if len(reg_data) >= min_data_points:
                                x = reg_data[reg_x].values
                                y = reg_data[reg_y].values
                                x_plot = np.linspace(x.min(), x.max(), 200)
                                
                                # Store model results
                                model_results = []
                                
                                # Fit each model
                                fig_models = go.Figure()
                                
                                # Add scatter plot of data
                                fig_models.add_trace(go.Scatter(
                                    x=x, y=y,
                                    mode='markers',
                                    name='Data',
                                    marker=dict(size=5, opacity=0.5)
                                ))
                                
                                # Linear model
                                if "Linear" in models_to_fit:
                                    coef = np.polyfit(x, y, 1)
                                    y_pred = np.polyval(coef, x)
                                    y_plot = np.polyval(coef, x_plot)
                                    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
                                    aic = len(y) * np.log(np.sum((y - y_pred)**2) / len(y)) + 2 * 2
                                    
                                    fig_models.add_trace(go.Scatter(
                                        x=x_plot, y=y_plot,
                                        mode='lines',
                                        name=f'Linear (R²={r2:.3f})'
                                    ))
                                    
                                    model_results.append({
                                        'Model': 'Linear',
                                        'R²': r2,
                                        'AIC': aic,
                                        'Parameters': 2,
                                        'Equation': f'y = {coef[0]:.3f}x + {coef[1]:.3f}'
                                    })
                                
                                # Polynomial models
                                for degree in [2, 3, 4]:
                                    model_name = f"Polynomial ({degree}{'nd' if degree == 2 else 'rd' if degree == 3 else 'th'})"
                                    if model_name in models_to_fit:
                                        coef = np.polyfit(x, y, degree)
                                        y_pred = np.polyval(coef, x)
                                        y_plot = np.polyval(coef, x_plot)
                                        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
                                        aic = len(y) * np.log(np.sum((y - y_pred)**2) / len(y)) + 2 * (degree + 1)
                                        
                                        fig_models.add_trace(go.Scatter(
                                            x=x_plot, y=y_plot,
                                            mode='lines',
                                            name=f'Poly {degree} (R²={r2:.3f})'
                                        ))
                                        
                                        model_results.append({
                                            'Model': f'Polynomial (degree {degree})',
                                            'R²': r2,
                                            'AIC': aic,
                                            'Parameters': degree + 1,
                                            'Equation': 'y = ' + ' + '.join([f'{c:.3e}x^{degree-i}' for i, c in enumerate(coef)])
                                        })
                                
                                # Logarithmic model (y = a*ln(x) + b)
                                if "Logarithmic" in models_to_fit and (x > 0).all():
                                    try:
                                        log_x = np.log(x)
                                        coef = np.polyfit(log_x, y, 1)
                                        y_pred = np.polyval(coef, log_x)
                                        y_plot = np.polyval(coef, np.log(x_plot[x_plot > 0]))
                                        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
                                        aic = len(y) * np.log(np.sum((y - y_pred)**2) / len(y)) + 2 * 2
                                        
                                        fig_models.add_trace(go.Scatter(
                                            x=x_plot[x_plot > 0], y=y_plot,
                                            mode='lines',
                                            name=f'Logarithmic (R²={r2:.3f})'
                                        ))
                                        
                                        model_results.append({
                                            'Model': 'Logarithmic',
                                            'R²': r2,
                                            'AIC': aic,
                                            'Parameters': 2,
                                            'Equation': f'y = {coef[0]:.3f}ln(x) + {coef[1]:.3f}'
                                        })
                                    except:
                                        st.warning("Logarithmic model failed (requires positive X values)")
                                
                                # Exponential model (y = a*exp(b*x))
                                if "Exponential" in models_to_fit and (y > 0).all():
                                    try:
                                        log_y = np.log(y)
                                        coef = np.polyfit(x, log_y, 1)
                                        y_pred = np.exp(np.polyval(coef, x))
                                        y_plot = np.exp(np.polyval(coef, x_plot))
                                        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
                                        aic = len(y) * np.log(np.sum((y - y_pred)**2) / len(y)) + 2 * 2
                                        
                                        fig_models.add_trace(go.Scatter(
                                            x=x_plot, y=y_plot,
                                            mode='lines',
                                            name=f'Exponential (R²={r2:.3f})'
                                        ))
                                        
                                        model_results.append({
                                            'Model': 'Exponential',
                                            'R²': r2,
                                            'AIC': aic,
                                            'Parameters': 2,
                                            'Equation': f'y = {np.exp(coef[1]):.3f}e^({coef[0]:.3f}x)'
                                        })
                                    except:
                                        st.warning("Exponential model failed (requires positive Y values)")
                                
                                # Power model (y = a*x^b)
                                if "Power" in models_to_fit and (x > 0).all() and (y > 0).all():
                                    try:
                                        log_x = np.log(x)
                                        log_y = np.log(y)
                                        coef = np.polyfit(log_x, log_y, 1)
                                        y_pred = np.exp(coef[1]) * x**coef[0]
                                        y_plot = np.exp(coef[1]) * x_plot[x_plot > 0]**coef[0]
                                        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
                                        aic = len(y) * np.log(np.sum((y - y_pred)**2) / len(y)) + 2 * 2
                                        
                                        fig_models.add_trace(go.Scatter(
                                            x=x_plot[x_plot > 0], y=y_plot,
                                            mode='lines',
                                            name=f'Power (R²={r2:.3f})'
                                        ))
                                        
                                        model_results.append({
                                            'Model': 'Power',
                                            'R²': r2,
                                            'AIC': aic,
                                            'Parameters': 2,
                                            'Equation': f'y = {np.exp(coef[1]):.3f}x^{coef[0]:.3f}'
                                        })
                                    except:
                                        st.warning("Power model failed (requires positive X and Y values)")
                                
                                # Sigmoid model
                                if "Sigmoid" in models_to_fit:
                                    try:
                                        from scipy.optimize import curve_fit
                                        def sigmoid(x, a, b, c, d):
                                            return a / (1 + np.exp(-c * (x - b))) + d
                                        
                                        # Initial guess
                                        p0 = [y.max() - y.min(), np.median(x), 1, y.min()]
                                        popt, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
                                        y_pred = sigmoid(x, *popt)
                                        y_plot = sigmoid(x_plot, *popt)
                                        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
                                        aic = len(y) * np.log(np.sum((y - y_pred)**2) / len(y)) + 2 * 4
                                        
                                        fig_models.add_trace(go.Scatter(
                                            x=x_plot, y=y_plot,
                                            mode='lines',
                                            name=f'Sigmoid (R²={r2:.3f})'
                                        ))
                                        
                                        model_results.append({
                                            'Model': 'Sigmoid',
                                            'R²': r2,
                                            'AIC': aic,
                                            'Parameters': 4,
                                            'Equation': f'y = {popt[0]:.3f}/(1+exp(-{popt[2]:.3f}(x-{popt[1]:.3f}))) + {popt[3]:.3f}'
                                        })
                                    except:
                                        st.warning("Sigmoid model fitting failed")
                                
                                # Update layout
                                fig_models.update_layout(
                                    title=f"Model Comparison: {get_param_info(reg_x)} vs {get_param_info(reg_y)}",
                                    xaxis_title=get_param_info(reg_x),
                                    yaxis_title=get_param_info(reg_y),
                                    height=600,
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig_models, use_container_width=True)
                                
                                # Model comparison table
                                if model_results:
                                    st.subheader("Model Comparison")
                                    model_df = pd.DataFrame(model_results).sort_values('AIC')
                                    
                                    # Highlight best model
                                    best_r2 = model_df['R²'].max()
                                    best_aic = model_df['AIC'].min()
                                    
                                    def highlight_best(val, col_name):
                                        if col_name == 'R²' and val == best_r2:
                                            return 'background-color: lightgreen'
                                        elif col_name == 'AIC' and val == best_aic:
                                            return 'background-color: lightgreen'
                                        return ''
                                    
                                    styled_df = model_df.style.apply(lambda x: [highlight_best(v, x.name) for v in x], axis=0)
                                    st.dataframe(styled_df, use_container_width=True)
                                    
                                    st.info("💡 **Model Selection Guide:**\n"
                                           "- **R²**: Higher is better (explains more variance)\n"
                                           "- **AIC**: Lower is better (balances fit and complexity)\n"
                                           "- Consider both metrics and domain knowledge for best model selection")
                                    
                                    # Best model details
                                    best_model = model_df.iloc[0]
                                    st.success(f"🏆 Best model by AIC: **{best_model['Model']}**\n\n"
                                              f"Equation: `{best_model['Equation']}`")
                            else:
                                st.warning(f"Need at least {min_data_points} data points.")
                
                # Phase-Space Analysis Tab
                with nl_tabs[3]:
                    st.subheader("Phase-Space and Dynamical Analysis")
                    st.info("Explore the dynamical relationship between parameters using phase-space representations and recurrence analysis.")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        phase_x = st.selectbox(
                            "X-axis parameter:",
                            options=numeric_cols_nl,
                            format_func=get_param_info,
                            key="phase_x"
                        )
                    with col2:
                        phase_y = st.selectbox(
                            "Y-axis parameter:",
                            options=numeric_cols_nl,
                            index=min(1, len(numeric_cols_nl)-1) if len(numeric_cols_nl) > 1 else 0,
                            format_func=get_param_info,
                            key="phase_y"
                        )
                    with col3:
                        time_delay = st.number_input(
                            f"Time delay ({time_unit}):",
                            min_value=0,
                            max_value=30,
                            value=1,
                            key="phase_time_delay"
                        )
                    
                    if phase_x and phase_y:
                        analysis_type = st.radio(
                            "Analysis Type:",
                            ["Phase Portrait", "Delayed Coordinates", "Recurrence Plot", "Return Map"],
                            horizontal=True,
                            key="phase_analysis_type"
                        )
                        
                        if st.button("Generate Phase-Space Analysis", key="gen_phase_button"):
                            phase_data = data_for_nonlinear_analysis[['Date', phase_x, phase_y]].dropna()
                            
                            if len(phase_data) >= min_data_points:
                                if analysis_type == "Phase Portrait":
                                    # Standard phase portrait
                                    fig_phase = go.Figure()
                                    
                                    # Add trajectory
                                    fig_phase.add_trace(go.Scatter(
                                        x=phase_data[phase_x],
                                        y=phase_data[phase_y],
                                        mode='lines+markers',
                                        name='Trajectory',
                                        line=dict(width=1),
                                        marker=dict(size=3)
                                    ))
                                    
                                    # Color by time
                                    fig_phase.add_trace(go.Scatter(
                                        x=phase_data[phase_x],
                                        y=phase_data[phase_y],
                                        mode='markers',
                                        name='Time evolution',
                                        marker=dict(
                                            size=6,
                                            color=np.arange(len(phase_data)),
                                            colorscale='Viridis',
                                            showscale=True,
                                            colorbar=dict(title=f"Time ({time_unit})")
                                        )
                                    ))
                                    
                                    # Add start and end points
                                    fig_phase.add_trace(go.Scatter(
                                        x=[phase_data[phase_x].iloc[0]],
                                        y=[phase_data[phase_y].iloc[0]],
                                        mode='markers',
                                        name='Start',
                                        marker=dict(size=10, color='green', symbol='star')
                                    ))
                                    fig_phase.add_trace(go.Scatter(
                                        x=[phase_data[phase_x].iloc[-1]],
                                        y=[phase_data[phase_y].iloc[-1]],
                                        mode='markers',
                                        name='End',
                                        marker=dict(size=10, color='red', symbol='star')
                                    ))
                                    
                                    fig_phase.update_layout(
                                        title=f"Phase Portrait: {get_param_info(phase_x)} vs {get_param_info(phase_y)}",
                                        xaxis_title=get_param_info(phase_x),
                                        yaxis_title=get_param_info(phase_y),
                                        height=600
                                    )
                                    st.plotly_chart(fig_phase, use_container_width=True)
                                    
                                    # Calculate phase space metrics
                                    dx = np.diff(phase_data[phase_x])
                                    dy = np.diff(phase_data[phase_y])
                                    velocity = np.sqrt(dx**2 + dy**2)
                                    
                                    col_m1, col_m2, col_m3 = st.columns(3)
                                    with col_m1:
                                        st.metric("Mean Velocity", f"{velocity.mean():.3f}")
                                    with col_m2:
                                        st.metric("Max Velocity", f"{velocity.max():.3f}")
                                    with col_m3:
                                        area = np.trapz(phase_data[phase_y], phase_data[phase_x])
                                        st.metric("Phase Space Area", f"{abs(area):.1f}")
                                
                                elif analysis_type == "Delayed Coordinates":
                                    # Time-delayed embedding
                                    if time_delay > 0 and time_delay < len(phase_data):
                                        x_delayed = phase_data[phase_x].iloc[:-time_delay].values
                                        x_future = phase_data[phase_x].iloc[time_delay:].values
                                        y_delayed = phase_data[phase_y].iloc[:-time_delay].values
                                        y_future = phase_data[phase_y].iloc[time_delay:].values
                                        
                                        fig_delay = make_subplots(
                                            rows=1, cols=2,
                                            subplot_titles=(f"{get_param_info(phase_x)} Delayed Embedding",
                                                          f"{get_param_info(phase_y)} Delayed Embedding")
                                        )
                                        
                                        # X parameter delayed
                                        fig_delay.add_trace(
                                            go.Scatter(x=x_delayed, y=x_future,
                                                     mode='markers',
                                                     marker=dict(size=5, opacity=0.6)),
                                            row=1, col=1
                                        )
                                        
                                        # Y parameter delayed
                                        fig_delay.add_trace(
                                            go.Scatter(x=y_delayed, y=y_future,
                                                     mode='markers',
                                                     marker=dict(size=5, opacity=0.6)),
                                            row=1, col=2
                                        )
                                        
                                        fig_delay.update_xaxes(title_text=f"Value at t", row=1, col=1)
                                        fig_delay.update_xaxes(title_text=f"Value at t", row=1, col=2)
                                        fig_delay.update_yaxes(title_text=f"Value at t+{time_delay}", row=1, col=1)
                                        fig_delay.update_yaxes(title_text=f"Value at t+{time_delay}", row=1, col=2)
                                        
                                        fig_delay.update_layout(height=500, showlegend=False)
                                        st.plotly_chart(fig_delay, use_container_width=True)
                                        
                                        # Calculate autocorrelation
                                        from scipy.stats import pearsonr
                                        x_autocorr = pearsonr(x_delayed, x_future)[0]
                                        y_autocorr = pearsonr(y_delayed, y_future)[0]
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric(f"{get_param_info(phase_x)} Autocorrelation (lag={time_delay})", 
                                                     f"{x_autocorr:.3f}")
                                        with col2:
                                            st.metric(f"{get_param_info(phase_y)} Autocorrelation (lag={time_delay})", 
                                                     f"{y_autocorr:.3f}")
                                    else:
                                        st.error("Time delay must be less than data length")
                                
                                elif analysis_type == "Recurrence Plot":
                                    # Simple recurrence plot
                                    st.info("Recurrence plots reveal periodic patterns and dynamical structure in time series.")
                                    
                                    # Select which parameter to analyze
                                    recur_param = st.selectbox(
                                        "Select parameter for recurrence analysis:",
                                        [phase_x, phase_y],
                                        format_func=get_param_info,
                                        key="recur_param"
                                    )
                                    
                                    threshold = st.slider(
                                        "Recurrence threshold (% of max distance):",
                                        min_value=5,
                                        max_value=50,
                                        value=15,
                                        key="recur_threshold"
                                    ) / 100
                                    
                                    # Calculate recurrence matrix
                                    data_vec = phase_data[recur_param].values
                                    n = len(data_vec)
                                    
                                    # Limit size for computational efficiency
                                    if n > 500:
                                        st.warning(f"Data truncated to last 500 points for efficiency")
                                        data_vec = data_vec[-500:]
                                        n = 500
                                    
                                    # Calculate distance matrix
                                    dist_matrix = np.abs(data_vec[:, np.newaxis] - data_vec[np.newaxis, :])
                                    max_dist = dist_matrix.max()
                                    
                                    # Create recurrence matrix
                                    recur_matrix = (dist_matrix < threshold * max_dist).astype(int)
                                    
                                    # Plot
                                    fig_recur = go.Figure(data=go.Heatmap(
                                        z=recur_matrix,
                                        colorscale='Greys',
                                        showscale=False
                                    ))
                                    
                                    fig_recur.update_layout(
                                        title=f"Recurrence Plot: {get_param_info(recur_param)}",
                                        xaxis_title=f"Time Index",
                                        yaxis_title=f"Time Index",
                                        height=600,
                                        yaxis=dict(autorange='reversed')
                                    )
                                    st.plotly_chart(fig_recur, use_container_width=True)
                                    
                                    # Recurrence metrics
                                    recurrence_rate = np.sum(recur_matrix) / (n * n)
                                    determinism = np.sum([np.sum(np.diag(recur_matrix, k)) for k in range(1, n)]) / np.sum(recur_matrix)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Recurrence Rate", f"{recurrence_rate:.3f}")
                                    with col2:
                                        st.metric("Determinism", f"{determinism:.3f}")
                                
                                elif analysis_type == "Return Map":
                                    # Poincaré return map
                                    st.info("Return maps help identify periodic orbits and chaotic behavior.")
                                    
                                    # Find local maxima or threshold crossings
                                    return_param = st.selectbox(
                                        "Select parameter for return map:",
                                        [phase_x, phase_y],
                                        format_func=get_param_info,
                                        key="return_param"
                                    )
                                    
                                    data_vec = phase_data[return_param].values
                                    
                                    # Find peaks
                                    from scipy.signal import find_peaks
                                    peaks, _ = find_peaks(data_vec, distance=5)
                                    
                                    if len(peaks) > 2:
                                        peak_values = data_vec[peaks]
                                        next_peak_values = peak_values[1:]
                                        current_peak_values = peak_values[:-1]
                                        
                                        fig_return = go.Figure()
                                        fig_return.add_trace(go.Scatter(
                                            x=current_peak_values,
                                            y=next_peak_values,
                                            mode='markers',
                                            marker=dict(size=8, color='blue'),
                                            name='Return map'
                                        ))
                                        
                                        # Add diagonal line
                                        min_val = min(current_peak_values.min(), next_peak_values.min())
                                        max_val = max(current_peak_values.max(), next_peak_values.max())
                                        fig_return.add_trace(go.Scatter(
                                            x=[min_val, max_val],
                                            y=[min_val, max_val],
                                            mode='lines',
                                            line=dict(dash='dash', color='red'),
                                            name='Identity line'
                                        ))
                                        
                                        fig_return.update_layout(
                                            title=f"Return Map: {get_param_info(return_param)} Peak Values",
                                            xaxis_title="Peak n",
                                            yaxis_title="Peak n+1",
                                            height=500
                                        )
                                        st.plotly_chart(fig_return, use_container_width=True)
                                        
                                        # Calculate Lyapunov exponent estimate
                                        if len(peaks) > 10:
                                            distances = np.abs(next_peak_values - current_peak_values)
                                            lyap_estimate = np.mean(np.log(distances[distances > 0]))
                                            st.metric("Lyapunov Exponent Estimate", f"{lyap_estimate:.3f}")
                                            
                                            if lyap_estimate > 0:
                                                st.warning("Positive Lyapunov exponent suggests chaotic behavior")
                                            else:
                                                st.info("Negative Lyapunov exponent suggests stable periodic behavior")
                                    else:
                                        st.warning("Not enough peaks found for return map analysis")
                            else:
                                st.warning(f"Need at least {min_data_points} data points.")
                
                # Causality Testing Tab
                with nl_tabs[4]:
                    st.subheader("Causality and Information Transfer Analysis")
                    st.info("Test for causal relationships between variables using Granger causality and transfer entropy methods.")
                    
                    # Parameter selection
                    col1, col2 = st.columns(2)
                    with col1:
                        cause_param = st.selectbox(
                            "Potential cause (X → Y):",
                            options=numeric_cols_nl,
                            format_func=get_param_info,
                            key="cause_param"
                        )
                    with col2:
                        effect_param = st.selectbox(
                            "Potential effect (Y):",
                            options=numeric_cols_nl,
                            index=min(1, len(numeric_cols_nl)-1) if len(numeric_cols_nl) > 1 else 0,
                            format_func=get_param_info,
                            key="effect_param"
                        )
                    
                    if cause_param and effect_param and cause_param != effect_param:
                        # Causality test options
                        col_opt1, col_opt2 = st.columns(2)
                        with col_opt1:
                            max_lag_causality = st.slider(
                                f"Maximum lag to test ({time_unit}):",
                                min_value=1,
                                max_value=min(30, len(data_for_nonlinear_analysis)//10),
                                value=10,
                                key="max_lag_causality"
                            )
                        with col_opt2:
                            causality_test = st.selectbox(
                                "Causality test type:",
                                ["Granger Causality", "Transfer Entropy", "Both"],
                                key="causality_test_type"
                            )
                        
                        if st.button("Run Causality Analysis", key="run_causality_button"):
                            causality_data = data_for_nonlinear_analysis[[cause_param, effect_param]].dropna()
                            
                            if len(causality_data) >= max_lag_causality * 3:
                                results_container = st.container()
                                
                                with results_container:
                                    if causality_test in ["Granger Causality", "Both"]:
                                        st.subheader("Granger Causality Test")
                                        
                                        # This try-except block handles potential errors during the test.
                                        try:
                                            from statsmodels.tsa.stattools import grangercausalitytests
                                            
                                            # Prepare data (ensure stationarity)
                                            x = causality_data[cause_param].values
                                            y = causality_data[effect_param].values
                                            
                                            # Simple differencing for stationarity
                                            x_diff = np.diff(x)
                                            y_diff = np.diff(y)
                                            
                                            test_data = np.column_stack([y_diff, x_diff])
                                            
                                            # Run Granger causality test
                                            with st.spinner("Running Granger causality tests..."):
                                                gc_results = grangercausalitytests(test_data, maxlag=max_lag_causality, verbose=False)
                                            
                                            # Extract p-values from the F-test result
                                            p_values = []
                                            lags = list(range(1, max_lag_causality + 1))
                                            for lag in lags:
                                                p_val = gc_results[lag][0]['ssr_ftest'][1]
                                                p_values.append(p_val)
                                            
                                            # Plot p-values
                                            fig_granger = go.Figure()
                                            fig_granger.add_trace(go.Scatter(
                                                x=lags,
                                                y=p_values,
                                                mode='lines+markers',
                                                name='p-value'
                                            ))
                                            fig_granger.add_hline(
                                                y=0.05,
                                                line_dash="dash",
                                                line_color="red",
                                                annotation_text="α = 0.05"
                                            )
                                            fig_granger.update_layout(
                                                title=f"Granger Causality: {get_param_info(cause_param)} → {get_param_info(effect_param)}",
                                                xaxis_title=f"Lag ({time_unit})",
                                                yaxis_title="p-value",
                                                yaxis_type="log",
                                                height=400
                                            )
                                            st.plotly_chart(fig_granger, use_container_width=True)
                                            
                                            # Find and report significant lags
                                            significant_lags = [lag for lag, p in zip(lags, p_values) if p < 0.05]
                                            if significant_lags:
                                                st.success(f"✅ Granger causality detected at lags: {significant_lags} {time_unit}")
                                                min_p_value = min(p_values)
                                                best_lag = p_values.index(min_p_value) + 1
                                                st.info(f"Strongest evidence at lag {best_lag} (p={min_p_value:.4f}): {get_param_info(cause_param)} helps predict {get_param_info(effect_param)}.")
                                            else:
                                                st.warning("❌ No significant Granger causality detected (all p-values > 0.05).")
                                                
                                        except Exception as e:
                                            st.error(f"Granger causality test failed. This can happen if data is too short or lacks variance. Error: {str(e)}")
                                    
                                    # Add Transfer Entropy section here if needed
                                    if causality_test in ["Transfer Entropy", "Both"]:
                                        st.subheader("Transfer Entropy Analysis")
                                        st.info("Transfer entropy analysis would go here. This requires additional implementation.")
                            else:
                                st.warning(f"Need at least {max_lag_causality * 3} data points for causality analysis.")
                    else:
                        st.info("Select two different parameters to test for causality.")
        else:
            st.error("❌ Failed to merge data or no data available for non-linear correlation analysis.")
            st.info("Please check that your data files contain proper date columns and numeric data.")

    # Interrupted Time Series Analysis Tab
    with tabs[9]:  # This should be the ITS Analysis tab (adjust index if needed)
        st.header("🔥 Interrupted Time Series (ITS) Analysis")
        st.info("**Natural Experiment Analysis**: Analyze the statistical impact of the 2023 wildfire on water quality parameters using robust ITS methodology. This provides quantitative evidence for claims about wildfire-induced changes.")
        
        merged_data = create_properly_merged_dataframe(available_data)
        
        if merged_data is not None and not merged_data.empty:
            # Configuration section
            st.subheader("📅 Analysis Configuration")
            
            col_config1, col_config2, col_config3 = st.columns(3)
            
            with col_config1:
                # Interruption date selection
                interruption_date = st.date_input(
                    "Wildfire/Interruption Date:",
                    value=date(2023, 7, 15),  # Default to mid-July 2023
                    min_value=date(2017, 1, 1),
                    max_value=date.today(),
                    key="its_interruption_date",
                    help="Select the date when the wildfire or other interruption occurred"
                )
            
            with col_config2:
                # Minimum data requirements
                min_pre_points = st.number_input(
                    "Min. pre-interruption points:",
                    min_value=10,
                    max_value=100,
                    value=20,
                    key="its_min_pre_points",
                    help="Minimum number of observations required before interruption"
                )
            
            with col_config3:
                # Minimum post-interruption points
                min_post_points = st.number_input(
                    "Min. post-interruption points:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="its_min_post_points",
                    help="Minimum number of observations required after interruption"
                )
            
            # Parameter selection for ITS analysis
            st.markdown("---")
            st.subheader("🎯 Parameter Selection for ITS Analysis")
            
            numeric_cols_its = [col for col in merged_data.columns 
                               if col != 'Date' and pd.api.types.is_numeric_dtype(merged_data[col]) 
                               and merged_data[col].notna().any()]
            
            # Filter for key water quality parameters
            priority_params = ['TOC', 'ΘΟΛΟΤΗΤΑ', 'ΘΕΡΜΟΚΡΑΣΙΑ', 'pH', 'ΑΓΩΓΙΜΟΤΗΤΑ', 
                              'ΧΛΩΡΙΟΥΧΑ', 'ΝΙΤΡΙΚΑ (ΝΟ3-)', 'Chl a_S2lwa', 'TSM']
            
            available_priority_params = []
            for param in priority_params:
                matching_params = [col for col in numeric_cols_its if param in col]
                available_priority_params.extend(matching_params)
            
            # Remove duplicates while preserving order
            available_priority_params = list(dict.fromkeys(available_priority_params))
            
            if not available_priority_params:
                available_priority_params = numeric_cols_its[:10]  # Fallback to first 10
            
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                selected_its_params = st.multiselect(
                    "Select parameters for ITS analysis:",
                    options=numeric_cols_its,
                    default=available_priority_params[:3],
                    format_func=get_param_info,
                    key="its_selected_params",
                    help="Select key water quality parameters affected by wildfire"
                )
            
            with col_param2:
                analysis_scope = st.selectbox(
                    "Analysis scope:",
                    ["Individual Analysis", "Batch Analysis", "Comparative Analysis"],
                    key="its_analysis_scope",
                    help="Choose between analyzing one parameter at a time or multiple parameters"
                )
            
            if not selected_its_params:
                st.warning("Please select at least one parameter for ITS analysis.")
            else:
                # Data availability check
                st.markdown("---")
                st.subheader("📊 Data Availability Assessment")
                
                interruption_datetime = pd.to_datetime(interruption_date)
                availability_summary = []
                
                for param in selected_its_params:
                    param_data = merged_data[['Date', param]].dropna()
                    pre_data = param_data[param_data['Date'] < interruption_datetime]
                    post_data = param_data[param_data['Date'] >= interruption_datetime]
                    
                    availability_summary.append({
                        'Parameter': get_param_info(param),
                        'Total Points': len(param_data),
                        'Pre-interruption': len(pre_data),
                        'Post-interruption': len(post_data),
                        'Date Range': f"{param_data['Date'].min().strftime('%Y-%m-%d')} to {param_data['Date'].max().strftime('%Y-%m-%d')}",
                        'ITS Feasible': len(pre_data) >= min_pre_points and len(post_data) >= min_post_points
                    })
                
                availability_df = pd.DataFrame(availability_summary)
                
                # Style the dataframe
                def highlight_feasible(val):
                    if isinstance(val, bool):
                        return 'background-color: lightgreen' if val else 'background-color: lightcoral'
                    return ''
                
                styled_availability = availability_df.style.applymap(highlight_feasible, subset=['ITS Feasible'])
                st.dataframe(styled_availability, use_container_width=True)
                
                feasible_params = [row['Parameter'] for _, row in availability_df.iterrows() if row['ITS Feasible']]
                
                if not feasible_params:
                    st.error("❌ No parameters have sufficient data for ITS analysis. Consider adjusting the interruption date or minimum data requirements.")
                else:
                    st.success(f"✅ {len(feasible_params)} parameter(s) suitable for ITS analysis: {', '.join(feasible_params[:3])}{'...' if len(feasible_params) > 3 else ''}")
                    
                    # Individual Parameter Analysis
                    if analysis_scope == "Individual Analysis":
                        st.markdown("---")
                        st.subheader("🔬 Individual ITS Analysis")
                        
                        # Select parameter for detailed analysis
                        feasible_param_keys = [param for param in selected_its_params 
                                             if get_param_info(param) in feasible_params]
                        
                        if feasible_param_keys:
                            selected_param = st.selectbox(
                                "Select parameter for detailed analysis:",
                                options=feasible_param_keys,
                                format_func=get_param_info,
                                key="its_individual_param"
                            )
                            
                            # Run ITS Analysis button
                            if st.button("🚀 Run ITS Analysis", key="run_its_individual", type="primary"):
                                with st.spinner("Performing Interrupted Time Series Analysis..."):
                                    
                                    # Prepare data
                                    analysis_data = merged_data[['Date', selected_param]].copy()
                                    
                                    # Perform ITS analysis
                                    its_results = perform_its_analysis(
                                        analysis_data, 
                                        selected_param, 
                                        interruption_datetime
                                    )
                                    
                                    if 'error' in its_results:
                                        st.error(f"❌ ITS Analysis failed: {its_results['error']}")
                                    else:
                                        # Display main results
                                        st.markdown("---")
                                        st.subheader("📈 ITS Analysis Results")
                                        
                                        # Main visualization
                                        fig_its = plot_its_results(its_results, selected_param)
                                        if fig_its:
                                            st.plotly_chart(fig_its, use_container_width=True)
                                        
                                        # Key findings
                                        col_results1, col_results2, col_results3 = st.columns(3)
                                        
                                        with col_results1:
                                            st.metric(
                                                "Immediate Level Change",
                                                f"{its_results['interruption_effect']:.4f}",
                                                help="Change in parameter level immediately after interruption"
                                            )
                                            
                                            # Statistical significance
                                            p_val_level = its_results['model'].pvalues['interruption']
                                            if p_val_level < 0.001:
                                                st.success("***Highly significant*** (p < 0.001)")
                                            elif p_val_level < 0.01:
                                                st.success("**Significant** (p < 0.01)")
                                            elif p_val_level < 0.05:
                                                st.success("*Significant* (p < 0.05)")
                                            else:
                                                st.info(f"Not significant (p = {p_val_level:.3f})")
                                        
                                        with col_results2:
                                            st.metric(
                                                "Trend Change (slope)",
                                                f"{its_results['trend_change']:.6f}",
                                                help="Change in the rate of change after interruption"
                                            )
                                            
                                            # Statistical significance for trend change
                                            p_val_trend = its_results['model'].pvalues['time_since_interruption']
                                            if p_val_trend < 0.001:
                                                st.success("***Highly significant*** (p < 0.001)")
                                            elif p_val_trend < 0.01:
                                                st.success("**Significant** (p < 0.01)")
                                            elif p_val_trend < 0.05:
                                                st.success("*Significant* (p < 0.05)")
                                            else:
                                                st.info(f"Not significant (p = {p_val_trend:.3f})")
                                        
                                        with col_results3:
                                            st.metric(
                                                "Model R²",
                                                f"{its_results['model'].rsquared:.3f}",
                                                help="Proportion of variance explained by the ITS model"
                                            )
                                            
                                            # Effect sizes
                                            st.metric(
                                                "Effect Size (Level)",
                                                f"{its_results['effect_size_immediate']:.2f}",
                                                help="Standardized effect size for immediate change"
                                            )
                                        
                                        # Model diagnostics
                                        with st.expander("🔍 Model Diagnostics"):
                                            fig_diagnostics, summary_stats = create_its_diagnostics_plots(its_results, selected_param)
                                            
                                            if fig_diagnostics:
                                                st.plotly_chart(fig_diagnostics, use_container_width=True)
                                            
                                            if summary_stats:
                                                col_diag1, col_diag2, col_diag3 = st.columns(3)
                                                
                                                with col_diag1:
                                                    st.metric("R-squared", f"{summary_stats['R-squared']:.3f}")
                                                    st.metric("AIC", f"{summary_stats['AIC']:.1f}")
                                                
                                                with col_diag2:
                                                    st.metric("Durbin-Watson", f"{summary_stats['Durbin-Watson']:.3f}")
                                                    if 1.5 <= summary_stats['Durbin-Watson'] <= 2.5:
                                                        st.success("✅ No autocorrelation detected")
                                                    else:
                                                        st.warning("⚠️ Potential autocorrelation")
                                                
                                                with col_diag3:
                                                    bp_p = summary_stats['BP Test p-value']
                                                    st.metric("BP Test p-value", f"{bp_p:.3f}" if not np.isnan(bp_p) else "N/A")
                                                    if not np.isnan(bp_p):
                                                        if bp_p > 0.05:
                                                            st.success("✅ Homoscedasticity")
                                                        else:
                                                            st.warning("⚠️ Heteroscedasticity detected")
                    
                    elif analysis_scope == "Batch Analysis":
                        st.markdown("---")
                        st.subheader("📊 Batch ITS Analysis")
                        st.info("Analyze multiple parameters simultaneously to identify patterns across water quality indicators.")
                        
                        # Filter to feasible parameters only
                        feasible_param_keys = [param for param in selected_its_params 
                                             if get_param_info(param) in feasible_params]
                        
                        if len(feasible_param_keys) < 2:
                            st.warning("Need at least 2 feasible parameters for batch analysis.")
                        else:
                            if st.button("🚀 Run Batch ITS Analysis", key="run_its_batch", type="primary"):
                                batch_results = {}
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, param in enumerate(feasible_param_keys):
                                    status_text.text(f"Analyzing {get_param_info(param)}...")
                                    
                                    analysis_data = merged_data[['Date', param]].copy()
                                    its_result = perform_its_analysis(analysis_data, param, interruption_datetime)
                                    
                                    if 'error' not in its_result:
                                        batch_results[param] = its_result
                                    
                                    progress_bar.progress((i + 1) / len(feasible_param_keys))
                                
                                status_text.text("Analysis complete!")
                                
                                if batch_results:
                                    # Summary table
                                    st.subheader("📋 Batch Analysis Summary")
                                    
                                    summary_data = []
                                    for param, result in batch_results.items():
                                        model = result['model']
                                        summary_data.append({
                                            'Parameter': get_param_info(param),
                                            'Level Change': result['interruption_effect'],
                                            'Level p-value': model.pvalues['interruption'],
                                            'Trend Change': result['trend_change'],
                                            'Trend p-value': model.pvalues['time_since_interruption'],
                                            'R²': model.rsquared,
                                            'Effect Size': result['effect_size_immediate'],
                                            'N (pre/post)': f"{result['pre_count']}/{result['post_count']}"
                                        })
                                    
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    # Key findings
                                    significant_level_changes = len(summary_df[summary_df['Level p-value'] < 0.05])
                                    significant_trend_changes = len(summary_df[summary_df['Trend p-value'] < 0.05])
                                    large_effects = len(summary_df[abs(summary_df['Effect Size']) > 0.5])
                                    
                                    col_batch1, col_batch2, col_batch3 = st.columns(3)
                                    
                                    with col_batch1:
                                        st.metric(
                                            "Parameters with Significant Level Change",
                                            f"{significant_level_changes}/{len(batch_results)}",
                                            f"{significant_level_changes/len(batch_results)*100:.1f}%"
                                        )
                                    
                                    with col_batch2:
                                        st.metric(
                                            "Parameters with Significant Trend Change",
                                            f"{significant_trend_changes}/{len(batch_results)}",
                                            f"{significant_trend_changes/len(batch_results)*100:.1f}%"
                                        )
                                    
                                    with col_batch3:
                                        st.metric(
                                            "Parameters with Large Effect Size",
                                            f"{large_effects}/{len(batch_results)}",
                                            f"{large_effects/len(batch_results)*100:.1f}%"
                                        )
                    
                    elif analysis_scope == "Comparative Analysis":
                        st.markdown("---")
                        st.subheader("🔄 Comparative ITS Analysis")
                        st.info("Compare the wildfire impact across different parameter categories.")
                        
                        # Define parameter categories
                        param_categories = {
                            'Physical Parameters': ['ΘΟΛΟΤΗΤΑ', 'ΘΕΡΜΟΚΡΑΣΙΑ', 'ΑΓΩΓΙΜΟΤΗΤΑ', 'ΣΚΛΗΡΟΤΗΤΑ'],
                            'Chemical Parameters': ['pH', 'TOC', 'ΧΛΩΡΙΟΥΧΑ', 'ΘΕΙΙΚΑ', 'ΦΩΣΦΟΡΟΣ (P2O5)'],
                            'Nutrients': ['ΝΙΤΡΙΚΑ (ΝΟ3-)', 'ΝΙΤΡΩΔΗ (ΝΟ2-)', 'ΑΜΜΩΝΙΑΚΑ (ΝΗ4+)'],
                            'Metals': ['ΣΙΔΗΡΟΣ', 'ΜΑΓΓΑΝΙΟ', 'ΑΡΓΙΛΙΟ', 'ΜΑΓΝΗΣΙΟ', 'ΑΣΒΕΣΤΙΟ'],
                            'Satellite Indicators': ['Chl a_S2lwa', 'Chl a_GEE', 'TSM', 'temperature']
                        }
                        
                        # Find which categories have feasible parameters
                        available_categories = {}
                        for category, params in param_categories.items():
                            category_params = []
                            for param in params:
                                matching_params = [col for col in selected_its_params if param in col and get_param_info(col) in feasible_params]
                                category_params.extend(matching_params)
                            if category_params:
                                available_categories[category] = category_params
                        
                        if len(available_categories) < 2:
                            st.warning("Need parameters from at least 2 categories for comparative analysis.")
                        else:
                            selected_categories = st.multiselect(
                                "Select parameter categories to compare:",
                                options=list(available_categories.keys()),
                                default=list(available_categories.keys())[:2],
                                key="its_selected_categories"
                            )
                            
                            if len(selected_categories) >= 2:
                                if st.button("🚀 Run Comparative Analysis", key="run_its_comparative", type="primary"):
                                    st.info("Comparative analysis implementation would go here...")
                
                # General recommendations and interpretation
                st.markdown("---")
                st.subheader("💡 ITS Analysis Interpretation Guide")
                
                with st.expander("📚 Understanding ITS Results"):
                    st.markdown("""
                    ### 🎯 Key ITS Components:
                    
                    **1. Level Change (β₂):**
                    - Immediate change in parameter value right after the wildfire
                    - **Positive value:** Parameter increased immediately after wildfire
                    - **Negative value:** Parameter decreased immediately after wildfire
                    - **Statistical significance (p < 0.05):** Strong evidence of wildfire impact
                    
                    **2. Trend Change (β₃):**
                    - Change in the rate of change (slope) after the wildfire
                    - **Positive value:** Parameter is increasing faster (or decreasing slower) post-wildfire
                    - **Negative value:** Parameter is decreasing faster (or increasing slower) post-wildfire
                    - **Statistical significance (p < 0.05):** Strong evidence of sustained impact
                    
                    **3. Effect Size:**
                    - **> 0.8:** Large effect (substantial practical significance)
                    - **0.5-0.8:** Medium effect (moderate practical significance)
                    - **0.2-0.5:** Small effect (minor but detectable change)
                    - **< 0.2:** Minimal effect (may not be practically significant)
                    """)
        
        else:
            st.error("❌ No merged data available for ITS analysis. Please check your data sources.")
            st.info("Ensure you have uploaded data files with proper date columns and numeric parameters.")