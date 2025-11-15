import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Economics Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Votre clé API FRED
FRED_API_KEY = "ce5dbb3d3fcd8669f2fe2cdd9c79a7da"

# CSS Bloomberg style
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    .bloomberg-header {
        background: #FFAA00;
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 14px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }
    
    h1, h2, h3, h4 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 12px !important;
        margin: 8px 0 !important;
        border-bottom: 1px solid #333;
        padding-bottom: 4px !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFAA00 !important;
        font-size: 10px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        font-weight: bold;
        border: 1px solid #FFAA00;
        padding: 6px 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 10px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    .stTextInput input, .stSelectbox select {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    hr {
        border-color: #333333;
        margin: 5px 0;
    }
    
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .warning-box {
        background-color: #1a0a00;
        border-left: 3px solid #FF6600;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    .indicator-box {
        background-color: #0a0a0a;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - ECONOMIC DATA</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonctions FRED API
@st.cache_data(ttl=3600)
def get_fred_series(series_id, observation_start=None):
    """Récupère une série de données FRED"""
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        if observation_start:
            params['observation_start'] = observation_start
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            
            if observations:
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                df = df.sort_values('date')
                
                return df
        
        return None
        
    except Exception as e:
        st.error(f"Erreur FRED API pour {series_id}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_fred_series_info(series_id):
    """Récupère les informations sur une série FRED"""
    try:
        url = "https://api.stlouisfed.org/fred/series"
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            series = data.get('seriess', [])
            if series:
                return series[0]
        
        return None
        
    except Exception as e:
        return None

def calculate_yoy_change(df):
    """Calcule la variation Year-over-Year"""
    if df is None or len(df) < 12:
        return None
    
    df = df.copy()
    df['yoy_change'] = df['value'].pct_change(12) * 100
    return df

# Dictionnaire des séries économiques importantes
ECONOMIC_SERIES = {
    'Interest Rates': {
        'FEDFUNDS': 'Fed Funds Rate',
        'DGS10': '10-Year Treasury',
        'DGS2': '2-Year Treasury',
        'DGS30': '30-Year Treasury',
        'MORTGAGE30US': '30Y Mortgage Rate'
    },
    'Inflation': {
        'CPIAUCSL': 'CPI (All Items)',
        'CPILFESL': 'Core CPI',
        'PCEPI': 'PCE Price Index',
        'PCEPILFE': 'Core PCE'
    },
    'Employment': {
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'JTSJOL': 'JOLTS Job Openings',
        'ICSA': 'Initial Claims'
    },
    'Growth': {
        'GDP': 'GDP',
        'GDPC1': 'Real GDP',
        'INDPRO': 'Industrial Production',
        'RSXFS': 'Retail Sales'
    },
    'Markets': {
        'SP500': 'S&P 500',
        'VIXCLS': 'VIX',
        'DEXUSEU': 'EUR/USD',
        'T10Y2Y': '10Y-2Y Spread'
    },
    'Monetary': {
        'M2SL': 'M2 Money Supply',
        'WALCL': 'Fed Balance Sheet'
    }
}

# ONGLETS PRINCIPAUX
tab1, tab2, tab3, tab4 = st.tabs(["📊 DASHBOARD", "📈 CUSTOM ANALYSIS", "🔍 SERIES SEARCH", "📥 DOWNLOAD DATA"])

# TAB 1: DASHBOARD
with tab1:
    st.markdown("### 📊 ECONOMIC INDICATORS DASHBOARD")
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_dash"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # TAUX D'INTÉRÊT
    st.markdown("#### 💰 INTEREST RATES")
    cols_rates = st.columns(4)
    
    rates_series = [
        ('FEDFUNDS', 'FED FUNDS'),
        ('DGS10', '10Y TREASURY'),
        ('DGS2', '2Y TREASURY'),
        ('MORTGAGE30US', '30Y MORTGAGE')
    ]
    
    for idx, (series_id, label) in enumerate(rates_series):
        with cols_rates[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change = current_value - previous_value
                
                st.metric(
                    label=label,
                    value=f"{current_value:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # INFLATION
    st.markdown("#### 📊 INFLATION")
    cols_inflation = st.columns(4)
    
    inflation_series = [
        ('CPIAUCSL', 'CPI'),
        ('CPILFESL', 'CORE CPI'),
        ('PCEPI', 'PCE'),
        ('PCEPILFE', 'CORE PCE')
    ]
    
    for idx, (series_id, label) in enumerate(inflation_series):
        with cols_inflation[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 12:
                df_yoy = calculate_yoy_change(df)
                current_yoy = df_yoy['yoy_change'].iloc[-1]
                previous_yoy = df_yoy['yoy_change'].iloc[-2]
                change = current_yoy - previous_yoy
                
                st.metric(
                    label=f"{label} YoY",
                    value=f"{current_yoy:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # YIELD CURVE
    st.markdown("#### 📉 YIELD CURVE")
    
    treasury_rates = {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    }
    
    curve_data = []
    maturities = []
    
    for maturity, series_id in treasury_rates.items():
        df = get_fred_series(series_id)
        if df is not None and len(df) > 0:
            curve_data.append(df['value'].iloc[-1])
            maturities.append(maturity)
    
    if curve_data:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=curve_data,
            mode='lines+markers',
            line=dict(color='#FFAA00', width=2),
            marker=dict(size=8, color='#00FF00')
        ))
        
        fig_curve.update_layout(
            title="US Treasury Yield Curve",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(title="Maturity", gridcolor='#333'),
            yaxis=dict(title="Yield (%)", gridcolor='#333'),
            height=400
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)

# TAB 2: ANALYSE PERSONNALISÉE
with tab2:
    st.markdown("### 📈 CUSTOM ECONOMIC ANALYSIS")
    
    col_analysis1, col_analysis2 = st.columns([2, 1])
    
    with col_analysis1:
        categories = list(ECONOMIC_SERIES.keys())
        selected_category = st.selectbox(
            "SELECT CATEGORY",
            options=categories,
            key="category_select"
        )
        
        series_in_category = ECONOMIC_SERIES[selected_category]
        
        selected_series = st.multiselect(
            "SELECT SERIES TO COMPARE",
            options=list(series_in_category.keys()),
            default=list(series_in_category.keys())[:2],
            format_func=lambda x: f"{x} - {series_in_category[x]}",
            key="series_multiselect"
        )
    
    with col_analysis2:
        lookback_period = st.selectbox(
            "LOOKBACK PERIOD",
            options=['1Y', '2Y', '5Y', '10Y', 'MAX'],
            index=2,
            key="lookback_select"
        )
    
    if st.button("📊 GENERATE ANALYSIS", use_container_width=True, key="generate_analysis"):
        if selected_series:
            if lookback_period == 'MAX':
                start_date = None
            else:
                years = int(lookback_period[:-1])
                start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
            
            series_data = {}
            for series_id in selected_series:
                df = get_fred_series(series_id, observation_start=start_date)
                if df is not None:
                    series_data[series_id] = df
            
            if series_data:
                fig = go.Figure()
                
                for series_id, df in series_data.items():
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['value'],
                        mode='lines',
                        name=f"{series_id}",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_category} Comparison",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Value"),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: RECHERCHE
with tab3:
    st.markdown("### 🔍 SEARCH FRED SERIES")
    
    st.markdown("#### 📚 AVAILABLE SERIES BY CATEGORY")
    
    for category, series in ECONOMIC_SERIES.items():
        with st.expander(f"📊 {category}", expanded=False):
            for series_id, description in series.items():
                st.markdown(f"**`{series_id}`** - {description}")
    
    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔍 CUSTOM SERIES SEARCH")
    
    custom_series_id = st.text_input(
        "Enter FRED Series ID",
        placeholder="Ex: GDP, UNRATE, CPIAUCSL...",
        key="custom_series_input"
    ).upper()
    
    if st.button("🔍 SEARCH", use_container_width=True, key="search_custom"):
        if custom_series_id:
            df = get_fred_series(custom_series_id)
            
            if df is not None:
                st.success(f"✅ Series found: {custom_series_id}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['value'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2)
                ))
                
                fig.update_layout(
                    title=f"{custom_series_id} - Historical Data",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333'),
                    yaxis=dict(gridcolor='#333'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Latest 20 Observations:**")
                st.dataframe(df.tail(20).sort_values('date', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Series '{custom_series_id}' not found")
        else:
            st.warning("⚠️ Please enter a series ID")

# TAB 4: TÉLÉCHARGEMENT
with tab4:
    st.markdown("### 📥 DOWNLOAD ECONOMIC DATA")
    
    download_category = st.selectbox(
        "SELECT CATEGORY",
        options=list(ECONOMIC_SERIES.keys()),
        key="download_category"
    )
    
    series_in_download = ECONOMIC_SERIES[download_category]
    
    download_series = st.multiselect(
        "SELECT SERIES",
        options=list(series_in_download.keys()),
        default=list(series_in_download.keys()),
        format_func=lambda x: f"{x} - {series_in_download[x]}",
        key="download_series"
    )
    
    if st.button("📥 DOWNLOAD CSV", use_container_width=True, key="download_button"):
        if download_series:
            combined_df = pd.DataFrame()
            
            for series_id in download_series:
                df = get_fred_series(series_id)
                if df is not None:
                    df_temp = df[['date', 'value']].copy()
                    df_temp = df_temp.rename(columns={'value': series_id})
                    
                    if combined_df.empty:
                        combined_df = df_temp
                    else:
                        combined_df = pd.merge(combined_df, df_temp, on='date', how='outer')
            
            if not combined_df.empty:
                csv_data = combined_df.to_csv(index=False)
                st.download_button(
                    label="💾 DOWNLOAD CSV",
                    data=csv_data,
                    file_name=f"economic_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success(f"✅ Data prepared! {len(combined_df)} observations")
                st.dataframe(combined_df.head(20), use_container_width=True)

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | FEDERAL RESERVE ECONOMIC DATA (FRED) | LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
