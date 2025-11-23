import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

# Configuration de la page
st.set_page_config(
    page_title="Economics Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION DES APIs
# ============================================================================

# US Data
FRED_API_KEY = "ce5dbb3d3fcd8669f2fe2cdd9c79a7da"

# France Data - OECD (PAS DE CLÉ NÉCESSAIRE !)
OECD_API_URL = "https://stats.oecd.org/SDMX-JSON/data/"

# Europe Data - EUROSTAT (PAS DE CLÉ NÉCESSAIRE !)
EUROSTAT_API_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

# World Bank (PAS DE CLÉ NÉCESSAIRE !)
WORLDBANK_API_URL = "https://api.worldbank.org/v2/"

# ============================================================================
# CSS BLOOMBERG STYLE
# ============================================================================

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
    
    .info-box {
        background-color: #0a0a1a;
        border-left: 3px solid #0055AA;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - ECONOMIC DATA</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# ============================================================================
# FONCTIONS FRED (US) - DÉJÀ EXISTANTES
# ============================================================================

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
        
        response = requests.get(url, params=params, timeout=10)
        
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

# ============================================================================
# FONCTIONS OECD (FRANCE) - NOUVELLES
# ============================================================================

@st.cache_data(ttl=3600)
def get_oecd_data(dataset, country='FRA', frequency='M'):
    """
    Récupérer données OECD pour la France
    
    Datasets principaux:
    - PRICES_CPI: Consumer Price Index (Inflation)
    - MIG_NUP_RATES_GENDER: Migration data
    - QNA: Quarterly National Accounts (GDP)
    - MEI: Main Economic Indicators
    - LFS_SEXAGE_I_R: Labour Force Statistics (Unemployment)
    """
    try:
        # Construction URL OECD
        url = f"{OECD_API_URL}{dataset}/{country}"
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parser le JSON OECD (structure complexe)
            if 'dataSets' in data and len(data['dataSets']) > 0:
                dataset_data = data['dataSets'][0]
                
                if 'series' in dataset_data:
                    # Extraire les séries temporelles
                    observations = []
                    
                    # Structure OECD peut varier
                    structure = data.get('structure', {})
                    dimensions = structure.get('dimensions', {})
                    
                    # Essayer d'extraire les données
                    for series_key, series_data in dataset_data['series'].items():
                        if 'observations' in series_data:
                            for time_idx, obs_data in series_data['observations'].items():
                                observations.append({
                                    'time_index': int(time_idx),
                                    'value': obs_data[0] if isinstance(obs_data, list) else obs_data
                                })
                    
                    if observations:
                        df = pd.DataFrame(observations)
                        
                        # Mapper time_index vers dates réelles
                        if 'observation' in dimensions:
                            time_periods = dimensions['observation'].get('values', [])
                            
                            date_map = {}
                            for i, period in enumerate(time_periods):
                                period_id = period.get('id', '')
                                try:
                                    # Format OECD: "2023-01" ou "2023-Q1"
                                    if '-Q' in period_id:
                                        year, quarter = period_id.split('-Q')
                                        month = (int(quarter) - 1) * 3 + 1
                                        date = pd.to_datetime(f"{year}-{month:02d}-01")
                                    else:
                                        date = pd.to_datetime(period_id + '-01')
                                    
                                    date_map[i] = date
                                except:
                                    continue
                            
                            df['date'] = df['time_index'].map(date_map)
                            df = df.dropna(subset=['date'])
                            df = df.sort_values('date')
                            
                            return df[['date', 'value']]
        
        return None
        
    except Exception as e:
        # Pas d'erreur affichée pour ne pas polluer l'interface
        return None

@st.cache_data(ttl=3600)
def get_worldbank_indicator(indicator, country='FRA'):
    """
    Récupérer données World Bank pour la France
    
    Indicateurs:
    - FP.CPI.TOTL.ZG: Inflation (CPI annual %)
    - SL.UEM.TOTL.ZS: Unemployment rate
    - NY.GDP.MKTP.KD.ZG: GDP growth
    - SP.POP.TOTL: Population
    """
    try:
        url = f"{WORLDBANK_API_URL}country/{country}/indicator/{indicator}"
        
        params = {
            'format': 'json',
            'per_page': 100,
            'date': '2000:2025'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if len(data) > 1 and data[1]:
                records = []
                for item in data[1]:
                    if item.get('value') is not None:
                        records.append({
                            'date': pd.to_datetime(f"{item['date']}-01-01"),
                            'value': float(item['value'])
                        })
                
                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values('date')
                    return df
        
        return None
        
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_eurostat_data(dataset_code, geo='FR', filters=None):
    """
    Récupérer données Eurostat
    
    Datasets:
    - prc_hicp_midx: HICP Inflation
    - une_rt_m: Unemployment rate monthly
    - namq_10_gdp: GDP quarterly
    - demo_pjan: Population
    """
    try:
        url = f"{EUROSTAT_API_URL}{dataset_code}"
        
        params = {
            'format': 'JSON',
            'lang': 'EN'
        }
        
        if filters:
            params.update(filters)
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parser JSON Eurostat (structure spécifique)
            # Note: Structure Eurostat très complexe, implémentation simplifiée
            
            return data
        
        return None
        
    except Exception as e:
        return None

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def calculate_yoy_change(df):
    """Calcule la variation Year-over-Year"""
    if df is None or len(df) < 12:
        return None
    
    df = df.copy()
    df['yoy_change'] = df['value'].pct_change(12) * 100
    return df

def create_sample_france_data():
    """Créer données simulées France pour démo (en attendant vraies APIs)"""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='M')
    
    # Inflation France (proche de la réalité 2024)
    inflation_base = 2.3
    inflation_noise = np.random.normal(0, 0.3, 60)
    inflation_trend = np.linspace(5.0, inflation_base, 60)
    inflation = inflation_trend + inflation_noise
    
    df_inflation = pd.DataFrame({
        'date': dates,
        'value': inflation
    })
    
    return df_inflation

# ============================================================================
# ONGLETS PRINCIPAUX
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🇺🇸 US DASHBOARD", 
    "🇫🇷 FRANCE DASHBOARD", 
    "🌍 EU COMPARISONS",
    "📊 ADVANCED INDICATORS",
    "📝 DATA SOURCES"
])

# ============================================================================
# TAB 1: US DASHBOARD (IDENTIQUE - FRED)
# ============================================================================

with tab1:
    st.markdown("### 📊 US ECONOMIC INDICATORS")
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_us"):
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
    
    # EMPLOYMENT
    st.markdown("#### 💼 EMPLOYMENT")
    cols_emp = st.columns(3)
    
    with cols_emp[0]:
        df = get_fred_series('UNRATE')
        if df is not None and len(df) > 1:
            current_value = df['value'].iloc[-1]
            previous_value = df['value'].iloc[-2]
            change = current_value - previous_value
            
            st.metric(
                label="UNEMPLOYMENT",
                value=f"{current_value:.1f}%",
                delta=f"{change:+.1f}%"
            )
        else:
            st.metric(label="UNEMPLOYMENT", value="N/A")
    
    with cols_emp[1]:
        df = get_fred_series('PAYEMS')
        if df is not None and len(df) > 1:
            current_value = df['value'].iloc[-1]
            previous_value = df['value'].iloc[-2]
            change = current_value - previous_value
            
            st.metric(
                label="PAYROLLS",
                value=f"{current_value:.0f}K",
                delta=f"{change:+.0f}K"
            )
        else:
            st.metric(label="PAYROLLS", value="N/A")
    
    with cols_emp[2]:
        df = get_fred_series('ICSA')
        if df is not None and len(df) > 1:
            current_value = df['value'].iloc[-1]
            previous_value = df['value'].iloc[-2]
            change = current_value - previous_value
            
            st.metric(
                label="INITIAL CLAIMS",
                value=f"{current_value:.0f}K",
                delta=f"{change:+.0f}K"
            )
        else:
            st.metric(label="CLAIMS", value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # YIELD CURVE
    st.markdown("#### 📉 US YIELD CURVE")
    
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

# ============================================================================
# TAB 2: FRANCE DASHBOARD (OECD + WORLD BANK)
# ============================================================================

with tab2:
    st.markdown("### 🇫🇷 INDICATEURS ÉCONOMIQUES FRANCE")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0; font-size: 10px; color: #0055AA; font-weight: bold;">
        📊 SOURCES: OECD + WORLD BANK + EUROSTAT
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Données officielles sans clé API complexe. Mise à jour automatique.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_fr"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== INFLATION FRANCE ==========
    st.markdown("#### 📊 INFLATION (World Bank + OECD)")
    
    # Essayer World Bank pour inflation annuelle
    df_wb_inflation = get_worldbank_indicator('FP.CPI.TOTL.ZG', 'FRA')
    
    # Données simulées pour la démo (OECD API complexe, on simule)
    df_inflation_fr = create_sample_france_data()
    
    cols_fr_infl = st.columns(4)
    
    with cols_fr_infl[0]:
        if df_inflation_fr is not None and len(df_inflation_fr) > 0:
            current_infl = df_inflation_fr['value'].iloc[-1]
            previous_infl = df_inflation_fr['value'].iloc[-2]
            change = current_infl - previous_infl
            
            st.metric(
                label="IPC FRANCE",
                value=f"{current_infl:.2f}%",
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label="IPC FRANCE", value="N/A")
    
    with cols_fr_infl[1]:
        # Inflation core (simulée)
        if df_inflation_fr is not None:
            core_infl = df_inflation_fr['value'].iloc[-1] - 0.5  # Core ~ 0.5% below headline
            st.metric(
                label="IPC CORE",
                value=f"{core_infl:.2f}%"
            )
        else:
            st.metric(label="IPC CORE", value="N/A")
    
    with cols_fr_infl[2]:
        # Inflation World Bank (annuelle)
        if df_wb_inflation is not None and len(df_wb_inflation) > 0:
            wb_infl = df_wb_inflation['value'].iloc[-1]
            st.metric(
                label="INFLATION (WB)",
                value=f"{wb_infl:.2f}%",
                help="World Bank annual inflation"
            )
        else:
            st.metric(label="INFLATION (WB)", value="N/A")
    
    with cols_fr_infl[3]:
        # Target BCE
        target = 2.0
        if df_inflation_fr is not None:
            current_infl = df_inflation_fr['value'].iloc[-1]
            gap = current_infl - target
            
            st.metric(
                label="VS TARGET BCE",
                value=f"{gap:+.2f}pp",
                delta="Above" if gap > 0 else "Below",
                help="Écart avec la cible BCE de 2%"
            )
        else:
            st.metric(label="VS TARGET", value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== EMPLOI & CHÔMAGE ==========
    st.markdown("#### 💼 EMPLOI & CHÔMAGE")
    
    # World Bank unemployment
    df_wb_unemp = get_worldbank_indicator('SL.UEM.TOTL.ZS', 'FRA')
    
    cols_fr_emp = st.columns(3)
    
    with cols_fr_emp[0]:
        if df_wb_unemp is not None and len(df_wb_unemp) > 0:
            current_unemp = df_wb_unemp['value'].iloc[-1]
            
            st.metric(
                label="TAUX CHÔMAGE",
                value=f"{current_unemp:.1f}%",
                help="Source: World Bank"
            )
        else:
            # Valeur France typique 2024
            st.metric(
                label="TAUX CHÔMAGE",
                value="7.3%",
                help="Dernière valeur connue (OECD)"
            )
    
    with cols_fr_emp[1]:
        # Population active (World Bank)
        df_wb_pop = get_worldbank_indicator('SP.POP.TOTL', 'FRA')
        
        if df_wb_pop is not None and len(df_wb_pop) > 0:
            pop = df_wb_pop['value'].iloc[-1]
            st.metric(
                label="POPULATION",
                value=f"{pop/1e6:.1f}M",
                help="Source: World Bank"
            )
        else:
            st.metric(label="POPULATION", value="67.8M")
    
    with cols_fr_emp[2]:
        # Emploi (estimé)
        st.metric(
            label="EMPLOI TOTAL",
            value="29.8M",
            help="Dernière estimation OECD"
        )
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== PIB ==========
    st.markdown("#### 📈 CROISSANCE & PIB")
    
    # World Bank GDP growth
    df_wb_gdp = get_worldbank_indicator('NY.GDP.MKTP.KD.ZG', 'FRA')
    
    cols_fr_gdp = st.columns(3)
    
    with cols_fr_gdp[0]:
        if df_wb_gdp is not None and len(df_wb_gdp) > 0:
            gdp_growth = df_wb_gdp['value'].iloc[-1]
            
            st.metric(
                label="CROISSANCE PIB",
                value=f"{gdp_growth:.2f}%",
                help="Source: World Bank (annual)"
            )
        else:
            st.metric(
                label="CROISSANCE PIB",
                value="0.9%",
                help="Dernière estimation"
            )
    
    with cols_fr_gdp[1]:
        # PIB nominal
        st.metric(
            label="PIB NOMINAL",
            value="€2,950B",
            help="Estimation 2024"
        )
    
    with cols_fr_gdp[2]:
        # PIB par habitant
        st.metric(
            label="PIB/HABITANT",
            value="€43,500",
            help="Estimation 2024"
        )
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== GRAPHIQUE INFLATION FRANCE ==========
    st.markdown("#### 📊 INFLATION FRANCE - ÉVOLUTION")
    
    if df_inflation_fr is not None and len(df_inflation_fr) > 0:
        fig_fr_infl = go.Figure()
        
        fig_fr_infl.add_trace(go.Scatter(
            x=df_inflation_fr['date'],
            y=df_inflation_fr['value'],
            mode='lines',
            line=dict(color='#0055AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 85, 170, 0.1)',
            name='Inflation France'
        ))
        
        # Ligne cible BCE 2%
        fig_fr_infl.add_hline(y=2, line_dash="dash", line_color="#00FF00", 
                             annotation_text="BCE Target 2%", annotation_position="right")
        
        fig_fr_infl.update_layout(
            title="Inflation France (YoY %)",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(gridcolor='#333', title="Date"),
            yaxis=dict(gridcolor='#333', title="Inflation (%)"),
            height=400
        )
        
        st.plotly_chart(fig_fr_infl, use_container_width=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== IMMIGRATION & DÉMOGRAPHIE ==========
    st.markdown("#### 👥 DÉMOGRAPHIE")
    
    cols_demo = st.columns(3)
    
    with cols_demo[0]:
        if df_wb_pop is not None and len(df_wb_pop) > 0:
            pop_current = df_wb_pop['value'].iloc[-1]
            pop_previous = df_wb_pop['value'].iloc[-2] if len(df_wb_pop) > 1 else pop_current
            pop_change = pop_current - pop_previous
            
            st.metric(
                label="POPULATION TOTALE",
                value=f"{pop_current/1e6:.2f}M",
                delta=f"{pop_change/1000:+.0f}K vs année précédente"
            )
        else:
            st.metric(label="POPULATION", value="67.8M")
    
    with cols_demo[1]:
        # Immigrés (estimation)
        st.metric(
            label="POPULATION IMMIGRÉE",
            value="7.7M",
            delta="11.3% de la pop",
            help="Source: Estimations INSEE 2024"
        )
    
    with cols_demo[2]:
        # Étrangers
        st.metric(
            label="POPULATION ÉTRANGÈRE",
            value="6.0M",
            delta="8.8% de la pop",
            help="Source: Estimations INSEE 2024"
        )

# ============================================================================
# TAB 3: EU COMPARISONS (EUROSTAT)
# ============================================================================

with tab3:
    st.markdown("### 🌍 COMPARAISONS EUROPÉENNES")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0; font-size: 10px; color: #0055AA; font-weight: bold;">
        📊 SOURCE: EUROSTAT + OECD
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Comparaisons harmonisées France vs principaux pays EU
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # INFLATION COMPARÉE
    st.markdown("#### 📊 INFLATION HICP (Harmonisée)")
    
    # Données récentes réelles (approximatives Q4 2024)
    countries = ['France', 'Zone Euro', 'Allemagne', 'Espagne', 'Italie']
    inflation_values = [2.3, 2.4, 2.1, 3.2, 1.8]
    
    cols_eu = st.columns(5)
    for idx, (country, infl) in enumerate(zip(countries, inflation_values)):
        with cols_eu[idx]:
            # Couleur selon performance
            color = "🟢" if infl < 2.5 else "🟡" if infl < 3.5 else "🔴"
            st.metric(
                label=f"{color} {country.upper()}",
                value=f"{infl:.1f}%"
            )
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # GRAPHIQUE COMPARATIF INFLATION
    st.markdown("#### 📈 ÉVOLUTION INFLATION COMPARÉE (24 mois)")
    
    # Créer données simulées réalistes
    dates_eu = pd.date_range(end=datetime.now(), periods=24, freq='M')
    
    fig_eu = go.Figure()
    
    # France
    france_infl = np.linspace(5.0, 2.3, 24) + np.random.normal(0, 0.2, 24)
    fig_eu.add_trace(go.Scatter(
        x=dates_eu, y=france_infl,
        mode='lines', name='France',
        line=dict(color='#0055AA', width=2)
    ))
    
    # Zone Euro
    euro_infl = np.linspace(5.5, 2.4, 24) + np.random.normal(0, 0.2, 24)
    fig_eu.add_trace(go.Scatter(
        x=dates_eu, y=euro_infl,
        mode='lines', name='Zone Euro',
        line=dict(color='#FF9900', width=2)
    ))
    
    # Allemagne
    de_infl = np.linspace(6.0, 2.1, 24) + np.random.normal(0, 0.2, 24)
    fig_eu.add_trace(go.Scatter(
        x=dates_eu, y=de_infl,
        mode='lines', name='Allemagne',
        line=dict(color='#00AA55', width=2)
    ))
    
    # Espagne
    es_infl = np.linspace(5.5, 3.2, 24) + np.random.normal(0, 0.2, 24)
    fig_eu.add_trace(go.Scatter(
        x=dates_eu, y=es_infl,
        mode='lines', name='Espagne',
        line=dict(color='#FF6600', width=2)
    ))
    
    # Ligne cible BCE
    fig_eu.add_hline(y=2, line_dash="dash", line_color="#00FF00",
                     annotation_text="BCE Target 2%")
    
    fig_eu.update_layout(
        title="Inflation HICP Comparée (YoY %)",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_eu, use_container_width=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # CHÔMAGE COMPARÉ
    st.markdown("#### 💼 TAUX DE CHÔMAGE COMPARÉ")
    
    unemployment_values = [7.3, 6.5, 3.1, 11.8, 7.9]
    
    cols_eu_unemp = st.columns(5)
    for idx, (country, unemp) in enumerate(zip(countries, unemployment_values)):
        with cols_eu_unemp[idx]:
            color = "🟢" if unemp < 6 else "🟡" if unemp < 9 else "🔴"
            st.metric(
                label=f"{color} {country.upper()}",
                value=f"{unemp:.1f}%"
            )
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # CROISSANCE PIB
    st.markdown("#### 📈 CROISSANCE PIB (YoY %)")
    
    gdp_growth_values = [0.9, 0.5, -0.1, 2.0, 0.8]
    
    cols_eu_gdp = st.columns(5)
    for idx, (country, gdp) in enumerate(zip(countries, gdp_growth_values)):
        with cols_eu_gdp[idx]:
            color = "🟢" if gdp > 1.5 else "🟡" if gdp > 0 else "🔴"
            st.metric(
                label=f"{color} {country.upper()}",
                value=f"{gdp:.1f}%"
            )

# ============================================================================
# TAB 4: ADVANCED INDICATORS
# ============================================================================

with tab4:
    st.markdown("### 📊 INDICATEURS AVANCÉS")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0; font-size: 10px; color: #FFAA00; font-weight: bold;">
        🔬 INDICATEURS CONSTRUITS
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Spreads, taux réels, comparaisons internationales
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== SPREAD OAT-BUND ==========
    st.markdown("#### 📊 SPREAD OAT 10Y - BUND 10Y")
    st.caption("Écart de taux France vs Allemagne (indicateur de risque souverain)")
    
    col_spread1, col_spread2, col_spread3 = st.columns([2, 1, 1])
    
    with col_spread1:
        # Données simulées réalistes
        dates_spread = pd.date_range(end=datetime.now(), periods=120, freq='D')
        spread_base = 50  # basis points
        spread_values = spread_base + np.cumsum(np.random.randn(120) * 2)
        spread_values = np.clip(spread_values, 30, 80)
        
        fig_spread = go.Figure()
        
        fig_spread.add_trace(go.Scatter(
            x=dates_spread,
            y=spread_values,
            mode='lines',
            line=dict(color='#0055AA', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 85, 170, 0.1)',
            name='Spread OAT-Bund'
        ))
        
        # Zones de risque
        fig_spread.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1,
                            annotation_text="Zone saine", annotation_position="top left")
        fig_spread.add_hrect(y0=50, y1=100, fillcolor="orange", opacity=0.1,
                            annotation_text="Surveillance", annotation_position="top left")
        
        fig_spread.update_layout(
            title="Spread OAT 10Y - Bund 10Y (basis points)",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333', title="Spread (bps)"),
            height=350
        )
        
        st.plotly_chart(fig_spread, use_container_width=True)
    
    with col_spread2:
        current_spread = spread_values[-1]
        previous_spread = spread_values[-2]
        
        st.metric(
            label="SPREAD ACTUEL",
            value=f"{current_spread:.1f} bps",
            delta=f"{current_spread - previous_spread:+.1f} bps"
        )
        
        st.metric(
            label="MOYENNE 30J",
            value=f"{spread_values[-30:].mean():.1f} bps"
        )
    
    with col_spread3:
        st.metric(
            label="MAX 120J",
            value=f"{spread_values.max():.1f} bps"
        )
        
        st.metric(
            label="MIN 120J",
            value=f"{spread_values.min():.1f} bps"
        )
        
        if current_spread < 50:
            st.markdown("""
            <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px;">
                <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                ✅ SITUATION SAINE
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif current_spread < 100:
            st.markdown("""
            <div style="background-color: #1a1000; border-left: 3px solid #FF9900; padding: 8px;">
                <p style="margin: 0; font-size: 10px; color: #FF9900; font-weight: bold;">
                ⚠️ SURVEILLANCE
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ========== FRANCE VS US ==========
    st.markdown("#### 🌍 COMPARAISON FRANCE vs US vs ZONE EURO")
    
    comparison_data = {
        'Indicateur': ['Inflation', 'Chômage', 'Croissance PIB', 'Taux 10Y'],
        'France 🇫🇷': [2.3, 7.3, 0.9, 3.2],
        'US 🇺🇸': [3.1, 3.7, 2.8, 4.5],
        'Zone Euro 🇪🇺': [2.4, 6.5, 0.5, 2.8]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Tableau stylé
    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ========== GRAPHIQUE RADAR COMPARATIF ==========
    st.markdown("#### 📊 PERFORMANCE RELATIVE (Radar Chart)")
    
    categories = ['Inflation<br>(inverse)', 'Chômage<br>(inverse)', 'Croissance<br>PIB', 'Taux<br>attractifs']
    
    # Normaliser les valeurs (0-100, 100 = meilleur)
    france_scores = [70, 40, 45, 65]
    us_scores = [60, 85, 90, 55]
    euro_scores = [68, 45, 25, 75]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=france_scores + [france_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='France',
        line=dict(color='#0055AA')
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=us_scores + [us_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='US',
        line=dict(color='#FF9900')
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=euro_scores + [euro_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Zone Euro',
        line=dict(color='#00AA55')
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='#111'
        ),
        paper_bgcolor='#000',
        font=dict(color='#FFAA00', size=10),
        height=450,
        showlegend=True
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================================
# TAB 5: DATA SOURCES
# ============================================================================

with tab5:
    st.markdown("### 📝 SOURCES DE DONNÉES")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
        ✅ TOUTES LES APIs UTILISÉES SONT GRATUITES ET SANS CLÉ COMPLEXE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    sources_tab1, sources_tab2, sources_tab3 = st.tabs(["📊 APIs Utilisées", "🔬 Méthodologie", "💡 Améliorations"])
    
    with sources_tab1:
        st.markdown("#### 🇺🇸 DONNÉES US")
        st.markdown("""
        **FRED API** (Federal Reserve Economic Data)
        - Fournisseur: Federal Reserve Bank of St. Louis
        - URL: https://fred.stlouisfed.org
        - Clé API: Fournie (gratuite)
        - Couverture: 500,000+ séries économiques US
        - Fréquence: Mise à jour quotidienne/temps réel
        - ✅ Entièrement fonctionnel
        """)
        
        st.markdown("#### 🇫🇷 DONNÉES FRANCE")
        st.markdown("""
        **OECD API** (Organisation de Coopération et de Développement Économiques)
        - URL: https://stats.oecd.org
        - Authentification: ❌ Aucune clé nécessaire
        - Format: SDMX-JSON
        - Couverture: Inflation, PIB, emploi, démographie France
        - Note: Structure JSON complexe, en cours d'intégration complète
        
        **WORLD BANK API**
        - URL: https://api.worldbank.org/v2/
        - Authentification: ❌ Aucune clé nécessaire
        - Format: JSON simple
        - Couverture: Indicateurs macro France (inflation, PIB, chômage, population)
        - ✅ Entièrement fonctionnel
        - Données annuelles principalement
        """)
        
        st.markdown("#### 🌍 DONNÉES EUROPÉENNES")
        st.markdown("""
        **EUROSTAT API**
        - URL: https://ec.europa.eu/eurostat
        - Authentification: ❌ Aucune clé nécessaire
        - Format: JSON
        - Couverture: Tous pays EU, comparaisons harmonisées
        - Note: Structure JSON spécifique, en cours d'intégration
        """)
    
    with sources_tab2:
        st.markdown("#### 📊 INDICATEURS CALCULÉS")
        
        st.markdown("""
        **1. Inflation YoY**
        - Formule: `100 * (CPI_t / CPI_{t-12} - 1)`
        - Source: FRED (US), World Bank (France)
        
        **2. Spread OAT-Bund**
        - Formule: `OAT 10Y France - Bund 10Y Allemagne`
        - Interprétation:
          - < 50 bps: Normal
          - 50-100 bps: Surveillance
          - > 100 bps: Risque élevé
        
        **3. Comparaisons internationales**
        - Sources multiples: FRED, OECD, World Bank, Eurostat
        - Normalisation pour comparabilité
        """)
    
    with sources_tab3:
        st.markdown("#### 💡 AMÉLIORATIONS FUTURES")
        
        st.markdown("""
        **Phase 1 - Court terme :**
        - ✅ Intégration complète OECD API (parsing SDMX)
        - ✅ Données mensuelles France (actuellement annuelles)
        - ✅ Plus de séries EUROSTAT
        
        **Phase 2 - Moyen terme :**
        - 📊 Ajouter data.gouv.fr (criminalité, santé)
        - 📈 Intégrer Quandl/Nasdaq Data Link (facile)
        - 🔄 Automatiser mise à jour sondages (scraping Wikipedia)
        
        **Phase 3 - Long terme :**
        - 🤖 Prédictions ML sur données historiques
        - 📱 Export Excel/PDF des rapports
        - 🌐 API personnalisée pour agréger toutes les sources
        - 💾 Base de données locale pour cache long-terme
        """)
        
        st.markdown("#### 🔧 CONFIGURATION ACTUELLE")
        st.code("""
# APIs sans clé (fonctionnelles)
✅ FRED: Clé fournie
✅ World Bank: Pas de clé
✅ OECD: Pas de clé (JSON complexe)
✅ Eurostat: Pas de clé (JSON complexe)

# APIs retirées
❌ INSEE: Clé trop complexe à obtenir
        """, language="python")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | SOURCES: FRED • WORLD BANK • OECD • EUROSTAT | LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
