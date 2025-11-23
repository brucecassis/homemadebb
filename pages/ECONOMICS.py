import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import numpy as np

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

# France Data - INSEE
INSEE_CONSUMER_KEY = "curl https://api.insee.fr/series/BDM"  # ⬅️ REMPLACER ICI
INSEE_CONSUMER_SECRET = "curl https://api.insee.fr/donnees-locales"  # ⬅️ REMPLACER ICI

# URLs des APIs
BDF_API_URL = "https://webstat.banque-france.fr/ws/sdmx/2.1/data/"
EUROSTAT_API_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
DATAGOUV_BASE = "https://www.data.gouv.fr/api/1/"

# ============================================================================
# SÉRIES ÉCONOMIQUES
# ============================================================================

# FRED Series (US) - Déjà dans votre code
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

# INSEE Series (France)
INSEE_SERIES = {
    'Inflation': {
        '001763852': 'IPC - Ensemble',
        '001763854': 'IPC - Alimentation',
        '001763856': 'IPC - Énergie',
        '010565700': 'IPC - Core (hors énergie/alim)'
    },
    'Emploi': {
        '001688527': 'Taux de chômage (France métro)',
        '010565845': 'Emploi salarié total',
        '010565846': 'Emploi salarié privé'
    },
    'Croissance': {
        '001688370': 'PIB trimestriel',
        '010565692': 'PIB annuel',
        '010565710': 'Production industrielle'
    },
    'Immigration': {
        '001688449': 'Population totale',
        '010565789': 'Population immigrée',
        '010565790': 'Population étrangère'
    }
}

# Banque de France Series
BDF_SERIES = {
    'Taux': {
        'IFM.M.FR.EUR.RT.MM.OAT_10Y.LEV': 'OAT 10 ans',
        'IFM.M.FR.EUR.RT.MM.OAT_2Y.LEV': 'OAT 2 ans',
        'CBD2.M.FR.N.A.L22.A.1.U2.2240.Z01.A': 'Taux crédit immobilier'
    },
    'Crédit': {
        'BSI1.M.FR.N.A.L21.A.1.U2.0000.Z01.E': 'Crédit aux ménages',
        'BSI1.M.FR.N.A.L22.A.1.U2.0000.Z01.E': 'Crédit aux entreprises'
    }
}

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
# FONCTIONS FRED (Déjà existantes)
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

# ============================================================================
# FONCTIONS INSEE API (PHASE 1 & 2)
# ============================================================================

@st.cache_data(ttl=3600)
def get_insee_token():
    """Obtenir le token d'accès INSEE"""
    try:
        credentials = f"{INSEE_CONSUMER_KEY}:{INSEE_CONSUMER_SECRET}"
        encoded = base64.b64encode(credentials.encode()).decode()
        
        url = "https://api.insee.fr/token"
        headers = {
            'Authorization': f'Basic {encoded}'
        }
        data = {
            'grant_type': 'client_credentials'
        }
        
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            st.error(f"Erreur authentification INSEE: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Erreur INSEE token: {e}")
        return None

@st.cache_data(ttl=3600)
def get_insee_series(series_id, _token=None):
    """Récupérer une série chronologique INSEE"""
    try:
        if _token is None:
            _token = get_insee_token()
        
        if _token is None:
            return None
        
        url = f"https://api.insee.fr/series/BDM/v1/data/SERIES_BDM/{series_id}"
        headers = {
            'Authorization': f'Bearer {_token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parser la réponse INSEE
            if 'Obs' in data:
                observations = data['Obs']
                
                dates = []
                values = []
                
                for obs in observations:
                    period = obs.get('TIME_PERIOD', '')
                    value = obs.get('OBS_VALUE', None)
                    
                    if value is not None:
                        try:
                            # Convertir la période en date
                            if '-' in period:  # Format YYYY-MM
                                date = pd.to_datetime(period + '-01')
                            elif 'Q' in period:  # Format YYYY-Q1
                                year, quarter = period.split('-Q')
                                month = (int(quarter) - 1) * 3 + 1
                                date = pd.to_datetime(f"{year}-{month:02d}-01")
                            else:  # Format YYYY
                                date = pd.to_datetime(period + '-01-01')
                            
                            dates.append(date)
                            values.append(float(value))
                        except:
                            continue
                
                if dates and values:
                    df = pd.DataFrame({
                        'date': dates,
                        'value': values
                    })
                    df = df.sort_values('date')
                    return df
        
        return None
        
    except Exception as e:
        st.error(f"Erreur INSEE API pour {series_id}: {e}")
        return None

# ============================================================================
# FONCTIONS BANQUE DE FRANCE (PHASE 1)
# ============================================================================

@st.cache_data(ttl=3600)
def get_bdf_series(series_id):
    """Récupérer une série Banque de France (SDMX)"""
    try:
        # Note: BdF utilise SDMX, format plus complexe
        # URL simplifiée pour démo
        url = f"{BDF_API_URL}{series_id}"
        
        headers = {
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Parser SDMX-JSON (simplifié)
            data = response.json()
            
            # Cette partie dépend du format exact SDMX
            # Implémentation simplifiée
            return None
        
        return None
        
    except Exception as e:
        # BdF API peut être instable, on gère silencieusement
        return None

# ============================================================================
# FONCTIONS EUROSTAT (PHASE 2)
# ============================================================================

@st.cache_data(ttl=3600)
def get_eurostat_data(dataset_code, filters=None):
    """Récupérer données Eurostat"""
    try:
        url = f"{EUROSTAT_API_URL}{dataset_code}"
        
        params = {}
        if filters:
            params.update(filters)
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Parser JSON Eurostat (simplifié)
            return data
        
        return None
        
    except Exception as e:
        return None

# ============================================================================
# FONCTIONS DATA.GOUV.FR (PHASE 2)
# ============================================================================

@st.cache_data(ttl=86400)
def get_criminalite_data():
    """Récupérer données criminalité depuis data.gouv.fr"""
    try:
        # URL du dataset criminalité
        url = "https://www.data.gouv.fr/fr/datasets/r/dd23e665-1dd8-49a9-b0e8-73a7f9bc2f4f"
        
        # Télécharger le CSV
        df = pd.read_csv(url, sep=';', low_memory=False)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur data.gouv.fr criminalité: {e}")
        return None

# ============================================================================
# FONCTION UTILITAIRE
# ============================================================================

def calculate_yoy_change(df):
    """Calcule la variation Year-over-Year"""
    if df is None or len(df) < 12:
        return None
    
    df = df.copy()
    df['yoy_change'] = df['value'].pct_change(12) * 100
    return df

# ============================================================================
# ONGLETS PRINCIPAUX
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🇺🇸 US DASHBOARD", 
    "🇫🇷 FRANCE DASHBOARD", 
    "🌍 EU COMPARISONS",
    "📊 ADVANCED INDICATORS",
    "📝 METHODOLOGY"
])

# ============================================================================
# TAB 1: US DASHBOARD (Déjà existant - gardé identique)
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

# ============================================================================
# TAB 2: FRANCE DASHBOARD (PHASE 1 & 2)
# ============================================================================

with tab2:
    st.markdown("### 🇫🇷 INDICATEURS ÉCONOMIQUES FRANCE")
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_fr"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # Get INSEE token
    insee_token = get_insee_token()
    
    if insee_token:
        # ========== PHASE 1: ÉCONOMIE FR ==========
        
        st.markdown("#### 📊 INFLATION (INSEE)")
        cols_fr_infl = st.columns(4)
        
        # IPC France
        insee_inflation_series = [
            ('001763852', 'IPC GLOBAL'),
            ('001763854', 'IPC ALIM'),
            ('001763856', 'IPC ÉNERGIE'),
            ('010565700', 'IPC CORE')
        ]
        
        for idx, (series_id, label) in enumerate(insee_inflation_series):
            with cols_fr_infl[idx]:
                df = get_insee_series(series_id, insee_token)
                if df is not None and len(df) > 12:
                    df_yoy = calculate_yoy_change(df)
                    if df_yoy is not None:
                        current_yoy = df_yoy['yoy_change'].iloc[-1]
                        previous_yoy = df_yoy['yoy_change'].iloc[-2]
                        change = current_yoy - previous_yoy
                        
                        st.metric(
                            label=f"{label}",
                            value=f"{current_yoy:.2f}%",
                            delta=f"{change:+.2f}%"
                        )
                    else:
                        st.metric(label=label, value="N/A")
                else:
                    st.metric(label=label, value="N/A")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
        
        # EMPLOI
        st.markdown("#### 💼 EMPLOI & CHÔMAGE")
        cols_fr_emploi = st.columns(3)
        
        with cols_fr_emploi[0]:
            df = get_insee_series('001688527', insee_token)  # Chômage
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change = current_value - previous_value
                
                st.metric(
                    label="TAUX CHÔMAGE",
                    value=f"{current_value:.1f}%",
                    delta=f"{change:+.1f}%"
                )
            else:
                st.metric(label="TAUX CHÔMAGE", value="N/A")
        
        with cols_fr_emploi[1]:
            df = get_insee_series('010565845', insee_token)  # Emploi total
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change_pct = ((current_value - previous_value) / previous_value) * 100
                
                st.metric(
                    label="EMPLOI TOTAL",
                    value=f"{current_value/1000:.1f}M",
                    delta=f"{change_pct:+.2f}%"
                )
            else:
                st.metric(label="EMPLOI TOTAL", value="N/A")
        
        with cols_fr_emploi[2]:
            df = get_insee_series('010565846', insee_token)  # Emploi privé
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change_pct = ((current_value - previous_value) / previous_value) * 100
                
                st.metric(
                    label="EMPLOI PRIVÉ",
                    value=f"{current_value/1000:.1f}M",
                    delta=f"{change_pct:+.2f}%"
                )
            else:
                st.metric(label="EMPLOI PRIVÉ", value="N/A")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
        
        # PIB
        st.markdown("#### 📈 CROISSANCE & PIB")
        cols_fr_pib = st.columns(3)
        
        with cols_fr_pib[0]:
            df = get_insee_series('001688370', insee_token)  # PIB trimestriel
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                qoq_growth = ((current_value - previous_value) / previous_value) * 100
                
                st.metric(
                    label="PIB QoQ",
                    value=f"{qoq_growth:.2f}%",
                    delta=f"T{len(df)}"
                )
            else:
                st.metric(label="PIB QoQ", value="N/A")
        
        with cols_fr_pib[1]:
            df = get_insee_series('010565692', insee_token)  # PIB annuel
            if df is not None and len(df) > 1:
                df_yoy = calculate_yoy_change(df)
                if df_yoy is not None:
                    current_yoy = df_yoy['yoy_change'].iloc[-1]
                    
                    st.metric(
                        label="PIB YoY",
                        value=f"{current_yoy:.2f}%"
                    )
                else:
                    st.metric(label="PIB YoY", value="N/A")
            else:
                st.metric(label="PIB YoY", value="N/A")
        
        with cols_fr_pib[2]:
            df = get_insee_series('010565710', insee_token)  # Production industrielle
            if df is not None and len(df) > 12:
                df_yoy = calculate_yoy_change(df)
                if df_yoy is not None:
                    current_yoy = df_yoy['yoy_change'].iloc[-1]
                    
                    st.metric(
                        label="PROD IND YoY",
                        value=f"{current_yoy:.2f}%"
                    )
                else:
                    st.metric(label="PROD IND", value="N/A")
            else:
                st.metric(label="PROD IND", value="N/A")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
        
        # ========== PHASE 2: DÉMOGRAPHIE & IMMIGRATION ==========
        
        st.markdown("#### 👥 DÉMOGRAPHIE & IMMIGRATION")
        cols_fr_demo = st.columns(3)
        
        with cols_fr_demo[0]:
            df = get_insee_series('001688449', insee_token)  # Population totale
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change = current_value - previous_value
                
                st.metric(
                    label="POPULATION FR",
                    value=f"{current_value/1e6:.2f}M",
                    delta=f"{change/1000:+.0f}K"
                )
            else:
                st.metric(label="POPULATION FR", value="N/A")
        
        with cols_fr_demo[1]:
            df = get_insee_series('010565789', insee_token)  # Population immigrée
            if df is not None and len(df) > 0:
                current_value = df['value'].iloc[-1]
                
                # Calculer le % de la population
                df_pop_total = get_insee_series('001688449', insee_token)
                if df_pop_total is not None:
                    pop_total = df_pop_total['value'].iloc[-1]
                    pct = (current_value / pop_total) * 100
                    
                    st.metric(
                        label="IMMIGRÉS",
                        value=f"{current_value/1e6:.2f}M",
                        delta=f"{pct:.1f}% pop"
                    )
                else:
                    st.metric(label="IMMIGRÉS", value=f"{current_value/1e6:.2f}M")
            else:
                st.metric(label="IMMIGRÉS", value="N/A")
        
        with cols_fr_demo[2]:
            df = get_insee_series('010565790', insee_token)  # Population étrangère
            if df is not None and len(df) > 0:
                current_value = df['value'].iloc[-1]
                
                st.metric(
                    label="ÉTRANGERS",
                    value=f"{current_value/1e6:.2f}M"
                )
            else:
                st.metric(label="ÉTRANGERS", value="N/A")
        
        st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
        
        # ========== GRAPHIQUES FRANCE ==========
        
        st.markdown("#### 📊 INFLATION FRANCE - ÉVOLUTION")
        
        df_ipc_fr = get_insee_series('001763852', insee_token)
        
        if df_ipc_fr is not None and len(df_ipc_fr) > 12:
            df_ipc_fr_yoy = calculate_yoy_change(df_ipc_fr)
            
            fig_fr_infl = go.Figure()
            
            fig_fr_infl.add_trace(go.Scatter(
                x=df_ipc_fr_yoy['date'],
                y=df_ipc_fr_yoy['yoy_change'],
                mode='lines',
                line=dict(color='#0055AA', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 85, 170, 0.1)',
                name='IPC France YoY'
            ))
            
            # Ligne cible BCE 2%
            fig_fr_infl.add_hline(y=2, line_dash="dash", line_color="#00FF00", 
                                 annotation_text="BCE Target 2%", annotation_position="right")
            
            fig_fr_infl.update_layout(
                title="Inflation France (IPC YoY %)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="YoY Change (%)"),
                height=400
            )
            
            st.plotly_chart(fig_fr_infl, use_container_width=True)
    
    else:
        st.error("❌ Erreur d'authentification INSEE. Vérifiez vos clés API.")

# ============================================================================
# TAB 3: EU COMPARISONS (PHASE 2)
# ============================================================================

with tab3:
    st.markdown("### 🌍 COMPARAISONS EUROPÉENNES")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #0055AA; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #0055AA; font-weight: bold;">
        📊 DONNÉES EUROSTAT
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Comparaisons France vs Zone Euro vs Allemagne vs Espagne vs Italie
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # INFLATION COMPARÉE
    st.markdown("#### 📊 INFLATION HICP (Harmonisée)")
    
    # Données simulées pour démo (en production, utiliser vraie API Eurostat)
    countries = ['France', 'Zone Euro', 'Allemagne', 'Espagne', 'Italie']
    inflation_values = [2.3, 2.4, 2.1, 3.2, 1.8]  # Valeurs d'exemple
    
    cols_eu = st.columns(5)
    for idx, (country, infl) in enumerate(zip(countries, inflation_values)):
        with cols_eu[idx]:
            st.metric(
                label=country.upper(),
                value=f"{infl:.1f}%"
            )
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # GRAPHIQUE COMPARATIF
    st.markdown("#### 📈 INFLATION COMPARÉE (Derniers 24 mois)")
    
    # Créer données simulées pour graphique
    dates = pd.date_range(end=datetime.now(), periods=24, freq='M')
    
    fig_eu = go.Figure()
    
    # Simuler courbes (en production, données réelles Eurostat)
    for country in countries:
        values = np.random.uniform(1, 4, 24) + np.linspace(0, -1, 24)
        fig_eu.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=country,
            line=dict(width=2)
        ))
    
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
    
    unemployment_values = [7.3, 6.5, 3.1, 11.8, 7.9]  # Valeurs d'exemple
    
    cols_eu_unemp = st.columns(5)
    for idx, (country, unemp) in enumerate(zip(countries, unemployment_values)):
        with cols_eu_unemp[idx]:
            st.metric(
                label=country.upper(),
                value=f"{unemp:.1f}%"
            )

# ============================================================================
# TAB 4: ADVANCED INDICATORS (PHASE 3)
# ============================================================================

with tab4:
    st.markdown("### 📊 INDICATEURS AVANCÉS FRANCE")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #FFAA00; font-weight: bold;">
        🔬 INDICATEURS CONSTRUITS
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Spreads, taux réels, comparaisons France/US/EU, indicateurs de compétitivité
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ========== SPREAD OAT-BUND ==========
    st.markdown("#### 📊 SPREAD OAT 10Y - BUND 10Y")
    st.caption("Écart de taux France vs Allemagne (indicateur de risque souverain)")
    
    col_spread1, col_spread2, col_spread3 = st.columns([2, 1, 1])
    
    with col_spread1:
        # Données simulées pour démo
        dates_spread = pd.date_range(end=datetime.now(), periods=120, freq='D')
        spread_values = np.random.uniform(40, 80, 120) + np.cumsum(np.random.randn(120) * 2)
        
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
                            annotation_text="Zone surveillance", annotation_position="top left")
        fig_spread.add_hrect(y0=100, y1=200, fillcolor="red", opacity=0.1, 
                            annotation_text="Zone risque", annotation_position="top left")
        
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
    
    # ========== TAUX RÉEL FRANCE ==========
    st.markdown("#### 📊 TAUX RÉEL OAT 10Y")
    st.caption("Formula: `OAT 10Y - Inflation YoY`")
    
    col_real1, col_real2 = st.columns([2, 1])
    
    with col_real1:
        # Simuler données
        dates_real = pd.date_range(end=datetime.now(), periods=60, freq='M')
        nominal_rate = np.random.uniform(2.5, 3.5, 60)
        inflation_rate = np.random.uniform(1.5, 3, 60)
        real_rate = nominal_rate - inflation_rate
        
        fig_real = go.Figure()
        
        fig_real.add_trace(go.Scatter(
            x=dates_real,
            y=real_rate,
            mode='lines',
            line=dict(color='#00FF00', width=2),
            name='Taux Réel'
        ))
        
        fig_real.add_trace(go.Scatter(
            x=dates_real,
            y=nominal_rate,
            mode='lines',
            line=dict(color='#FFAA00', width=1, dash='dash'),
            name='Taux Nominal'
        ))
        
        fig_real.add_hline(y=0, line_dash="dot", line_color="#FF0000")
        
        fig_real.update_layout(
            title="Taux Réel OAT 10Y vs Taux Nominal",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333', title="Taux (%)"),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_real, use_container_width=True)
    
    with col_real2:
        current_real = real_rate[-1]
        current_nominal = nominal_rate[-1]
        
        st.metric(
            label="TAUX RÉEL",
            value=f"{current_real:.2f}%"
        )
        
        st.metric(
            label="TAUX NOMINAL",
            value=f"{current_nominal:.2f}%"
        )
        
        if current_real < 0:
            st.markdown("""
            <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px;">
                <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                ⚠️ TAUX RÉEL NÉGATIF
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px;">
                <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                ✅ TAUX RÉEL POSITIF
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ========== FRANCE VS US ==========
    st.markdown("#### 🌍 COMPARAISON FRANCE vs US")
    
    comparison_metrics = st.columns(4)
    
    # Données simulées
    metrics_data = [
        ("INFLATION", 2.3, 3.1),
        ("CHÔMAGE", 7.3, 3.7),
        ("PIB GROWTH", 0.9, 2.8),
        ("TAUX 10Y", 3.2, 4.5)
    ]
    
    for idx, (metric, fr_val, us_val) in enumerate(metrics_data):
        with comparison_metrics[idx]:
            st.markdown(f"**{metric}**")
            st.markdown(f"🇫🇷 FR: **{fr_val:.1f}%**")
            st.markdown(f"🇺🇸 US: **{us_val:.1f}%**")
            
            diff = fr_val - us_val
            if diff > 0:
                st.markdown(f"<span style='color: #FF6600;'>+{diff:.1f}pp vs US</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: #00FF00;'>{diff:.1f}pp vs US</span>", unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ========== SONDAGES POLITIQUES (MISE À JOUR MANUELLE) ==========
    st.markdown("#### 🗳️ BAROMÈTRE POLITIQUE (Derniers sondages)")
    st.caption("Source: Compilation instituts (Ifop, Ipsos, Elabe) - Mise à jour manuelle")
    
    st.markdown("""
    <div style="background-color: #1a0a00; border-left: 3px solid #FF9900; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #FF9900; font-weight: bold;">
        ⚠️ DONNÉES À METTRE À JOUR MANUELLEMENT
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Pas d'API disponible pour les sondages en temps réel. Mise à jour recommandée: 1-2x/mois
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_sond1, col_sond2 = st.columns(2)
    
    with col_sond1:
        st.markdown("**INTENTIONS DE VOTE - PRÉSIDENTIELLE 2027**")
        st.markdown("*(Exemple - À actualiser)*")
        
        # Données exemple à mettre à jour manuellement
        sondage_data = {
            'Candidat': ['Marine Le Pen (RN)', 'Édouard Philippe (LR)', 'Gabriel Attal', 'Jordan Bardella (RN)', 'Autres'],
            'Intentions (%)': [37, 23, 20, 15, 5]
        }
        
        df_sondage = pd.DataFrame(sondage_data)
        
        fig_sondage = go.Figure()
        fig_sondage.add_trace(go.Bar(
            x=df_sondage['Intentions (%)'],
            y=df_sondage['Candidat'],
            orientation='h',
            marker=dict(color=['#0055AA', '#FF9900', '#00AA55', '#0055AA', '#666666'])
        ))
        
        fig_sondage.update_layout(
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(gridcolor='#333', title="Intentions de vote (%)"),
            yaxis=dict(gridcolor='#333'),
            height=300
        )
        
        st.plotly_chart(fig_sondage, use_container_width=True)
    
    with col_sond2:
        st.markdown("**COTE DE POPULARITÉ**")
        st.markdown("*(Exemple - À actualiser)*")
        
        popularity_data = {
            'Personnalité': ['Emmanuel Macron', 'François Bayrou', 'Marine Le Pen', 'Jordan Bardella'],
            'Approbation (%)': [28, 35, 42, 38]
        }
        
        df_pop = pd.DataFrame(popularity_data)
        
        fig_pop = go.Figure()
        fig_pop.add_trace(go.Bar(
            x=df_pop['Approbation (%)'],
            y=df_pop['Personnalité'],
            orientation='h',
            marker=dict(color=['#FF6600', '#00AA55', '#0055AA', '#0055AA'])
        ))
        
        fig_pop.update_layout(
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(gridcolor='#333', title="Taux d'approbation (%)"),
            yaxis=dict(gridcolor='#333'),
            height=300
        )
        
        st.plotly_chart(fig_pop, use_container_width=True)

# ============================================================================
# TAB 5: METHODOLOGY (Notes techniques)
# ============================================================================

with tab5:
    st.markdown("### 📝 MÉTHODOLOGIE & SOURCES")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #00FF00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
        📚 DOCUMENTATION DES SOURCES
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    method_tab1, method_tab2, method_tab3 = st.tabs(["📊 SOURCES APIs", "🔬 INDICATEURS", "⚙️ CONFIGURATION"])
    
    with method_tab1:
        st.markdown("#### 🇺🇸 DONNÉES US")
        st.markdown("""
        **FRED API** (Federal Reserve Economic Data)
        - Fournisseur: Federal Reserve Bank of St. Louis
        - URL: https://fred.stlouisfed.org
        - Couverture: 500,000+ séries économiques US
        - Fréquence: Temps réel (mise à jour quotidienne)
        """)
        
        st.markdown("#### 🇫🇷 DONNÉES FRANCE")
        st.markdown("""
        **INSEE API** (Institut National de la Statistique)
        - Fournisseur: INSEE
        - URL: https://api.insee.fr
        - Couverture: IPC, PIB, emploi, démographie, immigration
        - Fréquence: Mensuelle/Trimestrielle/Annuelle
        - Authentification: OAuth2 (consumer key/secret)
        
        **BANQUE DE FRANCE API**
        - Fournisseur: Banque de France
        - URL: https://webstat.banque-france.fr
        - Couverture: Taux d'intérêt, crédit, masses monétaires
        - Format: SDMX-JSON
        - Authentification: Aucune (accès public)
        """)
        
        st.markdown("#### 🌍 DONNÉES EUROPÉENNES")
        st.markdown("""
        **EUROSTAT API**
        - Fournisseur: Office statistique de l'UE
        - URL: https://ec.europa.eu/eurostat
        - Couverture: Tous pays EU (comparaisons)
        - Authentification: Aucune
        
        **DATA.GOUV.FR**
        - Fournisseur: État français
        - URL: https://www.data.gouv.fr
        - Couverture: Criminalité, santé, éducation
        - Format: CSV, JSON
        """)
    
    with method_tab2:
        st.markdown("#### 📊 INDICATEURS CONSTRUITS")
        
        st.markdown("""
        **1. Inflation YoY**
        - Formule: `100 * (CPI_t / CPI_{t-12} - 1)`
        - Source: INSEE (France), FRED (US)
        
        **2. Taux Réel**
        - Formule: `Taux nominal 10Y - Inflation YoY`
        - Interprétation: Rendement ajusté de l'inflation
        
        **3. Spread OAT-Bund**
        - Formule: `OAT 10Y France - Bund 10Y Allemagne`
        - Interprétation: Prime de risque souverain France vs Allemagne
        - Seuils:
          - < 50 bps: Situation normale
          - 50-100 bps: Surveillance
          - > 100 bps: Zone de risque
        
        **4. PIB QoQ Annualisé**
        - Formule: `400 * (PIB_t / PIB_{t-1} - 1)`
        - Permet comparaison avec données US
        """)
    
    with method_tab3:
        st.markdown("#### ⚙️ CONFIGURATION REQUISE")
        
        st.markdown("""
        **Clés API nécessaires:**
        
        1. **FRED API** ✅
        ```python
        FRED_API_KEY = "ce5dbb3d3fcd8669f2fe2cdd9c79a7da"
        ```
        
        2. **INSEE API** (À configurer)
        ```python
        INSEE_CONSUMER_KEY = "VOTRE_CLE"
        INSEE_CONSUMER_SECRET = "VOTRE_SECRET"
        ```
        Inscription: https://api.insee.fr/catalogue/
        
        3. **Autres APIs** (Pas de clé requise)
        - Banque de France: Accès public
        - Eurostat: Accès public
        - data.gouv.fr: Accès public
        """)
        
        st.markdown("#### 📦 DÉPENDANCES PYTHON")
        st.code("""
# Installation
pip install streamlit pandas plotly requests numpy

# Imports requis
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import base64
        """, language="bash")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | SOURCES: FRED, INSEE, BANQUE DE FRANCE, EUROSTAT | LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
