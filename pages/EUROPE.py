import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Europe Economic Data",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    
    .api-info-box {
        background-color: #0a0a0a;
        border-left: 3px solid #00FFFF;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    
    .country-flag {
        font-size: 24px;
        margin-right: 10px;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - EUROPEAN ECONOMIC DATA</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonctions API ECB
@st.cache_data(ttl=3600)
def get_ecb_data(series_key, start_date=None):
    """Récupère données ECB Statistical Data Warehouse"""
    try:
        # Format: https://sdw-wsrest.ecb.europa.eu/service/data/{flow}/{key}
        base_url = "https://sdw-wsrest.ecb.europa.eu/service/data"
        
        params = {
            'format': 'jsondata',
            'detail': 'dataonly'
        }
        
        if start_date:
            params['startPeriod'] = start_date
        
        url = f"{base_url}/{series_key}"
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parser la structure ECB
            if 'dataSets' in data and len(data['dataSets']) > 0:
                observations = data['dataSets'][0].get('series', {})
                
                if observations:
                    # Extraire les données
                    series_data = list(observations.values())[0].get('observations', {})
                    
                    dates = []
                    values = []
                    
                    for key, value in series_data.items():
                        dates.append(data['structure']['dimensions']['observation'][0]['values'][int(key)]['id'])
                        values.append(value[0])
                    
                    df = pd.DataFrame({
                        'date': pd.to_datetime(dates),
                        'value': values
                    })
                    
                    return df.sort_values('date')
        
        return None
        
    except Exception as e:
        st.error(f"ECB API Error: {e}")
        return None

# Fonctions Eurostat
@st.cache_data(ttl=3600)
def get_eurostat_data(dataset_code, filters=None):
    """Récupère données Eurostat"""
    try:
        base_url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_code}"
        
        params = {
            'format': 'JSON',
            'lang': 'EN'
        }
        
        if filters:
            params.update(filters)
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parser structure Eurostat (complexe)
            # Simplifié pour démo
            return data
        
        return None
        
    except Exception as e:
        st.error(f"Eurostat API Error: {e}")
        return None

# ONGLETS PRINCIPAUX
tab1, tab2, tab3 = st.tabs(["🇪🇺 EUROPEAN UNION", "🇫🇷 FRANCE", "🇨🇭 SWITZERLAND"])

# ===== TAB 1: UNION EUROPÉENNE =====
with tab1:
    st.markdown("### 🇪🇺 EUROPEAN UNION ECONOMIC INDICATORS")
    
    st.markdown("""
    <div class="api-info-box">
        <p style="margin: 0; font-size: 10px; color: #00FFFF; font-weight: bold;">
        📊 DATA SOURCES
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>ECB SDW:</strong> European Central Bank Statistical Data Warehouse</li>
            <li><strong>Eurostat:</strong> Statistical Office of the European Union</li>
            <li><strong>Bank of England:</strong> UK Economic Data (Brexit context)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Refresh button
    col_eu1, col_eu2 = st.columns([5, 1])
    with col_eu2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_eu"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ===== KEY INDICATORS DASHBOARD =====
    st.markdown("#### 📊 KEY EUROZONE INDICATORS")
    
    # Série ECB clés (simplifiées pour démo)
    ecb_series = {
        'ICP': 'HICP - Harmonized Index Consumer Prices',
        'MRO': 'Main Refinancing Operations Rate',
        'GDP': 'GDP Eurozone',
        'UNEMP': 'Unemployment Rate'
    }
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 ECB STATISTICAL DATA WAREHOUSE
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Access:</strong> https://sdw.ecb.europa.eu/
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API Endpoint:</strong> https://sdw-wsrest.ecb.europa.eu/service/data/
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Series Examples:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><code>ICP/M.U2.N.000000.4.ANR</code> - HICP Inflation YoY</li>
            <li><code>FM/D.U2.EUR.4F.KR.MRR_FR.LEV</code> - ECB Main Refinancing Rate</li>
            <li><code>MNA/Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N</code> - GDP Eurozone</li>
            <li><code>STS/M.I8.Y.UNEH.RTT000.4.000</code> - Unemployment Rate</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Démonstration avec indicateurs simulés
    st.markdown("#### 📊 EUROZONE DASHBOARD (DEMO)")
    
    cols_eu = st.columns(4)
    
    demo_indicators = [
        ('HICP INFLATION', '2.4%', '+0.2%', 'YoY'),
        ('ECB RATE', '4.50%', '0.0%', 'Deposit Facility'),
        ('UNEMPLOYMENT', '6.5%', '-0.1%', 'Eurozone'),
        ('GDP GROWTH', '0.5%', '+0.1%', 'QoQ')
    ]
    
    for idx, (label, value, delta, caption) in enumerate(demo_indicators):
        with cols_eu[idx]:
            st.metric(label, value, delta, help=caption)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== COUNTRY BREAKDOWN =====
    st.markdown("#### 🌍 EUROZONE COUNTRY BREAKDOWN")
    
    eurozone_countries = {
        '🇩🇪 Germany': {'GDP Weight': '29.2%', 'Inflation': '2.7%', 'Unemployment': '3.0%'},
        '🇫🇷 France': {'GDP Weight': '20.1%', 'Inflation': '2.4%', 'Unemployment': '7.3%'},
        '🇮🇹 Italy': {'GDP Weight': '15.2%', 'Inflation': '1.9%', 'Unemployment': '7.8%'},
        '🇪🇸 Spain': {'GDP Weight': '10.5%', 'Inflation': '3.1%', 'Unemployment': '11.8%'},
        '🇳🇱 Netherlands': {'GDP Weight': '6.8%', 'Inflation': '2.8%', 'Unemployment': '3.6%'},
        '🇧🇪 Belgium': {'GDP Weight': '4.0%', 'Inflation': '2.3%', 'Unemployment': '5.5%'},
        '🇦🇹 Austria': {'GDP Weight': '3.3%', 'Inflation': '3.4%', 'Unemployment': '5.1%'},
        '🇮🇪 Ireland': {'GDP Weight': '3.2%', 'Inflation': '2.0%', 'Unemployment': '4.2%'}
    }
    
    country_df = pd.DataFrame(eurozone_countries).T.reset_index()
    country_df.columns = ['Country', 'GDP Weight', 'Inflation', 'Unemployment']
    
    st.dataframe(country_df, use_container_width=True, hide_index=True)
    
    # ===== EUROSTAT INTEGRATION =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 EUROSTAT DATA ACCESS")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 EUROSTAT API
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> https://ec.europa.eu/eurostat
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API:</strong> https://ec.europa.eu/eurostat/api/dissemination/
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Popular Datasets:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><code>prc_hicp_midx</code> - HICP Monthly Index</li>
            <li><code>namq_10_gdp</code> - GDP and Main Components</li>
            <li><code>une_rt_m</code> - Unemployment by Sex and Age</li>
            <li><code>sts_inpr_m</code> - Industrial Production Index</li>
            <li><code>ert_bil_eur_m</code> - Bilateral Exchange Rates</li>
        </ul>
        <p style="margin: 10px 0 0 0; font-size: 9px; color: #00FFFF;">
        💡 No API key required - Open data!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    eurostat_dataset = st.selectbox(
        "SELECT EUROSTAT DATASET",
        options=[
            'prc_hicp_midx - HICP Inflation',
            'namq_10_gdp - GDP Quarterly',
            'une_rt_m - Unemployment Monthly',
            'sts_inpr_m - Industrial Production'
        ],
        key="eurostat_dataset"
    )
    
    if st.button("📊 FETCH EUROSTAT DATA", use_container_width=True, key="fetch_eurostat"):
        st.info("""
        **🔧 IMPLEMENTATION NOTE:**
        
        Eurostat API returns complex JSON structures that need parsing.
        
        **Python Example:**
```python
        import requests
        
        dataset = 'prc_hicp_midx'
        url = f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}'
        
        params = {
            'format': 'JSON',
            'lang': 'EN',
            'geo': 'EU27_2020',  # EU27
            'coicop': 'CP00'      # All items
        }
        
        response = requests.get(url, params=params)
        data = response.json()
```
        
        **Recommended Library:** `eurostat` package
```bash
        pip install eurostat
```
```python
        import eurostat
        df = eurostat.get_data_df('prc_hicp_midx')
```
        """)

# ===== TAB 2: FRANCE =====
with tab2:
    st.markdown("### 🇫🇷 FRENCH ECONOMIC INDICATORS")
    
    st.markdown("""
    <div class="api-info-box">
        <p style="margin: 0; font-size: 10px; color: #00FFFF; font-weight: bold;">
        📊 DATA SOURCES
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>INSEE:</strong> Institut national de la statistique et des études économiques</li>
            <li><strong>Banque de France:</strong> French Central Bank</li>
            <li><strong>OECD:</strong> Organisation for Economic Co-operation and Development</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_fr1, col_fr2 = st.columns([5, 1])
    with col_fr2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_fr"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ===== KEY FRENCH INDICATORS =====
    st.markdown("#### 📊 KEY FRENCH INDICATORS (DEMO)")
    
    cols_fr = st.columns(4)
    
    fr_indicators = [
        ('GDP GROWTH', '0.4%', '+0.1%', 'QoQ'),
        ('INFLATION', '2.9%', '+0.3%', 'YoY'),
        ('UNEMPLOYMENT', '7.3%', '-0.2%', 'ILO'),
        ('TRADE BALANCE', '-€8.2B', '-€1.1B', 'Monthly')
    ]
    
    for idx, (label, value, delta, caption) in enumerate(fr_indicators):
        with cols_fr[idx]:
            st.metric(label, value, delta, help=caption)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== INSEE API =====
    st.markdown("#### 📊 INSEE API ACCESS")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 INSEE API
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> https://www.insee.fr/
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API Documentation:</strong> https://api.insee.fr/catalogue/
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Setup Required:</strong>
        </p>
        <ol style="margin: 5px 0; font-size: 9px; color: #999;">
            <li>Create account: https://api.insee.fr/</li>
            <li>Subscribe to "Sirene" and "BDM" APIs (free)</li>
            <li>Get Consumer Key & Secret</li>
            <li>Generate Bearer token</li>
        </ol>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Series (BDM - Banque de Données Macro):</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><code>001688527</code> - PIB Trimestriel (GDP)</li>
            <li><code>001759970</code> - IPC (CPI)</li>
            <li><code>001688613</code> - Taux de Chômage (Unemployment)</li>
            <li><code>001769682</code> - Production Industrielle (Industrial Production)</li>
            <li><code>001641151</code> - Balance Commerciale (Trade Balance)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # API Setup
    with st.expander("🔧 INSEE API SETUP", expanded=False):
        st.markdown("""
        **Code Example:**
```python
        import requests
        import base64
        
        # 1. Get Bearer Token
        consumer_key = "YOUR_KEY"
        consumer_secret = "YOUR_SECRET"
        
        credentials = f"{consumer_key}:{consumer_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        
        token_url = "https://api.insee.fr/token"
        headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(token_url, headers=headers, data=data)
        token = response.json()['access_token']
        
        # 2. Get Data
        series_id = "001688527"  # PIB
        url = f"https://api.insee.fr/series/BDM/V1/data/SERIES_BDM/{series_id}"
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        data = response.json()
```
        
        **Recommended Library:**
```bash
        pip install pynsee
```
```python
        from pynsee.macrodata import get_series
        
        df = get_series("001688527")  # PIB
```
        """)
    
    insee_series_select = st.selectbox(
        "SELECT INSEE SERIES",
        options=[
            '001688527 - PIB (GDP)',
            '001759970 - IPC (CPI)',
            '001688613 - Chômage (Unemployment)',
            '001769682 - Production Industrielle',
            '001641151 - Balance Commerciale'
        ],
        key="insee_series"
    )
    
    if st.button("📊 FETCH INSEE DATA", use_container_width=True, key="fetch_insee"):
        st.warning("⚠️ Requires INSEE API credentials (free account)")
        st.info("""
        **Quick Start:**
        1. Create free account at: https://api.insee.fr/
        2. Subscribe to BDM API
        3. Get your credentials
        4. Use `pynsee` library for easy access
        """)
    
    # ===== SECTORAL DATA =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🏭 FRENCH SECTORAL INDICATORS")
    
    french_sectors = {
        'Industry': {'Weight': '13.5%', 'Growth': '+0.3%', 'Employment': '2.8M'},
        'Construction': {'Weight': '5.8%', 'Growth': '-0.1%', 'Employment': '1.5M'},
        'Services': {'Weight': '78.9%', 'Growth': '+0.5%', 'Employment': '18.2M'},
        'Agriculture': {'Weight': '1.8%', 'Growth': '-0.2%', 'Employment': '0.7M'}
    }
    
    sector_df = pd.DataFrame(french_sectors).T.reset_index()
    sector_df.columns = ['Sector', 'GDP Weight', 'QoQ Growth', 'Employment']
    
    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    
    # ===== CAC 40 LINK =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 📈 CAC 40 & FRENCH MARKETS")
    
    col_cac1, col_cac2 = st.columns(2)
    
    with col_cac1:
        st.metric("CAC 40", "7,521", "+1.2%")
        st.caption("Paris Stock Exchange")
    
    with col_cac2:
        st.metric("OAT 10Y", "3.12%", "+0.05%")
        st.caption("French Government Bond")
    
    st.markdown("""
    **🔗 Market Data Sources:**
    - **Yahoo Finance:** `^FCHI` (CAC 40)
    - **Euronext:** Official exchange data
    - **Banque de France:** Bond yields and credit data
    """)

# ===== TAB 3: SWITZERLAND =====
with tab3:
    st.markdown("### 🇨🇭 SWISS ECONOMIC INDICATORS")
    
    st.markdown("""
    <div class="api-info-box">
        <p style="margin: 0; font-size: 10px; color: #00FFFF; font-weight: bold;">
        📊 DATA SOURCES
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>Opendata.swiss:</strong> Swiss Federal Statistical Office (FSO/BFS)</li>
            <li><strong>KOF:</strong> KOF Swiss Economic Institute (ETH Zurich)</li>
            <li><strong>SNB:</strong> Swiss National Bank</li>
            <li><strong>SECO:</strong> State Secretariat for Economic Affairs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_ch1, col_ch2 = st.columns([5, 1])
    with col_ch2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_ch"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ===== KEY SWISS INDICATORS =====
    st.markdown("#### 📊 KEY SWISS INDICATORS (DEMO)")
    
    cols_ch = st.columns(4)
    
    ch_indicators = [
        ('GDP GROWTH', '0.3%', '+0.1%', 'QoQ'),
        ('INFLATION', '1.4%', '-0.2%', 'YoY'),
        ('UNEMPLOYMENT', '2.0%', '+0.1%', 'SECO'),
        ('KOF BAROMETER', '101.2', '+1.8', 'Leading Indicator')
    ]
    
    for idx, (label, value, delta, caption) in enumerate(ch_indicators):
        with cols_ch[idx]:
            st.metric(label, value, delta, help=caption)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== OPENDATA.SWISS =====
    st.markdown("#### 📊 OPENDATA.SWISS")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 OPENDATA.SWISS PORTAL
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> https://opendata.swiss/
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API Store:</strong> https://digital-public-services-switzerland.ch/api-store
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Datasets:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>BFS/FSO:</strong> Official statistics (GDP, Population, Employment)</li>
            <li><strong>SECO:</strong> Economic indicators (Unemployment, Business cycles)</li>
            <li><strong>SNB:</strong> Monetary policy, Exchange rates</li>
            <li><strong>Regional Data:</strong> Cantonal statistics</li>
        </ul>
        <p style="margin: 10px 0 0 0; font-size: 9px; color: #00FFFF;">
        💡 Most datasets are CSV/JSON downloadable - No API key required!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example datasets
    swiss_datasets = [
        {'ID': 'je-e-06.02.01.01', 'Name': 'Unemployment Rate', 'Source': 'SECO', 'Frequency': 'Monthly'},
        {'ID': 'px-x-0602010000_101', 'Name': 'GDP Quarterly', 'Source': 'BFS/FSO', 'Frequency': 'Quarterly'},
        {'ID': 'cc-e-05.02.55', 'Name': 'Consumer Price Index', 'Source': 'BFS/FSO', 'Frequency': 'Monthly'},
        {'ID': 'je-e-06.02.02.07', 'Name': 'Job Vacancies', 'Source': 'SECO', 'Frequency': 'Quarterly'},
        {'ID': 'su-e-11.03.03', 'Name': 'Production Index', 'Source': 'BFS/FSO', 'Frequency': 'Monthly'}
    ]
    
    datasets_df = pd.DataFrame(swiss_datasets)
    st.dataframe(datasets_df, use_container_width=True, hide_index=True)
    
    st.caption("**Access:** Most datasets available as direct CSV/JSON downloads from opendata.swiss")
    
    # ===== KOF DATA =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 KOF SWISS ECONOMIC INSTITUTE")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 KOF DATENSERVICE (ETH Zurich)
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> https://kof.ethz.ch/en/forecasts-and-indicators.html
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>Data Portal:</strong> https://kof.ethz.ch/en/data.html
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Indicators:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>KOF Economic Barometer:</strong> Leading indicator for Swiss economy</li>
            <li><strong>KOF Employment Indicator:</strong> Labor market outlook</li>
            <li><strong>KOF Consensus Forecasts:</strong> GDP, Inflation predictions</li>
            <li><strong>KOF Business Surveys:</strong> Sectoral confidence indices</li>
            <li><strong>KOF Globalisation Index:</strong> Economic globalization measure</li>
        </ul>
        <p style="margin: 10px 0 0 0; font-size: 9px; color: #00FFFF;">
        💡 Data available via web download (Excel/CSV) - No API but regular updates
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    kof_indicators = {
        'KOF Barometer': {'Current': 101.2, 'Previous': 99.4, 'Long-term Avg': 100.0},
        'Employment Indicator': {'Current': 9.8, 'Previous': 10.2, 'Long-term Avg': 10.5},
        'Manufacturing Confidence': {'Current': 5.2, 'Previous': 4.8, 'Long-term Avg': 0.0},
        'Services Confidence': {'Current': 12.5, 'Previous': 11.9, 'Long-term Avg': 10.0}
    }
    
    kof_df = pd.DataFrame(kof_indicators).T.reset_index()
    kof_df.columns = ['Indicator', 'Current', 'Previous', 'Long-term Average']
    
    st.dataframe(kof_df, use_container_width=True, hide_index=True)
    
    # KOF Barometer explanation
    with st.expander("📊 KOF ECONOMIC BAROMETER EXPLAINED", expanded=False):
        st.markdown("""
        **What is it?**
        - Composite leading indicator for Swiss economy
        - Predicts GDP growth 6-9 months ahead
        - Based on ~200 variables from surveys and statistics
        
        **Interpretation:**
        - **> 100:** Above-trend growth expected
        - **= 100:** Trend growth (long-term average)
        - **< 100:** Below-trend growth expected
        
        **Components:**
        - Manufacturing orders and production
        - Construction activity
        - Banking and insurance
        - Foreign demand
        - Consumer confidence
        
        **Publication:** Monthly (last working day)
        
        **Historical Accuracy:** High correlation with actual GDP
        """)
    
    # ===== SNB DATA =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 📊 SWISS NATIONAL BANK (SNB)")
    
    st.markdown("""
    <div style="background-color: #111; border: 1px solid #333; padding: 15px; margin: 10px 0;">
        <p style="margin: 0; font-size: 11px; color: #FFAA00; font-weight: bold;">
        📡 SNB DATA PORTAL
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #999;">
        <strong>Website:</strong> https://data.snb.ch/en
        </p>
        <p style="margin: 5px 0 0 0; font-size: 10px; color: #999;">
        <strong>API:</strong> SDMX Web Service available
        </p>
        <p style="margin: 10px 0 0 0; font-size: 10px; color: #FFAA00;">
        <strong>Key Series:</strong>
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li><strong>Interest Rates:</strong> SNB Policy Rate, SARON, Libor</li>
            <li><strong>Exchange Rates:</strong> CHF/EUR, CHF/USD, Trade-weighted CHF</li>
            <li><strong>Balance Sheet:</strong> Foreign reserves, Gold holdings</li>
            <li><strong>Credit:</strong> Bank lending, Mortgage volume</li>
            <li><strong>Money Supply:</strong> M1, M2, M3 aggregates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    cols_snb = st.columns(3)
    
    snb_data = [
        ('SNB POLICY RATE', '1.75%', '0.00%'),
        ('CHF/EUR', '0.93', '-0.01'),
        ('FOREIGN RESERVES', 'CHF 716B', '+CHF 5B')
    ]
    
    for idx, (label, value, delta) in enumerate(snb_data):
        with cols_snb[idx]:
            st.metric(label, value, delta)
    
    # SNB API Example
    with st.expander("🔧 SNB API SETUP", expanded=False):
        st.markdown("""
        **SDMX Web Service:**
```python
        import requests
        import pandas as pd
        
        # SNB SDMX endpoint
        base_url = "https://data.snb.ch/api/cube"
        
        # Example: Get CHF/EUR exchange rate
        dataset = "devkum"  # Daily exchange rates
        params = {
            'format': 'csv',
            'D0': 'EUR'  # EUR/CHF
        }
        
        response = requests.get(f"{base_url}/{dataset}", params=params)
        
        # Parse CSV
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
```
        
        **Alternative: Download directly from portal**
        - Navigate to https://data.snb.ch/en
        - Select series (e.g., Interest rates)
        - Download as CSV/Excel
        - No authentication required
        
        **Recommended for Python:**
```python
        # Use pandas to read SNB CSV directly
        url = "https://data.snb.ch/api/cube/devkum/data/csv/en"
        df = pd.read_csv(url)
```
        """)
    
    # ===== SWISS MARKET DATA =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 📈 SWISS MARKET INDICATORS")
    
    col_market1, col_market2, col_market3 = st.columns(3)
    
    with col_market1:
        st.metric("SMI INDEX", "11,234", "+0.8%")
        st.caption("Swiss Market Index")
    
    with col_market2:
        st.metric("SPI INDEX", "14,892", "+0.7%")
        st.caption("Swiss Performance Index")
    
    with col_market3:
        st.metric("SWISS GOVT 10Y", "0.85%", "+0.03%")
        st.caption("Confederation Bonds")
    
    st.markdown("""
    **🔗 Swiss Market Data Sources:**
    - **Yahoo Finance:** `^SSMI` (SMI), `^SSPI` (SPI)
    - **SIX Swiss Exchange:** Official market data
    - **SNB:** Bond yields and money market rates
    """)
    
    # ===== CANTONAL DATA =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🗺️ CANTONAL ECONOMIC DATA")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #FFAA00; font-weight: bold;">
        📊 REGIONAL BREAKDOWN AVAILABLE
        </p>
        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
        Swiss statistical data often includes cantonal-level detail for:
        </p>
        <ul style="margin: 5px 0; font-size: 9px; color: #999;">
            <li>GDP by Canton</li>
            <li>Unemployment rates (cantonal SECO data)</li>
            <li>Population and demographics</li>
            <li>Tax rates and fiscal data</li>
            <li>Real estate prices</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Example: Top cantons by GDP
    top_cantons = {
        '🏙️ Zürich': {'GDP (CHF B)': '145', 'Population (M)': '1.55', 'GDP per Capita': '94k'},
        '🏦 Geneva': {'GDP (CHF B)': '52', 'Population (M)': '0.51', 'GDP per Capita': '102k'},
        '🏔️ Bern': {'GDP (CHF B)': '82', 'Population (M)': '1.04', 'GDP per Capita': '79k'},
        '🏢 Vaud': {'GDP (CHF B)': '62', 'Population (M)': '0.82', 'GDP per Capita': '76k'},
        '🔬 Basel-Stadt': {'GDP (CHF B)': '40', 'Population (M)': '0.20', 'GDP per Capita': '200k'}
    }
    
    cantons_df = pd.DataFrame(top_cantons).T.reset_index()
    cantons_df.columns = ['Canton', 'GDP (CHF B)', 'Population (M)', 'GDP per Capita']
    
    st.dataframe(cantons_df, use_container_width=True, hide_index=True)
    st.caption("Source: BFS/FSO - Federal Statistical Office")
    
    # ===== COMPARISON TOOL =====
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔍 SWISS-EU COMPARISON")
    
    comparison_data = {
        'Indicator': ['GDP Growth (2023)', 'Inflation (Latest)', 'Unemployment', 'Debt/GDP', 'Current Account'],
        'Switzerland': ['0.9%', '1.4%', '2.0%', '38%', '+8.5%'],
        'Eurozone': ['0.5%', '2.4%', '6.5%', '91%', '+2.1%'],
        'France': ['0.9%', '2.9%', '7.3%', '111%', '-1.8%']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.caption("""
    **🔗 OECD Data for International Comparisons:**
    - Website: https://data.oecd.org/
    - API: https://data.oecd.org/api/
    - Free access, no key required
    - Standardized indicators across countries
    """)

# ===== INTEGRATION SECTION (All Tabs) =====
st.markdown('<div style="border-top: 2px solid #FFAA00; margin: 20px 0;"></div>', unsafe_allow_html=True)
st.markdown("### 🔗 DATA INTEGRATION & TOOLS")

integration_cols = st.columns(3)

with integration_cols[0]:
    st.markdown("#### 📊 QUICK LINKS")
    st.markdown("""
    **🇪🇺 European Data:**
    - [ECB SDW](https://sdw.ecb.europa.eu/)
    - [Eurostat](https://ec.europa.eu/eurostat)
    - [Bank of England](https://www.bankofengland.co.uk/statistics)
    
    **🇫🇷 French Data:**
    - [INSEE](https://www.insee.fr/)
    - [Banque de France](https://www.banque-france.fr/statistiques)
    - [OECD France](https://www.oecd.org/fr/france/)
    
    **🇨🇭 Swiss Data:**
    - [Opendata.swiss](https://opendata.swiss/)
    - [KOF ETH](https://kof.ethz.ch/)
    - [SNB Data](https://data.snb.ch/)
    - [BFS/FSO](https://www.bfs.admin.ch/)
    """)

with integration_cols[1]:
    st.markdown("#### 🛠️ PYTHON LIBRARIES")
    st.markdown("""
    **Recommended packages:**
```bash
    # European data
    pip install eurostat
    pip install pandas-datareader
    
    # French data
    pip install pynsee
    
    # Swiss data
    pip install pandas requests
    
    # General
    pip install wbdata  # World Bank
    pip install fredapi  # For comparisons
```
    """)

with integration_cols[2]:
    st.markdown("#### 📈 NEXT STEPS")
    st.markdown("""
    **To implement full integration:**
    
    1. **Get API credentials:**
       - INSEE (free account)
       - ECB SDW (no key needed)
       - SNB (no key needed)
    
    2. **Install libraries:**
       - See Python packages →
    
    3. **Test connections:**
       - Start with simple requests
       - Verify data format
    
    4. **Build dashboards:**
       - Combine multiple sources
       - Create comparative views
       - Add forecasting models
    """)

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | EUROPEAN ECONOMIC DATA | LAST UPDATE: {last_update}
    <br>
    Data sources: ECB, Eurostat, INSEE, BFS/FSO, KOF, SNB, OECD
</div>
""", unsafe_allow_html=True)
