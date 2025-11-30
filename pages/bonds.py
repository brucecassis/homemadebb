import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import base64
import json

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Bond Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG TERMINAL  
# =============================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .stApp {
        background-color: #000000 !important;
        transition: none !important;
    }
    
    .main {
        transition: none !important;
        animation: none !important;
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    section.main > div {
        animation: none !important;
        opacity: 1 !important;
    }
    
    .stApp [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
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
        font-size: 18px !important;
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
        font-size: 11px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        border: 2px solid #FFAA00 !important;
        padding: 6px 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-radius: 0px !important;
        font-size: 10px !important;
        font-family: 'Courier New', monospace !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
        transform: translateY(-2px) !important;
    }
    
    hr {
        border-color: #333333;
        margin: 8px 0;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .section-box {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FFAA00;
    }
    
    /* Style pour les tables */
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 9px !important;
        color: #FFAA00 !important;
        background-color: #111 !important;
    }
    
    .dataframe th {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        border: 1px solid #555 !important;
        padding: 6px !important;
        font-size: 9px !important;
    }
    
    .dataframe td {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        padding: 4px !important;
        font-size: 9px !important;
    }
    
    /* Highlight rows on hover */
    .dataframe tbody tr:hover {
        background-color: #222 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #333;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# FINRA API INTEGRATION
# =============================================

class FINRAClient:
    """Client pour l'API FINRA"""
    
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.finra.org"
        self.token_url = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
        self.access_token = None
        self.token_expiry = None
    
    def get_access_token(self):
        """Obtient un token d'accès OAuth2"""
        try:
            # Encoder les credentials en base64
            credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "client_credentials"
            }
            
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                # Token valide généralement 1h
                self.token_expiry = datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))
                return True
            else:
                st.error(f"❌ Échec authentification FINRA: {response.status_code}")
                st.error(f"Détails: {response.text}")
                return False
        except Exception as e:
            st.error(f"❌ Erreur connexion FINRA: {str(e)}")
            return False
    
    def ensure_token_valid(self):
        """Vérifie que le token est valide"""
        if not self.access_token or not self.token_expiry:
            return self.get_access_token()
        
        if datetime.now() >= self.token_expiry:
            return self.get_access_token()
        
        return True
    
    def get_corporate_bonds(self, limit=5000):
        """Récupère les obligations corporate depuis FINRA"""
        if not self.ensure_token_valid():
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json"
            }
            
            # Endpoint pour les corporate bonds
            # Note: L'endpoint exact peut varier, consultez la doc FINRA
            endpoint = f"{self.base_url}/data/group/FIXEDINCOME/name/corporateBondReference"
            
            params = {
                "limit": limit,
                "offset": 0
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get('data', []))
            else:
                st.warning(f"⚠️ Réponse FINRA: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"❌ Erreur récupération données: {str(e)}")
            return None
    
    def search_bonds_by_issuer(self, issuer_name):
        """Recherche d'obligations par émetteur"""
        if not self.ensure_token_valid():
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json"
            }
            
            # Recherche avec filtre
            endpoint = f"{self.base_url}/data/group/FIXEDINCOME/name/corporateBondReference"
            
            params = {
                "limit": 1000,
                "compareFilters": json.dumps([{
                    "fieldName": "issuerName",
                    "fieldValue": issuer_name,
                    "compareType": "CONTAINS"
                }])
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get('data', []))
            else:
                return None
                
        except Exception as e:
            return None

# =============================================
# CONFIGURATION FINRA
# =============================================

def setup_finra_credentials():
    """Configuration des identifiants FINRA"""
    st.sidebar.markdown("## 🔐 FINRA API CONFIG")
    
    # Vérifier si les credentials sont déjà en session
    if 'finra_configured' not in st.session_state:
        st.session_state['finra_configured'] = False
    
    # Formulaire de configuration
    with st.sidebar.expander("📝 Configure FINRA API", expanded=not st.session_state['finra_configured']):
        client_id = st.text_input(
            "API Client ID",
            value=st.session_state.get('finra_client_id', '4c7a3b25323c4ddd91ab'),
            help="Votre API Client User ID FINRA"
        )
        
        client_secret = st.text_input(
            "API Client Secret",
            value=st.session_state.get('finra_client_secret', ''),
            type="password",
            help="Votre mot de passe API FINRA"
        )
        
        if st.button("🔑 Connect to FINRA"):
            if client_id and client_secret:
                st.session_state['finra_client_id'] = client_id
                st.session_state['finra_client_secret'] = client_secret
                
                # Tester la connexion
                with st.spinner("🔄 Connexion à FINRA..."):
                    client = FINRAClient(client_id, client_secret)
                    if client.get_access_token():
                        st.session_state['finra_client'] = client
                        st.session_state['finra_configured'] = True
                        st.success("✅ Connecté à FINRA API!")
                        st.rerun()
                    else:
                        st.error("❌ Échec de connexion. Vérifiez vos identifiants.")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs")
    
    # Afficher le statut
    if st.session_state['finra_configured']:
        st.sidebar.success("✅ FINRA API: Connected")
    else:
        st.sidebar.warning("⚠️ FINRA API: Not configured")
    
    return st.session_state.get('finra_client')

# =============================================
# BASE D'ETFs OBLIGATAIRES
# =============================================

BOND_ETFS = {
    # US TREASURIES
    'SHY': {'name': 'iShares 1-3Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Short', 'region': 'US'},
    'IEI': {'name': 'iShares 3-7Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'IEF': {'name': 'iShares 7-10Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'TLH': {'name': 'iShares 10-20Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    'TLT': {'name': 'iShares 20+Y Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Very Long', 'region': 'US'},
    'VGIT': {'name': 'Vanguard Int-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'VGLT': {'name': 'Vanguard Long-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    'SCHO': {'name': 'Schwab Short-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Short', 'region': 'US'},
    'SCHR': {'name': 'Schwab Int-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Medium', 'region': 'US'},
    'SPTL': {'name': 'SPDR Long-Term Treasury', 'type': 'ETF', 'category': 'Treasury', 'duration': 'Long', 'region': 'US'},
    
    # CORPORATE IG
    'LQD': {'name': 'iShares iBoxx IG Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'VCIT': {'name': 'Vanguard Int-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'VCLT': {'name': 'Vanguard Long-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Long', 'region': 'US'},
    'VCSH': {'name': 'Vanguard Short-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    'IGSB': {'name': 'iShares Short-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    'IGIB': {'name': 'iShares Int-Term Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'USIG': {'name': 'iShares Broad USD IG', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Medium', 'region': 'US'},
    'SLQD': {'name': 'iShares 0-5Y IG Corp', 'type': 'ETF', 'category': 'Investment Grade', 'duration': 'Short', 'region': 'US'},
    
    # HIGH YIELD
    'HYG': {'name': 'iShares High Yield Corp', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'JNK': {'name': 'SPDR High Yield Bond', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'SHYG': {'name': 'iShares 0-5Y High Yield', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Short', 'region': 'US'},
    'FALN': {'name': 'iShares Fallen Angels', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    'SJNK': {'name': 'SPDR Short-Term HY', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Short', 'region': 'US'},
    'HYDB': {'name': 'iShares High Yield Discount', 'type': 'ETF', 'category': 'High Yield', 'duration': 'Medium', 'region': 'US'},
    
    # EMERGING MARKETS
    'EMB': {'name': 'iShares EM USD Bond', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    'EMHY': {'name': 'iShares EM High Yield', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    'EMLC': {'name': 'VanEck EM Local Currency', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    'PCY': {'name': 'Invesco EM Sovereign', 'type': 'ETF', 'category': 'Emerging Markets', 'duration': 'Medium', 'region': 'EM'},
    
    # TIPS
    'TIP': {'name': 'iShares TIPS Bond', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Medium', 'region': 'US'},
    'VTIP': {'name': 'Vanguard Short-Term TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Short', 'region': 'US'},
    'LTPZ': {'name': 'PIMCO 15+ Year TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Very Long', 'region': 'US'},
    'SCHP': {'name': 'Schwab US TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Medium', 'region': 'US'},
    'SPIP': {'name': 'SPDR TIPS', 'type': 'ETF', 'category': 'TIPS', 'duration': 'Medium', 'region': 'US'},
    
    # MUNICIPAL
    'MUB': {'name': 'iShares National Muni', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
    'VTEB': {'name': 'Vanguard Tax-Exempt', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
    'SUB': {'name': 'iShares Short-Term Muni', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Short', 'region': 'US'},
    'HYD': {'name': 'VanEck High Yield Muni', 'type': 'ETF', 'category': 'Municipal', 'duration': 'Medium', 'region': 'US'},
    
    # INTERNATIONAL
    'BNDX': {'name': 'Vanguard Total Intl Bond', 'type': 'ETF', 'category': 'International', 'duration': 'Medium', 'region': 'International'},
    'IAGG': {'name': 'iShares Core Intl Agg', 'type': 'ETF', 'category': 'International', 'duration': 'Medium', 'region': 'International'},
    
    # AGGREGATE
    'AGG': {'name': 'iShares Core US Agg', 'type': 'ETF', 'category': 'Aggregate', 'duration': 'Medium', 'region': 'US'},
    'BND': {'name': 'Vanguard Total Bond', 'type': 'ETF', 'category': 'Aggregate', 'duration': 'Medium', 'region': 'US'},
    'SCHZ': {'name': 'Schwab US Aggregate', 'type': 'ETF', 'category': 'Aggregate', 'duration': 'Medium', 'region': 'US'},
}

# =============================================
# FONCTIONS
# =============================================

@st.cache_data(ttl=300)
def get_etf_data(ticker):
    """Récupère les données d'un ETF obligataire"""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        hist = etf.history(period='1y')
        
        if len(hist) < 2:
            return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_1d = ((current_price - prev_close) / prev_close) * 100
        
        if len(hist) > 0:
            change_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        else:
            change_ytd = None
        
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else None
        
        dividend_yield = info.get('yield', None)
        if dividend_yield:
            dividend_yield = dividend_yield * 100
        
        expense_ratio = info.get('expenseRatio', None)
        if expense_ratio:
            expense_ratio = expense_ratio * 100
        
        return {
            'Price': current_price,
            '1D %': change_1d,
            'YTD %': change_ytd,
            'Volatility %': volatility,
            'Yield %': dividend_yield,
            'Expense %': expense_ratio,
        }
    except:
        return None

def calculate_ytm_approximate(coupon, price, years_to_maturity, face_value=100):
    """Calcule un YTM approximatif"""
    try:
        annual_interest = (coupon / 100) * face_value
        capital_gain = (face_value - price) / years_to_maturity
        ytm = ((annual_interest + capital_gain) / ((face_value + price) / 2)) * 100
        return ytm
    except:
        return None

def get_years_to_maturity(maturity_date_str):
    """Calcule les années jusqu'à maturité"""
    try:
        maturity = datetime.strptime(maturity_date_str, '%Y-%m-%d')
        today = datetime.now()
        years = (maturity - today).days / 365.25
        return max(0, years)
    except:
        return None

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - BOND SCREENER PRO</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">MARKETS</a>
    </div>
    <div>{current_time} UTC • FINRA API INTEGRATED</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# SETUP FINRA
# =============================================
finra_client = setup_finra_credentials()

# =============================================
# INSTRUCTIONS
# =============================================
st.markdown("""
<div style='background:#111;border:1px solid #333;padding:10px;margin:10px 0;border-left:4px solid #FFAA00;'>
<b style='color:#FFAA00;'>🔍 BOND SCREENER PRO - FINRA API INTEGRATION:</b><br>
• <b style='color:#00FF00;'>REAL FINRA DATA</b>: Access live corporate bond data from FINRA TRACE<br>
• <b style='color:#00FFFF;'>40+ BOND ETFs</b>: Complete ETF coverage with Yahoo Finance data<br>
• <b style='color:#FF00FF;'>COMPARISON TOOL</b>: Chart multiple bonds/ETFs side-by-side<br>
• <b style='color:#FFAA00;'>Configure your FINRA API credentials in the sidebar to access real bond data!</b><br>
</div>
""", unsafe_allow_html=True)

# =============================================
# TABS
# =============================================
tab1, tab2, tab3 = st.tabs(["🏢 CORPORATE BONDS (FINRA)", "📊 BOND ETFs", "📈 COMPARISON TOOL"])

# =============================================
# TAB 1: CORPORATE BONDS (FINRA)
# =============================================
with tab1:
    st.markdown("### 🏢 CORPORATE BONDS - FINRA API")
    
    if finra_client and st.session_state.get('finra_configured'):
        st.success("✅ Connected to FINRA API - Ready to load real bond data")
        
        if st.button("🔄 LOAD CORPORATE BONDS FROM FINRA", key="load_finra"):
            with st.spinner("📡 Fetching data from FINRA API..."):
                df_bonds = finra_client.get_corporate_bonds(limit=5000)
                
                if df_bonds is not None and len(df_bonds) > 0:
                    st.session_state['finra_bonds'] = df_bonds
                    st.success(f"✅ Loaded {len(df_bonds)} corporate bonds from FINRA!")
                else:
                    st.warning("⚠️ No data received from FINRA. The endpoint may be different or requires special access.")
                    st.info("""
                    **Note**: L'endpoint exact pour les obligations corporate peut varier selon votre accès FINRA.
                    
                    Endpoints possibles:
                    - `/data/group/FIXEDINCOME/name/corporateBondReference`
                    - `/data/group/OTCMARKET/name/bondData`
                    - `/data/group/TRACE/name/corporateBonds`
                    
                    Consultez la documentation FINRA à https://developer.finra.org/docs pour l'endpoint exact
                    correspondant à votre subscription.
                    """)
        
        # Afficher les données si disponibles
        if 'finra_bonds' in st.session_state and st.session_state['finra_bonds'] is not None:
            df_bonds = st.session_state['finra_bonds']
            
            st.markdown(f"### 📊 FINRA BONDS: {len(df_bonds)} bonds loaded")
            
            # Afficher les colonnes disponibles
            with st.expander("📋 Available columns in FINRA data"):
                st.write(df_bonds.columns.tolist())
            
            # Afficher un échantillon
            st.markdown("#### Sample Data:")
            st.dataframe(df_bonds.head(20), use_container_width=True)
            
            # Export
            csv_finra = df_bonds.to_csv(index=False)
            st.download_button(
                label="📥 DOWNLOAD FINRA BONDS (CSV)",
                data=csv_finra,
                file_name=f"finra_bonds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    
    else:
        st.warning("⚠️ FINRA API not configured")
        st.info("""
        **Pour accéder aux données FINRA:**
        
        1. **Configurez vos identifiants** dans la barre latérale (sidebar)
        2. Entrez votre **API Client ID**: `4c7a3b25323c4ddd91ab`
        3. Entrez votre **API Client Secret** (mot de passe)
        4. Cliquez sur **Connect to FINRA**
        5. Une fois connecté, cliquez sur **LOAD CORPORATE BONDS FROM FINRA**
        
        **Note**: L'accès FINRA peut nécessiter une subscription spécifique.
        Si vous n'avez pas accès, utilisez l'onglet "BOND ETFs" pour les données Yahoo Finance.
        """)

# =============================================
# TAB 2: BOND ETFs
# =============================================
with tab2:
    st.markdown("### 📊 BOND ETFs SCREENER")
    st.markdown(f"**Database: 40+ bond ETFs - Yahoo Finance data**")
    
    if st.button("🔄 LOAD BOND ETFs DATA", key="load_etfs"):
        with st.spinner("Loading 40+ bond ETFs..."):
            etf_data = []
            progress_bar = st.progress(0)
            total = len(BOND_ETFS)
            
            for idx, (ticker, info) in enumerate(BOND_ETFS.items()):
                progress_bar.progress((idx + 1) / total)
                
                metrics = get_etf_data(ticker)
                
                if metrics:
                    row = {
                        'Ticker': ticker,
                        'Name': info['name'],
                        'Type': info['type'],
                        'Category': info['category'],
                        'Duration': info['duration'],
                        'Region': info['region'],
                        **metrics
                    }
                    etf_data.append(row)
                
                time.sleep(0.1)
            
            progress_bar.empty()
            st.session_state['etf_bonds'] = pd.DataFrame(etf_data)
        
        st.success(f"✅ {len(st.session_state['etf_bonds'])} bond ETFs loaded!")
    
    if 'etf_bonds' in st.session_state and st.session_state['etf_bonds'] is not None:
        df_etf = st.session_state['etf_bonds'].copy()
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # FILTRES
        with st.sidebar:
            st.markdown("## 🎯 ETF FILTERS")
            
            categories = ['All'] + sorted(df_etf['Category'].unique().tolist())
            selected_cat = st.selectbox("Category", categories, key="etf_cat")
            
            durations = ['All'] + sorted(df_etf['Duration'].unique().tolist())
            selected_dur = st.selectbox("Duration", durations, key="etf_dur")
            
            regions = ['All'] + sorted(df_etf['Region'].unique().tolist())
            selected_reg = st.selectbox("Region", regions, key="etf_reg")
            
            st.markdown("---")
            
            ytd_min_etf, ytd_max_etf = st.slider(
                "YTD Return (%)",
                min_value=float(df_etf['YTD %'].min()),
                max_value=float(df_etf['YTD %'].max()),
                value=(float(df_etf['YTD %'].min()), float(df_etf['YTD %'].max())),
                key="etf_ytd"
            )
            
            vol_min_etf, vol_max_etf = st.slider(
                "Volatility (%)",
                min_value=float(df_etf['Volatility %'].min()),
                max_value=float(df_etf['Volatility %'].max()),
                value=(float(df_etf['Volatility %'].min()), float(df_etf['Volatility %'].max())),
                key="etf_vol"
            )
        
        # Appliquer filtres
        filtered_etf = df_etf.copy()
        
        if selected_cat != 'All':
            filtered_etf = filtered_etf[filtered_etf['Category'] == selected_cat]
        
        if selected_dur != 'All':
            filtered_etf = filtered_etf[filtered_etf['Duration'] == selected_dur]
        
        if selected_reg != 'All':
            filtered_etf = filtered_etf[filtered_etf['Region'] == selected_reg]
        
        filtered_etf = filtered_etf[
            (filtered_etf['YTD %'] >= ytd_min_etf) &
            (filtered_etf['YTD %'] <= ytd_max_etf) &
            (filtered_etf['Volatility %'] >= vol_min_etf) &
            (filtered_etf['Volatility %'] <= vol_max_etf)
        ]
        
        # RÉSULTATS
        st.markdown(f"### 📊 RESULTS: {len(filtered_etf)} ETFs found")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            avg_ytd_etf = filtered_etf['YTD %'].mean()
            st.metric("Avg YTD", f"{avg_ytd_etf:.2f}%")
        
        with col_stat2:
            avg_vol_etf = filtered_etf['Volatility %'].mean()
            st.metric("Avg Volatility", f"{avg_vol_etf:.2f}%")
        
        with col_stat3:
            if filtered_etf['Yield %'].notna().any():
                avg_yield_etf = filtered_etf['Yield %'].mean()
                st.metric("Avg Yield", f"{avg_yield_etf:.2f}%")
            else:
                st.metric("Avg Yield", "N/A")
        
        with col_stat4:
            avg_price_etf = filtered_etf['Price'].mean()
            st.metric("Avg Price", f"${avg_price_etf:.2f}")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Tableau
        display_etf = filtered_etf.copy()
        
        for col in ['Price', '1D %', 'YTD %', 'Volatility %', 'Yield %', 'Expense %']:
            if col in display_etf.columns:
                display_etf[col] = display_etf[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_etf,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Export
        csv_etf = filtered_etf.to_csv(index=False)
        st.download_button(
            label="📥 DOWNLOAD BOND ETFs (CSV)",
            data=csv_etf,
            file_name=f"bond_etfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # VISUALISATIONS
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("### 📈 ETF PERFORMANCE ANALYSIS")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            cat_perf = filtered_etf.groupby('Category')['YTD %'].mean().sort_values()
            
            fig_cat = go.Figure()
            
            colors_cat = ['#FF0000' if x < 0 else '#00FF00' for x in cat_perf.values]
            
            fig_cat.add_trace(go.Bar(
                x=cat_perf.values,
                y=cat_perf.index,
                orientation='h',
                marker_color=colors_cat,
                text=cat_perf.values,
                texttemplate='%{text:.2f}%',
                textposition='outside',
            ))
            
            fig_cat.update_layout(
                title="YTD Performance by Category",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', showgrid=True, title="YTD (%)"),
                yaxis=dict(gridcolor='#333', showgrid=False),
                height=400
            )
            
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col_viz2:
            fig_rr = go.Figure()
            
            for category in filtered_etf['Category'].unique():
                cat_data = filtered_etf[filtered_etf['Category'] == category]
                
                fig_rr.add_trace(go.Scatter(
                    x=cat_data['Volatility %'],
                    y=cat_data['YTD %'],
                    mode='markers+text',
                    name=category,
                    text=cat_data['Ticker'],
                    textposition='top center',
                    marker=dict(size=10),
                ))
            
            fig_rr.update_layout(
                title="Risk vs Return",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', showgrid=True, title="Volatility (%)"),
                yaxis=dict(gridcolor='#333', showgrid=True, title="YTD Return (%)"),
                height=400
            )
            
            st.plotly_chart(fig_rr, use_container_width=True)
    
    else:
        st.info("👆 Click 'LOAD BOND ETFs DATA' to start screening")

# =============================================
# TAB 3: COMPARISON TOOL
# =============================================
with tab3:
    st.markdown("### 📈 BOND & ETF COMPARISON TOOL")
    st.markdown("**Compare multiple bond ETFs side-by-side**")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    available_etfs = list(BOND_ETFS.keys())
    
    col_comp1, col_comp2 = st.columns([3, 1])
    
    with col_comp1:
        selected_compare = st.multiselect(
            "Select bond ETFs to compare (up to 8)",
            options=available_etfs,
            default=['TLT', 'LQD', 'HYG', 'AGG'],
            max_selections=8
        )
    
    with col_comp2:
        compare_period = st.selectbox(
            "Period",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3,
            key="compare_period"
        )
    
    if selected_compare:
        st.markdown('<hr>', unsafe_allow_html=True)
        
        fig_compare = go.Figure()
        
        colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00', '#00CED1', '#FF1493']
        
        comparison_stats = []
        
        for idx, ticker in enumerate(selected_compare):
            try:
                bond = yf.Ticker(ticker)
                hist = bond.history(period=compare_period)
                
                if len(hist) > 0:
                    # Normaliser à 100
                    normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                    
                    # Stats
                    total_return = normalized.iloc[-1] - 100
                    volatility = (hist['Close'].pct_change().std() * np.sqrt(252)) * 100
                    max_price = hist['Close'].max()
                    min_price = hist['Close'].min()
                    
                    comparison_stats.append({
                        'Ticker': ticker,
                        'Name': BOND_ETFS[ticker]['name'],
                        'Return %': total_return,
                        'Volatility %': volatility,
                        'Max Price': max_price,
                        'Min Price': min_price,
                        'Current': hist['Close'].iloc[-1]
                    })
                    
                    fig_compare.add_trace(go.Scatter(
                        x=hist.index,
                        y=normalized,
                        mode='lines',
                        name=f"{ticker}",
                        line=dict(color=colors[idx % len(colors)], width=2),
                        hovertemplate=f'<b>{ticker}</b><br>%{{y:.2f}}%<br>%{{x}}<extra></extra>'
                    ))
            except Exception as e:
                st.warning(f"Could not load data for {ticker}")
                continue
        
        fig_compare.update_layout(
            title=f"Bond ETF Performance Comparison - {compare_period.upper()}",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Date"
            ),
            yaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Performance (%)"
            ),
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(17,17,17,0.8)'
            ),
            height=600
        )
        
        fig_compare.add_hline(y=100, line_dash="dash", line_color="#666", annotation_text="Start")
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Tableau de comparaison
        if comparison_stats:
            st.markdown("### 📊 COMPARISON STATISTICS")
            
            df_comp_stats = pd.DataFrame(comparison_stats)
            df_comp_stats = df_comp_stats.round(2)
            
            st.dataframe(df_comp_stats, use_container_width=True, hide_index=True)
            
            # Winner/Loser
            st.markdown('<hr>', unsafe_allow_html=True)
            
            col_win1, col_win2, col_win3 = st.columns(3)
            
            with col_win1:
                best_performer = df_comp_stats.loc[df_comp_stats['Return %'].idxmax()]
                st.metric(
                    "🏆 BEST PERFORMER",
                    best_performer['Ticker'],
                    f"+{best_performer['Return %']:.2f}%"
                )
            
            with col_win2:
                worst_performer = df_comp_stats.loc[df_comp_stats['Return %'].idxmin()]
                st.metric(
                    "📉 WORST PERFORMER",
                    worst_performer['Ticker'],
                    f"{worst_performer['Return %']:.2f}%"
                )
            
            with col_win3:
                least_volatile = df_comp_stats.loc[df_comp_stats['Volatility %'].idxmin()]
                st.metric(
                    "🎯 LEAST VOLATILE",
                    least_volatile['Ticker'],
                    f"{least_volatile['Volatility %']:.2f}% vol"
                )
    
    else:
        st.info("👆 Select bond ETFs above to compare their performance")

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

with st.expander("📖 FINRA API SETUP GUIDE"):
    st.markdown("""
    ## 🔐 FINRA API Configuration Guide
    
    ### Step 1: Get Your FINRA API Credentials
    
    1. Go to https://developer.finra.org/
    2. Create an account or log in
    3. Navigate to the API Console
    4. Create a new API Credential
    5. Note your **API Client ID** and **API Client Secret**
    
    ### Step 2: Configure in This App
    
    1. Open the sidebar (click `>` on the left)
    2. Find "FINRA API CONFIG" section
    3. Expand "Configure FINRA API"
    4. Enter your credentials:
       - API Client ID: Your client ID (e.g., `4c7a3b25323c4ddd91ab`)
       - API Client Secret: Your secret/password
    5. Click "Connect to FINRA"
    
    ### Step 3: Load Bond Data
    
    1. Go to the "CORPORATE BONDS (FINRA)" tab
    2. Click "LOAD CORPORATE BONDS FROM FINRA"
    3. Wait for data to load (may take 10-30 seconds)
    
    ### Important Notes:
    
    - **Free Access**: FINRA API is free but requires registration
    - **Rate Limits**: API has rate limits (typically 100,000 records max)
    - **Endpoints**: Exact endpoints may vary based on your FINRA access level
    - **Support**: Contact FINRA at (888) 507-3665 for API support
    
    ### Alternative: Use ETF Data
    
    If you don't have FINRA access, use the "BOND ETFs" tab which uses Yahoo Finance (no API key required).
    
    ### Troubleshooting:
    
    **Error "Échec authentification FINRA"**:
    - Verify your Client ID and Secret are correct
    - Check your internet connection
    - Ensure your FINRA account is active
    
    **Error "No data received from FINRA"**:
    - The endpoint may require special access rights
    - Contact FINRA to verify your subscription includes corporate bond data
    - Try the ETF screener as an alternative
    """)

col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DATA SOURCES: FINRA API + YAHOO FINANCE<br>
        🔄 REAL-TIME CORPORATE BONDS • 40+ ETFs • COMPARISON TOOL
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    finra_status = "CONNECTED" if st.session_state.get('finra_configured') else "NOT CONFIGURED"
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 SESSION: {last_update}<br>
        📍 FINRA STATUS: {finra_status}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | BOND SCREENER PRO | FINRA API INTEGRATED<br>
    CORPORATE BONDS + ETFs • REAL-TIME DATA • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
