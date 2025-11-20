import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time
import requests

# =============================================
# AUTO-REFRESH TOUTES LES 3 SECONDES
# =============================================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 3:
    st.session_state.last_refresh = time.time()
    st.rerun()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Markets",
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
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    .stApp {
        background-color: #000000;
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
</style>
""", unsafe_allow_html=True)

# =============================================
# API COINMARKETCAP
# =============================================
CMC_API_KEY = "09e527de-bfea-4816-8afe-ae6a37bf5799"

@st.cache_data(ttl=3)
def get_crypto_data_cmc(symbols):
    """Récupère les données crypto depuis CoinMarketCap"""
    try:
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        parameters = {
            'symbol': ','.join(symbols),
            'convert': 'USD'
        }
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': CMC_API_KEY,
        }
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()
        
        crypto_data = {}
        for symbol in symbols:
            if symbol in data['data']:
                quote = data['data'][symbol]['quote']['USD']
                crypto_data[symbol] = {
                    'price': quote['price'],
                    'change_24h': quote['percent_change_24h']
                }
        return crypto_data
    except Exception as e:
        return None

# =============================================
# FONCTION DONNÉES MARCHÉ
# =============================================
@st.cache_data(ttl=3)
def get_market_data(ticker):
    """Récupère les données réelles de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None, None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return current_price, change_percent
    except:
        return None, None

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - LIVE MARKETS</div>
        <a href="accueil.html" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • AUTO-REFRESH: 3s</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# INDICES GLOBAUX
# =============================================
st.markdown("### 📊 GLOBAL INDICES - LIVE")

indices = {
    'NASDAQ': '^IXIC',
    'S&P 500': '^GSPC',
    'DOW JONES': '^DJI',
    'RUSSELL 2000': '^RUT',
    'VIX': '^VIX',
    'CAC 40': '^FCHI',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'SMI (SIX)': '^SSMI',
    'FTSE MIB': 'FTSEMIB.MI',
    'NIKKEI 225': '^N225',
    'SSE (CHINA)': '000001.SS',
    'DXY (USD)': 'DX-Y.NYB'
}

cols_indices = st.columns(6)

for idx, (name, ticker) in enumerate(indices.items()):
    with cols_indices[idx % 6]:
        current, change = get_market_data(ticker)
        
        if current is not None:
            if name == 'VIX':
                value_display = f"{current:.2f}"
            else:
                value_display = f"{current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# FOREX
# =============================================
st.markdown("### 💱 FOREX - LIVE")

forex = {
    'EUR/USD': 'EURUSD=X',
    'CHF/USD': 'CHFUSD=X',
    'CHF/EUR': 'CHFEUR=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'JPY=X',
    'USD/CNY': 'CNY=X'
}

cols_fx = st.columns(6)

for idx, (name, ticker) in enumerate(forex.items()):
    with cols_fx[idx]:
        current, change = get_market_data(ticker)
        
        if current is not None:
            st.metric(
                label=name,
                value=f"{current:.4f}",
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# COMMODITIES
# =============================================
st.markdown("### 💰 COMMODITIES - LIVE")

commodities = {
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'PLATINUM': 'PL=F',
    'COPPER': 'HG=F',
    'OIL (WTI)': 'CL=F',
    'NAT GAS': 'NG=F',
    'BRENT OIL': 'BZ=F',
    'ALUMINUM': 'ALI=F'
}

cols_comm = st.columns(4)

for idx, (name, ticker) in enumerate(commodities.items()):
    with cols_comm[idx % 4]:
        current, change = get_market_data(ticker)
        
        if current is not None:
            value_display = f"${current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# CRYPTO - COINMARKETCAP
# =============================================
st.markdown("### ₿ CRYPTO - COINMARKETCAP LIVE")

crypto_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'AAVE', 'LINK']
crypto_data_cmc = get_crypto_data_cmc(crypto_symbols)

cols_crypto = st.columns(6)

crypto_pairs = [
    ('BTC', 'BTCUSDT'),
    ('ETH', 'ETHUSDT'),
    ('SOL', 'SOLUSDT'),
    ('XRP', 'XRPUSDT'),
    ('AAVE', 'AAVEUSDT'),
    ('LINK', 'LINKUSDT')
]

if crypto_data_cmc:
    for idx, (symbol, pair) in enumerate(crypto_pairs):
        with cols_crypto[idx]:
            if symbol in crypto_data_cmc:
                price = crypto_data_cmc[symbol]['price']
                change = crypto_data_cmc[symbol]['change_24h']
                
                if price >= 100:
                    value_display = f"${price:,.2f}"
                else:
                    value_display = f"${price:.4f}"
                
                st.metric(
                    label=pair,
                    value=value_display,
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=pair, value="ERROR", delta="0%")
else:
    for idx, (_, pair) in enumerate(crypto_pairs):
        with cols_crypto[idx]:
            st.metric(label=pair, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# BOUTONS ET INFO
# =============================================
col_info1, col_info2, col_refresh = st.columns([4, 4, 2])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES EN TEMPS RÉEL • YAHOO FINANCE + COINMARKETCAP<br>
        🔄 RAFRAÎCHISSEMENT AUTOMATIQUE: 3 SECONDES
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 CONNEXION: PARIS, FRANCE
    </div>
    """, unsafe_allow_html=True)

with col_refresh:
    if st.button("🔄 REFRESH NOW", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE • COINMARKETCAP | SYSTÈME OPÉRATIONNEL<br>
    DONNÉES DE MARCHÉ DISPONIBLES • REFRESH AUTO: 3s • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)

# Auto-refresh JavaScript
st.markdown("""
<script>
    setTimeout(function(){
        window.parent.location.reload();
    }, 3000);
</script>
""", unsafe_allow_html=True)
