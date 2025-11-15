import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time

import requests

import time

# Configuration auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Auto-refresh toutes les 3 secondes
if time.time() - st.session_state.last_refresh > 3:
    st.session_state.last_refresh = time.time()
    st.rerun()

# Configuration API CoinMarketCap
CMC_API_KEY = "09e527de-bfea-4816-8afe-ae6a37bf5799"  # Remplacez par votre clé API

@st.cache_data(ttl=3)  # Cache de 3 secondes
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
        print(f"Erreur CMC API: {e}")
        return None

# Configuration de la page
st.set_page_config(
    page_title="Bloomberg Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS + JavaScript pour horloge en temps réel
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
    
    .news-item {
        background-color: #0a0a0a;
        border-left: 3px solid #FFAA00;
        padding: 8px 10px;
        margin-bottom: 5px;
        border-bottom: 1px solid #222;
        font-family: 'Courier New', monospace;
    }
    
    .news-title {
        color: #FFAA00;
        font-size: 11px;
        font-weight: 600;
        margin: 0;
        line-height: 1.3;
    }
    
    .news-meta {
        color: #666;
        font-size: 9px;
        margin-top: 3px;
    }
    
    .news-category {
        color: #FFAA00;
        font-weight: bold;
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
    
    .stTextInput input {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    .stTextInput input:focus {
        border-color: #FFF;
        box-shadow: 0 0 3px #FFAA00;
    }
    
    .stMultiSelect {
        background-color: #000;
    }
    
    .stMultiSelect > div {
        background-color: #000;
        border: 1px solid #FFAA00;
    }
    
    .stSelectbox select {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        font-family: 'Courier New', monospace;
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
    
    .live-clock {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    .command-line {
        background: #000;
        padding: 5px 10px;
        border: 1px solid #333;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #FFAA00;
        margin: 5px 0;
    }
    
    .prompt {
        color: #FFAA00;
        font-weight: bold;
        margin-right: 8px;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    /* Chart container */
    .chart-section {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
    }
</style>

<script>
    // Horloge en temps réel
    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        const timeString = hours + ':' + minutes + ':' + seconds + ' PARIS';
        
        const clockElements = document.querySelectorAll('.live-clock');
        clockElements.forEach(el => {
            el.textContent = timeString;
        });
    }
    
    setInterval(updateClock, 1000);
    updateClock();
</script>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données réelles
@st.cache_data(ttl=60)
def get_market_data(ticker):
    """Récupère les données réelles de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None, None, None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return current_price, change_percent, hist
        
    except:
        return None, None, None

# Fonction pour récupérer les données historiques
@st.cache_data(ttl=300)
def get_historical_data(ticker, period):
    """Récupère les données historiques pour un ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except:
        return None

# Fonction pour normaliser en %
def normalize_to_percentage(df):
    """Normalise les prix à 100% au début"""
    if df is None or len(df) == 0:
        return None
    return (df / df.iloc[0]) * 100

# Fonction pour récupérer les news réelles
@st.cache_data(ttl=300)
def get_real_news(ticker):
    """Récupère les vraies news de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:10] if news else []
    except:
        return []

# ===== HEADER BLOOMBERG avec horloge =====
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL</div>
    <div class="live-clock">{current_time.strftime("%H:%M:%S")} PARIS</div>
</div>
''', unsafe_allow_html=True)

# ===== MARKET OVERVIEW =====
st.markdown("### 📊 GLOBAL MARKETS - LIVE")

markets = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    'EUR/USD': 'EURUSD=X'
}

cols = st.columns(4)

for idx, (name, ticker) in enumerate(markets.items()):
    with cols[idx]:
        current, change, hist = get_market_data(ticker)
        
        if current is not None:
            if 'USD' in ticker or 'EUR' in ticker:
                value_display = f"{current:.4f}"
            else:
                value_display = f"{current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

# ===== CRYPTO LIVE (CoinMarketCap) =====
st.markdown("### ₿ CRYPTO LIVE - COINMARKETCAP")

crypto_symbols = ['BTC', 'ETH', 'SOL']
crypto_data_cmc = get_crypto_data_cmc(crypto_symbols)

cols_crypto = st.columns(3)

if crypto_data_cmc:
    crypto_pairs = [
        ('BTC', 'BTCUSDT'),
        ('ETH', 'ETHUSDT'),
        ('SOL', 'SOLUSDT')
    ]
    
    for idx, (symbol, pair) in enumerate(crypto_pairs):
        with cols_crypto[idx]:
            if symbol in crypto_data_cmc:
                price = crypto_data_cmc[symbol]['price']
                change = crypto_data_cmc[symbol]['change_24h']
                
                st.metric(
                    label=pair,
                    value=f"${price:,.2f}",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=pair, value="ERROR", delta="0%")
else:
    for idx, pair in enumerate(['BTCUSDT', 'ETHUSDT', 'SOLUSDT']):
        with cols_crypto[idx]:
            st.metric(label=pair, value="LOAD...", delta="0%")

# Auto-refresh toutes les 3 secondes
st.markdown("""
<script>
    setTimeout(function(){
        window.parent.location.reload();
    }, 3000);
</script>
""", unsafe_allow_html=True)

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

# ===== COMMODITIES =====
st.markdown("### 💰 COMMODITIES")

commodities = {
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'OIL': 'CL=F',
    'GAS': 'NG=F'
}

cols_comm = st.columns(4)

for idx, (name, ticker) in enumerate(commodities.items()):
    with cols_comm[idx]:
        current, change, _ = get_market_data(ticker)
        
        if current is not None:
            value_display = f"${current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)



# ===== BARRE DE RECHERCHE / NAVIGATION =====
st.markdown("""
<div class="command-line">
    <span class="prompt">FUNCTION></span>
    <span style="color: #666;">Tapez: PRICE, NEWS, SCREENER, PORTFOLIO, HELP...</span>
</div>
""", unsafe_allow_html=True)

# Formulaire pour gérer l'entrée
with st.form(key="nav_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        search_command = st.text_input(
            "",
            placeholder="Entrez une fonction (PRICE, NEWS, HELP...)",
            label_visibility="collapsed",
            key="command_input"
        )
    
    with col2:
        submit = st.form_submit_button("EXEC", use_container_width=True)
    
    if submit and search_command:
        cmd = search_command.upper().strip()
        
        if cmd == "PRICE" or cmd == "PRICING":
            st.info("📊 Page PRICING : Cliquez sur le bouton dans la sidebar →")
        elif cmd == "EDGAR" or cmd == "E":
            st.info("📋 Page EDGAR : Cliquez sur 'EDGAR' dans la sidebar →")
            st.page_link("pages/EDGAR.py", label="🔗 Ouvrir EDGAR", icon="📋")
        elif cmd == "NEWS" or cmd == "N":
            st.info("📰 Page NEWS en construction...")
        elif cmd == "SCREENER" or cmd == "SCREEN":
            st.info("📊 Page SCREENER en construction...")
        elif cmd == "PORTFOLIO" or cmd == "PORT":
            st.info("💼 Page PORTFOLIO en construction...")
        elif cmd == "HELP" or cmd == "H":
            st.info("""
            **📋 FONCTIONS DISPONIBLES:**
            - PRICE / PRICING : Options pricing calculator
            - NEWS / N : Market news
            - SCREENER / SCREEN : Stock screener
            - PORTFOLIO / PORT : Portfolio tracker
            - HELP / H : Afficher cette aide
            """)
        else:
            st.warning(f"⚠️ Fonction '{cmd}' non reconnue. Tapez HELP pour voir les commandes.")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 5px 0;"></div>', unsafe_allow_html=True)

# Bouton refresh manuel
col_r1, col_r2 = st.columns([5, 1])
with col_r2:
    if st.button("🔄 REFRESH", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===== MARKET OVERVIEW =====
st.markdown("### 📈 COMPARATIVE CHART - PERFORMANCE %")

col_chart1, col_chart2, col_chart3 = st.columns([3, 1, 1])

with col_chart1:
    # Liste de tickers populaires
    popular_tickers = ['A','AIT', 'AAL', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS','AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'AZN', 'BIDU', 'BIIB', 'BKNG', 'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EBAY', 'ENPH', 'EXC', 'FANG', 'FAST', 'FISV', 'FTNT', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX', 'NTES', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'RIVN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZM', 'ZS','MC.PA', 'OR.PA', 'SAN.PA', 'AIR.PA', 'AI.PA', 'BNP.PA', 'CS.PA', 'SU.PA', 'TTE.PA', 'BN.PA', 'CA.PA', 'ACA.PA', 'SGO.PA', 'GLE.PA', 'VIE.PA', 'DSY.PA', 'EN.PA', 'CAP.PA', 'DG.PA', 'RMS.PA', 'SAF.PA', 'ORA.PA', 'PUB.PA', 'KER.PA', 'URW.PA', 'RI.PA', 'ML.PA', 'STM.PA', 'TEP.PA', 'VIV.PA', 'SOP.PA', 'WLN.PA', 'BOL.PA', 'ERF.PA', 'EL.PA', 'BVI.PA', 'GET.PA', 'FP.PA', 'LR.PA', 'STLA.PA','NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ABBN.SW', 'ZURN.SW', 'SREN.SW', 'GIVN.SW', 'CSGN.SW', 'SLHN.SW', 'LOGN.SW', 'SIKA.SW', 'GEBN.SW', 'SCMN.SW', 'ALC.SW', 'LONN.SW', 'CFR.SW', 'PGHN.SW', 'HOLN.SW', 'VATN.SW',
                       'SPY', 'QQQ','VIXY', 'BTC-USD', 'ETH-USD', 'TRI','IVV','VOO','VTI','QQQ','VEA','VUG','VTV','IEFA','BND','AGG','IWF','IJH','IJR','VIG','IEMG','VWO','VXUS','VGT','XLK','IWM','VO','GLD','VB','SCHD','IWD','VYM','RSP','ITOT','BNDX','EFA','TLT','VCIT','IVW','QUAL','XLV','SCHX','XLE','VEU','XLF','SCHF','MUB','IXUS','VT','VCSH','IWB','VV','DIA','VNQ','JEPI','IVE','SPLG','IWR','LQD','VTEB','BSV','BIL','VBR','MBB','IEF','IUSB','SCHB','DFAC','IAU','DGRO','SCHG','VGIT','GOVT','SPYG','SHY','USMV','QQQM','GBTC','JPST','COWZ','TQQQ','MDY','SPYV','IGSB','SDY','VGSH','XLY','SPDW','VGK','ACWI','SGOV','VONG','MGK','DVY','TIP','VXF','SMH','XLI','VHT','EEM','SHV','XLC','VMBS','VBK','USFR','BIV','IBIT','EFV','SCHA','IUSG','IUSV','HYG','EWJ','VOE','IYW','XLP','IWP','FNDX','MOAT','PFF','IWS','IWV','EMB','IEI','VOT','IDEV','GDX','SOXX','IGIB','FNDF','EMXC','ESGU','DGRW','GSLC','IWN','XLU','EFG','VGLT','NOBL','JEPQ','USHY','BBJP','OEF','IWO','SCHM','SCHP','MINT','VTIP','SCHV','SOXL','AVUV','USIG','DFUV','SCHO','SPSM','HDV','MTUM','RDVY','SLV','VOOG','IWY','FVD','SPMD','DFAT','FBTC','SCZ','CALF','FTEC','IJK','VFH','VTWO','SPTL','FTCS','INDA','DFUS','SUB','JNK','SPHQ','SPIB','VDE','SPEM','SCHE','ESGV','VONV','VSS','AMLP','FNDA','SPTM','ESGD','QYLD','IQLT','VCLT','SCHR','TLH','DFAS','FBND','SPAB','EZU','BBEU','ARKK','STIP','IBB','SPLV','SPSB','IJJ','SCHZ','XBI','IGV','JAAA','VYMI','VLUE','PAVE','MGV','FLOT','VPL','BKLN','IJS','DYNF','DFIV','EFAV','CGDV','BBCA','PRF','SPYD','GLDM','TFLO','DFAX','FTSM','VDC','CIBR','VIGI','AVUS','GUNR','SCHH','SCHI','FDN','OMFL','HEFA','PULS','QLD','XLRE','ONEQ','ITA','JIRE','IJT','SHYG','DFIC','IAGG','XLB','IHI','BLV','DFAI','VCR','AVDV','SRLN','ICSH','GBIL','FNDE','DBEF','DFAU','MGC','KWEB','SPTS','FPE','SPTI','VIS','CGGR','FIXD','IOO','DUHP','EWY','EWZ','VPU','AVEM','SPMB','TMF','MCHI','SPGP','DFCF','VONE','DXJ','PDBC','ACWX','VOOV','PBUS','IXN','SSO','SJNK','PGX','IEUR','GDXJ','BBAX','XMHQ','DSI','FXI','EEMV','AVDE','IGM','ACWV','FNGU','IYR','VWOB','BOND','JQUA','VUSB','IDV','XOP','SPHY','SPXL','BBIN','XLG','ISTB','ESGE','IXJ','SCHC','EWT','FDL','QTEC','VOX','DLN','SDVY','SLYV','LMBS','RSPT','JMBS','SHM','JHMM','FV','VSGX','BSCP','BUFR','DFAE','CWB','DON','TFI','GSIE','REET','VNQI','EAGG','IGF','IYH','SUSA','GNR','BSCO','XT','DFEM','SCHK','URTH','FEZ','AMJ','IXC','SLYG','FNDC','ITB','EDV','FHLC','TECL','UPRO','CGGO','HYLB','GVI','DIVO','AVLV','SKYY','BBUS','ANGL','SQQQ','TBIL','ARKB','PPA','BITO','EMLC','MDYG','VAW','NEAR','NUGO','RWL','KRE','HYD','BSCQ','SPHD','SGOL','URA','DFSV','IBTE','JCPB','BOTZ','XYLD','EPI','EWC','TOTL','DEM','CMF','BILS','PZA','DGS','PTLC','IBDP','JMST','VIOO','IBDQ','IYF','EWU','HYMB','KNG','MDYV','DFSD','IBDR','FDVV','CGUS','FELC','PFFD','FELG','EMLP','ICLN','FAS','TDIV','IHDG','AAXJ','PRFZ','IFRA','VTHR','USRT','CGXU','BKLC','NFRA','FLRN','USMC','FTGC','VNLA','IBDS','NRGU','IBTF','OIH','JGLO','TNA','IGLB','UCON','QQEW','FTSL','RPV','IWX','PHO','XAR','BITB','EWW','SLQD','ILCG','IPAC','ICF','XMMO','XHB','IMCG','CGCP','USCA','NVDL','AOR','DIHP','IVOO','IMTM','USCL','UITB','KOMP','FELV','EBND','DES','FMB','XSOE','FXR','BOXX','EPP','GCOW','GSY','FLJP','IYY','HEDJ','AOA','EWA','ITM','GSUS','COPX','ICVT','DSTL','TDTT','EAGL','FENY','DISV','XME','FLCB','LRGF','HACK','QDF','PABU','IVLU','ARKW','IBDT','BSCR','SPYX','PXF','BKAG','DFLV','ILF','FALN','BBMC','IEV','DBC','RPG','ASHR','BBAG','SECT','FIW','FNCL','LIT','AIQ','GUSA','ESML','PFXF','NULV','VRP','TILT','DFGR','FXD','CWI','MLPA','ARKG','REGL','FMDE','IYJ','URNM','TBLL','SYLD','KBE','RWJ','PCY','BSCS','HTRB','KBWB','HDEF','FENI','XSD','HYLS','AIA','BITX','FDIS','FBCG','FTXL','PDP','HYGV','LCTU','RYLD','RWR','FLTR','MUNI','AOM','VIOV','FXL','USO','ROBO','NULG','SPBO','JETS','JGRO','ICOW','IYG','IYE','IYK','TIPX','HYS','BAB','EUFN','FXH','SPIP','QUS','FPEI','TCAF','FLQL','IWL','TAN','QQQE','JMUB','ZROZ','GTO','IBDU','DFIS','FTA','DFNM','PEY','FEX','BSVO','DTD','LABU','SPMO','PJAN','RODM','PXH','FNX','RWO','PTNQ','BULZ','NUSC','FBT','EFIV','SDOG','ARKF','SUSL','SNPE','JPIE','DWAS','RAVI','PKW','FUTY','FTC','XHLF','PICK','MSOS','FSTA','SIVR','FTLS','CORP','DLS','EWL','USPX','DHS','FSIG','USSG','IAUM','SUSC','EQTY','MLPX','JAVA','EWP','SMMD','JMEE','QLTA','FIDU','SPYI','INTF','NTSX','BIZD','RSPH','SMLF','FDLO','SH','FCOM','GSG','JEMA','IYC','IVOG','EWG','AVSC','GRID','BAR','USXF','IBTG','CGW','VTC','AGGY','IQDG','IYT','QEFA','FREL','DFAR','QVML','XCEM','QGRO','PGF','BCI','VTWG','GLTR','YINN','CRBN','ILCV','YEAR','DBMF','GEM','PID','PFFA','DFSU','PPLT','TLTW','IWC','RDVI','CDC','LDUR','BBHY','ILCB','IVOV','BWX','FYX','FLIN','ARKQ','JMOM','IDU','PWV','JHML','USD','KXI','HNDL','GLOV','AIRR','IMCB','FXO','SIL','CCMG','INDY','SUSB','BSCT','IHF','PWB','PREF','JVAL','EPS','PDEC','OUNZ','LVHI','VIOG','EWX','IHAK','PFEB','FQAL','IBDV','TSLL','CATH','PAUG','PSK','VRIG','PWZ','KSA','PJUL','MOO','XMLV','EZM','SVOL','BSJP','XNTK','ONEY','VTWV','BBRE','BUFD','UNG','PSI','IVOL','TDTF','FRDM','QCLN','PNOV','XSVM','SPLB','FPX','SMMV','DBA','SCHY','FVAL','IBDW','BNDW','BUG','SPGM','FFEB','IBTH','PNQI','RDIV','KIE','DIVI','SMDV','TMFC','IEO','EUSA','GSID','PCEF','SDIV','OUSA','RWK','SMIN','FJUL','FDEC','BLOK','FNDB','LTPZ','GIGB','FJAN','ACIO','POCT','TSLY','JHMD','LGLV','PSEP','FLGV','VFVA','QQQJ','SFY','PMAR','DPST','RSPN','HAUZ','SILJ','AGZ','UYG','AVIG','BSJO','ROM','JHEM','EWQ','PHB','DVYE','GOVI','IRBO','NYF','JPMB','SSUS','JCPI','ONEV','PFM','IGRO','IHE','LVHD','BSCU','MMIT','GSEW','WCLD','FLGB','SOXS','IAK','IAT','EES','RPAR','DMXF','GWX','COMT','DNL','INFL','KRMA','SCHQ','MEAR','FMAR','AAAU','IYM','SPHB','EQWL','IMCV','BKIE','ILTB','FXN','UDOW','REZ','REM','DIV','AOK','EUSB','APUE','CLOU','FLQM','FAUG','FJUN','EQAL','DBEU','VOTE','DRIV','SPXU','FMHI','QAI','IGEB','CQQQ','SDS','FNOV','RSPG','USTB','ULST','NUGT','PSQ','HODL','UCO','BALT','GSST','EMGF','FIVG','VUSE','BBEM','DWM','KOKU','EXI','ISCF','OUSM','ECH','DFEV','SMMU','EMCR','VNM','IQDF','HFXI','IGE','ISCG','ROBT','TCHP','JPEF','FGD','IBDX','PLDR','FTHI','FMAY','PPH','TZA','RLY','VCEB','NANR','IBMN','SPXS','CFA','FDRR','TLTD','PTF','FMAT','IGOV','FOCT','BOIL','PDN','BRRR','FESM','PSC','STPZ','DOL','VFMO','PAPR','GUSH','SPUS','HYDB','DFIP','GSSC','XRT','EIDO','BUFC','BUFQ','FSEP','EWH','XLSR','FLIA','STRV','GGUS','TPYP','IBMM','BSJQ','DUSA','IBHD','JPSE','BBSC','BKMC','IAI','DJP','GXC','CONL','SRVR','IBHE','IMFL','SEIV','NBOS','CFO','DWX','SEIM','PJUN','TAFI','RSPS','IBTI','EELV','FAPR','MMIN','BBH','YYY','FM','IBMO','DRSK','TMFG','MLN','LEMB','RECS','GXUS','JPUS','SMIG','HELO','KJAN','RSPD','FXG','QLTY','IUS','FXZ','AVRE','FCG','PMAY','JHSC','RING','TAXF','MDIV','CMBS','CEMB','GCOR','EWI','QQH','EWS','BSCW','METV','CSM','BUYW','IDLV','ISCV','DCOR','ROUS','JPIB','HLAL','BSCV','VBND','FEM','FDT','IBMP','DIAL','PVAL','FLDR','FSMB','HEZU','PTMC','NUDM','TECB','XONE','WIP','EEMA','DFSI','FLCA','JANW','JSCP','AVES','BUFF','IQSU','DDM','AGQ','TDVG','NVDX','TTAC','GSPY','HMOP','IXG','GII','BTCO','SCHJ','NUMG','VTES','LGOV','HYEM','URTY','GFEB','QDEC','SIXH','NXTG','CAPE','DBJP','AMZA','NUSI','VIDI','XSW','GRPM','GVUS','XDEC','ERX','JPME','JPIN','WINN','CONY','XSMO','OMFS','AIVL','BIBL','EDIV','KBWD','JBBB','SOXQ','QMAR','EMHY','IBHF','HYBB','QDEF','EEMS','FLCO','LCTD','EMQQ','PBW','LGH','JSMD','IPAY','GINN','CXSE','FINX','CDL','EMHC','SIZE','NVDY','EWJV','DRLL','AVIV','AVSF','VGSR','NUMV','BGRN','USOI','NUBD','DGRS','VLU','XSLV','DJUL','FXY','JUST','QVAL','DTH','UTWO','SMRI','ATMP','CARK','URNJ','DDEC','DIVB','AVGE','SKOR','GMF','EQL','SMOT','REMX','VFQY','CWEB','EZBC','UUP','GNMA','EYLD','HEQT','SDOW','VXX','NAIL','MODL','SHYD','FLBL','FEMS','DFAW','JPRE','RFG','XES','FNY','QGRW','NTSI','KCE','KBWP','CSB','DAUG','QJUN','PSCT','RFV','HEWJ','TBT','HAPI','UBND','LONZ','LQDH','JTEK','QDPL','SVXY','IBMQ','JUCY','GQRE','JPEM','SDG','CGBL','QVMM','BNKU','PALC','RWX','KRBN','JNUG','FLKR','FDHY','PKB','DJD','PEJ','TSLT','EPOL','FNGO','PSFF','DWLD','PIO','RSPM','USMF','MJ','IEZ','QQQY','CVLC','FWD','MNA','PJP','HCMT','SNSR','FFLG','FYLD','BKHY','CLOI','DFEB','IGPT','XHE','GJAN','TBFC','IGHG','APIE','EIPX','DMAR','ALTL','RSPF','TDV','SPEU','EWD','TBFG','NJAN','EDEN','WTV','IXP','DFSB','GDIV','QSPT','IDHQ','DJAN','IDOG','OSCV','UVXY','RXI','FYC','DNOV','FLHY','GJUL','TLTE','MORT','NBCM','EFAX','EZA','KMLM','VFMF','TDSC','DDLS','BJAN','DDWM','PRN','HCRB','FNGS','EWN','PFLD','IDRV','QID','DBND','MXI','FTQI','VFLO','KCCA','AGOX','WWJD','QTUM','VSDA','IBD','ESPO','PFUT','AVSU','XVV','EDOW','HTAB','KORP','EWM','THD','COM','MFDX','GMAR','DFJ','UMI','PFFV','RZV','DBO','ARKX','SMB','GDXU','RSPU','FFLC','UIVM','PBE','UAUG','TPHD','WTAI','INCO','CGDG','AUSF','MLPB','NUEM','PSCE','PIN','GAL','GBF','SHE','XMPT','HEGD','VALQ','TJUL','FNK','IETC','MAGS','BTAL','SIXA','PSP','QINT','TPLC','IYZ','DFSE','EWZS','BRNY','CHIQ','CMDY','HYGH','ISCB','WFHY','ACES','UWM','SEIQ','LRGE','FLV','FLRT','CMDT','USVM','IBTJ','FXU','FLTW','SGDM','PSCH','BFEB','FISR','XMVM','TMV','PALL','MOTI','FTXN','XPH','INDS','AVLC','TACK','ESGB','BSMP','HYDW','FUMB','SRET','FCOR','IBTK','JSML','RTH','HAWX','FEBW','FCAL','CURE','FDG','FEP','XJUN','FSMD','QMOM','CNRG','TOK','IAPR','IJAN','BSMO','FBGX','FJP','IQIN','BCD','CZA','CCOR','POWA','UGL','WTMF','LSAF','WOOD','GJUN','TUR','ESG','UJAN','XEMD','CLSM','PTRB','GOVZ','TMFM','BUFG','WDIV','FXE','SCO','DEHP','FUNL','DGT','ESGA','IBTM','GDEC','PSCI','QLC','KBWY','FLMI','SPUU','IFV','GREK','IQSM','FYT','NJUL','MARW','FTGS','MUSI','FAD','BJUL','IGTR','XTN','IPO','PRAE','DFE','ERTH','RWM','FDMO','IQSI','KBA','ETHO','FAN','BSJR','DFNL','WEBL','AGGH','NFTY','USNZ','QVMS','BMAR','FTRI','TUG','ECML','SQEW','DJUN','FLSP','ARGT','CNYA','SPSK','ISVL','SMLV','DSEP','IBUY','CSML','PTBD','ESGG','DOCT','BAUG','CAOS','SPDN','CEFS','GTEK','XJH','ISMD','THNQ','BOCT','BLES','XMAR','GVIP','FDM','SWAN','INCM','BSMQ','BNDC','QUVU','JSTC','UEVM','TECS','DAPR','LQDW','BSJS','GNOV','IDMO','EIS','IMTB','EMNT','BWZ','STOT','DOG','SFLR','YLD','IJUL','TSPA','IBTL','SLVP','BYLD','HYZD','DFVX','DINT','USCI','YJUN','FVC','NOCT','AGZD','DFEN','OPER','DVAL','USDU','TSME','BKCH','SPFF','GVLU','SOCL','ADME','IBND','KLIP','FLBR','CTA','BSEP','PPIE','FDD','FEPI','QLV','FEMB','FPXI','BBLU','FAB','LSAT','EMC','DEUS','VIXY','TEQI','IVAL','CPER','PTIN','ULVM','EJAN','WEAT','SIXL','ONEO','BAPR','XCOR','JULW','IDUB','KJUL','GDMA','IWMY','BDEC','DBAW','QQXT','DMAY','PGJ','OACP','PTH','FCPI','FTSD','HYBL','FXF','ROOF','PXE','XSHQ','NLR','MFUS','MVV','FDTX','CWS','OGIG','LSGR','DBP','GMOM','NVDU','FLTB','FMF','BFOR','DUBS','AUGW','FAAR','SMOG','MGMT','IDNA','HKND','OVL','JEPY','SRHQ','AVSD','YANG','AVMU','MINO','BITQ','XSVN','OILK','HEEM','LABD','TOLZ','RUNN','ENFR','EMBD','KLDW','UMAR','SCJ','PINK','VSMV','SCYB','DIM','MRSK','TMAT','PGHY','OCTW','CAML','DGRE','OCIO','HYHG','PFIX','NETZ','PRNT','FAZ','GLIN','VEGI','XSEP','CHAT','MSTB','GHYG','BNO','BGIG','HERO','PBJ','PIE','TBUX','GHYB','RFDI','MJUS','HGER','SARK','SHDG','HAPS','PIZ','XTEN','EPHE','PPTY','HYXF','CHGX','FNGD','SELV','UCRD','FBCV','JANT','PICB','SIHY','RVNU','CACG','HSCZ','SRTY','GLOF','PHYL','GSEP','JXI','USSE','GCC','HYFI','SEIX','MILN','PBD','BSMR','TPSC','TDSB','STNC','EIRL','MPRO','FLLV','BKCI','FRI','KAPR','UPGD','GTIP','IYLD','UGA','INTL','DALI','SURI','OAIM','HFGO','COWG','DBB','PHDG','XXXX','IFGL','FIVA','DIG','IEUS','GVAL','ONOF','SPVU','OWNS','CVY','RSPR','IOCT','EPRF','QYLG','SBIO','FTXO','HDUS','CLOA','QWLD','WGMI','TRTY','BKSE','TOTR','FLMB','AMJB','SLX','FLCH','NZAC','HAP','DEW','DAPP','DCRE','CSMD','AIEQ','GSEE','IZRL','FBOT','AFIF','VSLU','BMAY','MDCP','EPU','AMDY','BNOV','ZECP','VFMV','LDSF','HSRT','CGIE','WCBR','MSOX','SGDJ','BUSA','BUFB','NAPR','LEGR','SIXJ','GAUG','KARS','DWUS','CGV','SJB','NBSM','FLRG','SIO','TPIF','BUFT','JOET','IGBH','IDVO','GNOM','FEBT','TSLQ','UTEN','DBEM','TIPZ','RZG','OALC','FILL','XITK','RIGS','SHYL','SLVO','SPRE','BMVP','FDIF','FICS','GOCT','BITI','FDIG','BSMS','XBJA','DTEC','XTWO','KOLD','APRW','MART','MMTM','ONLN','FIDI','HERD','BKEM','YDEC','MBOX','TMSL','SPD','IBHG','PIFI','GMAY','BRZU','MFEM','FFTY','AESR','CIL','UOCT','PUTW','PZT','UFEB','VEGN','ACTV','BIB','EJUL','DRV','SSPY','RXL','BETZ','ZALT','BJUN','SVIX','GAPR','FMCX','RISN','FDV','PBP','FDIV','FPAG','GRNB','UDEC','PXJ','CVIE','TAGG','TBJL','BSMT','TBF','DBE','JIG','JPXN','XJAN','BATT','IGLD','HYMU','ENZL','MIDU','QPFF','AWAY','GOAU','FDLS','NFLT','STXE','ECOW','XHS','KOCT','GBUY','TAIL','HUSV','DGP','IMOM','OSEA','PSL','UPAR','QVOY','ITEQ','COMB','PFFR','TMFS','LCR','FSZ','EFAD','ENTR','VIXM','INDL','UMMA','PSTP','ELD','ABEQ','DECW','DJIA','AMNA','MBCC','MSFU','UBT','PCGG','BSMU','ACVF','FMAG','AMUB','SGLC','PSCC','VAMO','ROE','TOUS','FCVT','SFYX','EMTL','ASIA','LGRO','LRGC','YMAX','PJFG','LEXI','MOHR','RAAX','AJAN','EEMX','QABA','MBSD','ISHG','DXD','RSSB','EDC','UJUL','ACSI','QAT','DUST','PSMD','BLCN','GQI','FCLD','EBIZ','XFIV','FID','XBJL','KEMX','IQDY','NUHY','IG','CVRD','DWAW','RFDA','SDVD','FCTR','TGRT','SVAL','UVIX','BAMG','AMZU','NRGD','EVX','SIXS','BAMV','TFPN','KWT','UAPR','EMXF','BKUI','TGRW','IFED','INKM','SPRX','XJR','XBOC','EAPR','AIVI','TPLE','CLSE','CRDT','AGQI','EETH','DYLD','PXI','GPIX','EUDG','TPHE','OARK','RSPC','PBDC','BUFZ','FBL','LCG','HDGE','BINV','SPC','FLEE','PAWZ','XYLG','BAMD','XBAP','MAMB','AVSE','BUZZ','NVDS','TVAL','QFLR','HIPS','PSR','BKF','YMAR','ARB','PYZ','GLDI','AMTR','IPKW','AUGT','MMLG','DRN','OILU','FLLA','IBHH','YALL','FXA','ISPY','FFOG','CORN','DXJS','JDST','DIVL','NANC','SAMT','TTAI','FXB','MINV','ISRA','LCLG','FLN','HIBL','QARP','FDEV','CHAU','MKOR','SPYC','AVGV','ICLO','WBIY','WUCT','HYGW','TUSI','DIVZ','FIBR','CDX','HTEC','PEXL','RETL','TARK','PBTP','SPDV','USL','URE','USEP','UDIV','DEED','GGLL','PPI','UDN','VPN','AMZY','FKU','GTR','TSLS','SHOC','FGDL','CUT','RUFF','SMCO','STXG','QQQI','DSMC','NETL','SIXF','PSMJ','DYTA','BDRY','PEZ','PY','TWM','CRPT','RIET','GLRY','PGRO','OVT','IEDI','DURA','DEEF','HRTS','RPHS','EWO','AGNG','ECON','VEGA','PFRL','TBG','MEM','HSUN','SPXN','DRUP','XDSQ','CSD','MSSS','FXC','CAMX','LEAD','ESGS','MLPR','GEMD','XTL','LQIG','BSMV','XRLX','UNOV','ADPV','FTXG','NXTE','PFI','XHYI','FFSM','DAX','AMID','KGRN','FITE','TDI','XCCC','XBB','FBY','XISE','BTF','FIGB','UTES','NTSE','MFUL','LOUP','KBWR','EMCB','CVMC','APLY','UJUN','PABD','FMED','WOMN','PRAY','MVPA','EBLU','SFEB','BSTP','XOCT','YOLO','USAI','SBND','DRIP','AAPU','MISL','ICAP','RISR','PFIG','SPXT','SFIG','CTEC','PAMC','SRHR','XTRE','VETZ','EDOC','TRND','ECNS','GAA','RINC','RAFE','GINX','XJUL','BSJT','CVSB','FLJH','DECT','WBIG','THLV','SURE','STLG','FEDM','EASG','SPXE','ISEP','OVB','NORW','FDEM','BBCB','DVOL','BUL','PSET','MSMR','HYXU','CSF','RSEE','SNAV','MBND','IIGD','QLVD','UTSL','OAEM','MTGP','CRAK','NURE','KURE','MSTY','MARB','FDFF','ARCM','RAYE','BSJU','RAYD','FLMX','FEUS','UYM','DECZ','MSFO','SFLO','FLSW','DALT','CCRV','SZNE','DFRA','DFNV','TYA','HTUS','FDCF','TUGN','FEDL','YLDE','GPIQ','QEMM','ASEA','FNGG','SDEM','BTEC','BJK','LQDI','WBIL','ZIG','PAB','EUSC','NACP','NFLY','DDIV','UMAY','AMND','GSFP','LBAY','PSFD','STXK','ZHDG','BOAT','JHCB','PSCD','IDGT','TYD','PSMO','FFIU','HYRM','GURU','UBOT','BSMC','FTHF','TTT','MVFG','XC','LRNZ','IHYF','DWMF','QULL','TMFE','XRLV','GXG','DJCB','HOMZ','RULE','YCL','QCON','KORU','DVYA','REVS','PTEU','SRS','FEIG','AZTD','IVVB','GAMR','AFK','EINC','SAUG','JHMB','KNCT','EWUS','MBNE','MARZ','PSMR','DSCF','FTXR','PP','SPVM','QQMG','UAE','RNRG','OCTT','DUSL','HYSA','WBIF','SSFI','APRJ','XHYC','DBEZ','SHLD','XHYE','SIXO','MAYW','ISWN','XHYF','DYNI','RNSC','STCE','HYDR','HAIL','CARZ','ROMO','OEUR','WBND','EUO','CNBS','SMHB','SEPW','STXV','BCHP','PUI','SOVF','JHPI','JHPI','FDRV','CIZ','EOCT','STXD','SHAG','OBND','AAPD','SAGP','SCDL','EFIX','PVI','SXQG','WPS','VXZ','TRFM','SEMI','AOTG','YSEP','VPC','RFEM','IWDL','CRTC','PAPI','RDFI','CRUZ','FLQS','HIBS','RAYC','IWFL','BAMA','AHLT','PLTM','CDEI','JULT','GDXD','XHYT','DEEP','IVVM','SHUS','HEWG','MVFD','DBEH','DSTX','ALTY','LCF','UFO','USML','ITAN','IBHI','GHTA','GOEX','FLAX','IAUF','VNSE','XFEB','FPFD','TWIO','UMDD','LQDB','INNO','MCSE','MDPL','AHYB','FXED','INQQ','SOYB','HDMV','BAMO','GFGF','SIFI','SSXU','BDCX','OVLH','LOWV','HISF','DEMZ','LDEM','SAA','BAMY','DVND','IVES','NVBW','SMCP','XRMI','FRTY','LKOR','WLDR','YCS','SQY','PHEQ','DIVS','EALT','DFHY','GGME','THY','MGNR','KVLE','YMAG','SCRD','IHY','FCEF','TRFK','WUGI','ROSC','CVAR','IMSI','XDQQ','XSHD','SDSI','ILDR','NULC','KNGZ','BNDD','IBBQ','NVD','DOGG','SPTE','NUAG','GOOY','CVSE','PBL','SMIZ','AEMB','FMET','BCUS','SQLV','WANT','TPOR','IWLG','NBCT','CNCR','ENOR','WFH','SNOV','HDG','AIYY','FORH','EAOA','ESUS','BRF','APRT','VMOT','BITS','ADFI','IDX','UCIB','DIVY','QTJA','CSA','MYLD','FLAU','XTWY','KONG','FIG','ECLN','FCSH','FRNW','TMFX','PSFJ','BDCZ','JDVI','LFEQ','AADR','CCSO','EMM','EURL','EPV','FCUS','BTR','CBON','FDAT','GDOC','FHYS','FDNI','XNAV','ZSL','MTUL','BALI','EDOG','GSC','EMSG','GRN','XHYH','JULJ','MOON','DVLU','QDIV','XNOV','GDVD','RFCI','TSLZ','EFZ','BCIM','MMCA','BLLD','QPX','GXTG','WLTG','FMQQ','BBLB','APRH','XIDE','REIT','UXI','PIT','KSTR','SIXP','PQDI','TEMP','SMAY','CLIA','RFFC','IQDE','JULH','JUNW','PPEM','GSJY','KOCG','FLJJ','EMMF','JANJ','IBLC','MRNY','GMET','IMAR','BLDG','EFNL','RHCB','NUSA','JHMU','MSVX','OCTJ','APRQ','SAEF','BGLD','WZRD','FTDS','UTRN','TXS','PWS','GYLD','EMIF','MAGA','IPDP','PSFM','ARKA','HIDE','OVM','TDVI','GK','HJEN','TIME','HDRO','EWK','DUG','FTXH','MCH','SHRY','FLEU','QQQN','XTJA','PSCX','IQM','SPXV','ASHS','WISE','RINF','ERY','RTAI','RENW','AAPB','IWML','EDZ','PCIG','AVMV','FLGR','BBP','CPAI','CEFD','WTRE','MUSQ','WEBS','PSCF','ROAM','NERD','REK','CID','GGM','MINN','EMSF','OCTH','XUSP','SPBC','SYUS','BDGS','AMAX','LOCT','FLSA','EAOR','RSHO','INMU','CANC','FOVL','PST','XTOC','SIMS','EQRR','FSYD','HSMV','XPND','GDE','SMI','GREI','KEMQ','MOTG','CEFA','DTRE','RNMC','MVRL','OVF','UCC','RNEM','SSLY','RODE','LSST','SEPZ','MEDX','NOVZ','FMNY','AAA','EUM','SKF','AMOM','FXP','IVVW','FDGR','DGIN','XOMO','FEUZ','BKGI','XAUG','AVDS','UNL','VRAI','EMFM','ROKT','SPAQ','ATFV','CANE','AVNM','MEXX','IWMW','PSCM','AVMC','NBDS','SFYF','HYIN','LJAN','AUGZ','JPSV','TSLR','SUPP','SPAX','PSFO','SDCI','BTEK','TBX','ESGY','EET','SHRT','MOOD','PSCU','CNXT','KHYB','IDEC','PILL','BOUT','MSTQ','GLL','OILD','ULTR','UST','BKIV','TFJL','BBSA','TAGS','INAV','CPII','JANQ','PSCJ','CVRT','OILT','HOCT','DARP','XB','TYO','BABX','FPA','PJFV','JIVE','PYPY','IFEB','KRUZ','SEPT','NVDQ','VNAM','GABF','WFIG','KOIN','QTJL','SEF','FSST','DFVE','FPRO','QRMI','EMDV','BKWO','HAPY','SPCX','HIDV','DIEM','RHRX','CBSE','MIG','QTOC','MMSC','USBF','MAYT','PSCW','HYUP','RORO','OOTO','LVOL','HDLB','QQQA','ELQD','LOPP','MEMX','PJBF','FGM','DWSH','QLVE','TXSS','OCTQ','JPMO','ESMV','SETM','RDOG','DISO','KEUA','AETH','FSBD','SPWO','BBC','GCLN','HEAT','NDIV','BNKD','MNTL','FFND','NRES','IPPP','SROI','BUYZ','PJFM','XHYD','HDAW','PJIO','COPP','GERM','DEFI','UPW','MOTO','GSEU','NDVG','OVS','EZJ','TGLR','NDIA','ASET','SMDY','WINC','TOKE','CLIX','MDLV','DFND','AVEE','HCOM','HLGE','PFFL','BTOP','BERZ','CBLS','JULZ','HIYS','MRGR','JCHI','XTJL','EFO','FTAG','MEDI','GMUN','ISDB','FLHK','NVBT','IBOT','HELX','RFEU','EFAS','AQWA','QTAP','NRSH','FDHT','VABS','PEX','JUNT','VWID','ABCS','RHTX','SATO','SPXB','BYRE','GSIG','JANH','RBLD','KFVG','FLYU','BCDF','IWFG','NVDD','ERET','FTWO','HART','RSPE','SPMV','FDCE','GFOF','PSCQ','JDOC','XPP','JULQ','BFIX','AVMA','HJAN','SSPX','ITDC','OFOS','EUDV','MBBB','TMDV','FDTS','QRFT','LUX','CAFG','INFR','TYLG','SYII','QDTE','ANEW','IBAT','JHDV','EVMT','NUDV','BETH','IQQQ','QQQS','RYLG','JHID','APRD','TOLL','LRND','INDF','EEMD','DMCY','SNPG','ESGN','DWCR','IWIN','SPYT','COPJ','EAOK','VICE','HYTR','AFMC','SXUS','PEMX','MIDE','GDMN','NWLG','ITDB','WEIX','PSIL','CGRO','IRBA','GDEF','XVOL','XDJA','BLCV','WRND','XDOC','CEW','IDAT','EQUL','JANZ','BLCR','UGE','DIVD','BTHM','NSCS','MSFD','ICOP','WEED','CLDL','ULE','UJB','XDAT','BZQ','EV','FBZ','MOTE','RAYS','GOOX','YXI','INDE','IVRS','CHPS','GPOW','NSI','RITA','FCA','CCEF','VCLN','RJMG','SBB','DIVP','JRNY','ITDE','NTZG','MKAM','MID','TCHI','ITDD','EEMO','XDJL','XDAP','USVT','KALL','LALT','FEBZ','XTAP','MSFX','IWTR','SNPV','DMAT','PSWD','BETE','EAOM','MEMS','NBGR','SEA','EATV','EGUS','UDI','LITP','VMAX','MYY','ITDF','HQGO','SPUC','GBLD','CLNR','AFSM','NBCC','EEV','QQJG','DAT','EMDM','MVPL','XDTE','GGLS','PWER','SPDG','IPOS','NSCR','SYNB','ISZE','SNPD','SSG','VERS','JCTR','KTEC','AGRH','BNE','SPCZ','OCEN','BILD','FPXE','GCAD','IVEG','PTL','ROIS','AVIE','BNGE','SOLR','BBBI','TSL','OAIA','GAST','IQRA','JAND','TINY','BBBL','SHNY','BPAY','NBCE','EFRA','BEEZ','SKRE','EMTY','OCTD','RYSE','APRZ','IVRA','BWTG','FEEM','UPV','QQQU','AFLG','EVNT','ARKD','UCYB','UBR','MAYZ','AWEG','ESIX','ISHP','UPGR','CTEX','EVUS','LTL','VNMC','JULD','GGRW','KROP','AFTY','WBAT','BEDZ','LQAI','BMED','REW','OBOR','QMID','BBIB','ITDG','RVRB','EWV','FLOW','PBMR','COAL','OCTZ','AMDL','EVAV','GLIF','FDWM','BDVG','DZZ','IBRN','MAPP','VCAR','CARU','KLNE','FSLD','USE','FLUD','KEM','JETU','FEBP','SKYU','FLDZ','ERUS','SPQ','NIKL','VIRS','PBFB','TSDD','WTIU','IRVH','CFCV','ITDI','BIS','INOV','WCEO','NVIR','ETEC','BRAZ','REAI','VSHY','IRTR','YUMY','ADIV','JFWD','JETD','JOJO','HCOW','BECO','GOLY','MDEV','PTEC','GHEE','FSEC','JANP','CRED','KNGS','IWFH','IOPP','EATZ','MRCP','BLKC','JHAC','SCAP','DGZ','AVXC','BWEB','SMDD','XTR','JUNZ','BBSB','AHOY','ITDH','AAPX','AMZD','KPOP','QOWZ','DIVG','HYLG','JRE','ITDA','DYLG','ILIT','MVPS','KSEA','KDIV','TILL','XCLR','KLXY','XXCH','FYLG','HYGI','PBJA','FIXT','WDNA','QSWN','KBUF','QCLR','SCLZ','FAIL','WNDY','QTR','KPRO','EKG','BBBS','AGIH','JPAN','QYLE','DULL','QQQD','AVNV','WTID','NZUS','XIMR','SDD','RNWZ','ALUM','ION','CARD','DWAT','NUKZ','VWI','HAUS','EMFQ','DMDV','MCHS','SUPL','REC','CRIT','LBO','MPAY','LGHT','RATE','AMDS','SETH','FLYD','ARVR','RNEW','EMCC','TINT','ADVE','IRET','MZZ','ZSB','FDVL','AUMI','PCCE','TSLP','SIJ','SPAM','SMLE','CLOD','LUXX','XYLE','QSML','SCC','CANQ','BITC','BCIL','ZZZ','FTIF','FCFY','KDRN','DIP','SHPP','DESK','RXD','FFLV','AMZZ','MSFL','FDND','EAFG','BBIP','SMN','BULD','MAKX','MRAD','TSLH','AMZP','SDP','ODDS','MSFY','SZK','KARB','INC','CETF','LNGG','OND','HYKE','EFU','NFLP','SMCF','USRD','DVDN','USCF','CZAR','GOOP','GSIB','AIRL','FINE','AAPY','ROYA','LNGZ','PSLV','BITW','MAGX','ULTY','USDX','MAGQ' ]
    
    selected_tickers = st.multiselect(
        "Sélectionnez des tickers à comparer",
        options=popular_tickers,
        default=['AAPL', 'MSFT', 'TSLA'],
        help="Sélectionnez jusqu'à 5 tickers"
    )

with col_chart2:
    timeframe = st.selectbox(
        "Timeframe",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y'],
        index=3,
        help="Période d'analyse"
    )

with col_chart3:
    if st.button("📊 UPDATE", use_container_width=True, key="update_chart"):
        st.cache_data.clear()

# Génération du graphique
if selected_tickers:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Couleurs Bloomberg style
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00', 
              '#FF1493', '#00CED1', '#32CD32', '#FFD700']
    
    # Stocker les données pour la corrélation
    correlation_data = pd.DataFrame()
    
    with st.spinner('📊 Chargement des données...'):
        for idx, ticker in enumerate(selected_tickers[:10]):  # Limite à 10 tickers
            hist = get_historical_data(ticker, timeframe)
            
            if hist is not None and len(hist) > 0:
                # Normaliser à 100%
                normalized = normalize_to_percentage(hist['Close'])
                
                # Stocker pour corrélation
                correlation_data[ticker] = hist['Close']
                
                # Ajouter la courbe
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'<b>{ticker}</b><br>%{{y:.2f}}%<br>%{{x}}<extra></extra>'
                ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title=f"Performance Comparison - {timeframe.upper()}",
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
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Ligne horizontale à 100%
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="paper",
        y0=100, y1=100,
        line=dict(color="#666", width=1, dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistiques des performances
    st.markdown("#### 📊 PERFORMANCE SUMMARY")
    
    cols_perf = st.columns(min(len(selected_tickers), 10))
    
    for idx, ticker in enumerate(selected_tickers[:10]):
        with cols_perf[idx]:
            hist = get_historical_data(ticker, timeframe)
            if hist is not None and len(hist) > 1:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                perf = ((end_price - start_price) / start_price) * 100
                
                st.metric(
                    label=ticker,
                    value=f"{end_price:.2f}",
                    delta=f"{perf:+.2f}%"
                )
    
    # ===== MATRICE DE CORRÉLATION =====
    if len(correlation_data.columns) > 1:
        st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 CORRELATION MATRIX")
        
        # Calculer la matrice de corrélation
        corr_matrix = correlation_data.corr()
        
        # Créer la heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0, '#FF0000'],      # Rouge pour corrélation négative
                [0.5, '#000000'],    # Noir pour 0
                [1, '#00FF00']       # Vert pour corrélation positive
            ],
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10, "color": "#FFAA00"},
            showscale=True,
            colorbar=dict(
                title="Corr",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0', '0.5', '1.0']
            )
        ))
        
        fig_corr.update_layout(
            title="Correlation Matrix (1.0 = parfaitement corrélé, -1.0 = inversement corrélé)",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                tickfont=dict(color='#FFAA00', size=9),
                showgrid=False
            ),
            yaxis=dict(
                tickfont=dict(color='#FFAA00', size=9),
                showgrid=False
            ),
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Interprétation
        st.caption("""
        **📖 Comment lire la matrice :**
        - 🟢 **Vert (proche de 1.0)** : Les actions évoluent ensemble (forte corrélation positive)
        - ⚫ **Noir (proche de 0)** : Pas de relation claire
        - 🔴 **Rouge (proche de -1.0)** : Les actions évoluent en sens inverse (corrélation négative)
        """)

# ===== FOOTER =====
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE | LAST UPDATE: {last_update}
</div>
""", unsafe_allow_html=True)
