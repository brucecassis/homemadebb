import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time
import requests
"""
from auth_utils import init_session_state, logout
from login import show_login_page

from adsense_utils import add_header_ad, add_footer_ad
# =================================

from auth_utils import init_session_state, logout
from login import show_login_page
"""
init_session_state()

if not st.session_state.get('authenticated', False):
    show_login_page()
    st.stop()

# Votre code existant continue ici...

# =============================================
# AUTO-REFRESH TOUTES LES 3 SECONDES
# =============================================
#from streamlit_autorefresh import st_autorefresh

#Rafraîchissement automatique toutes les 3000ms (3 secondes)
#count = st_autorefresh(interval=3000, limit=None, key="market_refresh")

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

        /* ANTI-FLASH: Empêcher l'assombrissement */
    .stApp {
        background-color: #000000 !important;
        transition: none !important;
    }
    
    .main {
        transition: none !important;
        animation: none !important;
    }
    
    section.main > div {
        animation: none !important;
        opacity: 1 !important;
    }
    
    /* Masquer le "Running..." */
    .stApp [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }

        /* STABILISER LA PAGE - Empêcher les sauts */
    .main .block-container {
        min-height: 100vh !important;
    }
    
    /* Fixer la hauteur des metrics pour éviter les sauts */
    [data-testid="stMetric"] {
        min-height: 80px !important;
    }
    
    /* Stabiliser les graphiques */
    .js-plotly-plot {
        min-height: 500px !important;
    }
    
    /* Empêcher le scroll auto pendant le refresh */
    html {
        scroll-behavior: auto !important;
    }
    
    /* Skeleton loading - Préserver l'espace pendant le chargement */
    .element-container {
        min-height: fit-content;
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

     /* CACHER COMPLÈTEMENT LE TEXTE keyboard_double_arrow_right */
    [data-testid="stExpander"] details summary span[data-testid="stExpanderToggleIcon"] {
        display: none !important;
    }
    
    /* Cacher aussi le texte directement */
    [data-testid="stExpander"] details summary > div > span {
        font-size: 0 !important;
    }
    
    /* Garder uniquement le titre visible */
    [data-testid="stExpander"] details summary p {
        font-size: 11px !important;
        color: #FFAA00 !important;
    }
    
    /* Ajouter la flèche personnalisée */
    [data-testid="stExpander"] details summary::before {
        content: "▼ " !important;
        color: #FFAA00 !important;
        font-size: 14px !important;
        margin-right: 8px;
    }
    
    [data-testid="stExpander"] details[open] summary::before {
        content: "▲ " !important;
    }
    
    /* Si ça ne marche toujours pas, forcer avec visibility */
    [data-testid="stExpanderToggleIcon"] {
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# WIDGET TRADINGVIEW - TICKER TAPE
# =============================================
st.markdown("### MARKET TICKER TAPE")

# Interface discrète pour sélectionner les tickers
with st.expander("CONFIGURE TICKER TAPE", expanded=False):
    ticker_tape_options = st.multiselect(
        "Sélectionnez les tickers à afficher dans le bandeau",
        options=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT",
            "DIS", "NFLX", "BA", "GE", "GM", "F", "T", "VZ", "INTC", "AMD",
            "BTCUSD", "ETHUSD", "XOM","QQQ","EURUSD", "GOLD", "SLHN", "HSBC", 
                "VIX","EURUSD", "GBPUSD", "USDJPY", "GOLD", "SILVER", "CRUDE_OIL"
        ],
        default=["XOM","QQQ","BTCUSD", "EURUSD", "GOLD", "SLHN", "HSBC"],
        help="Choisissez jusqu'à 20 tickers"
    )

    # Options d'affichage
    col_widget1, col_widget2 = st.columns(2)
    with col_widget1:
        show_market = st.selectbox(
            "Marché",
            options=["stocks", "forex", "crypto", "futures"],
            index=0,
            key="widget_market"
        )

    with col_widget2:
        color_theme = st.selectbox(
            "Thème",
            options=["dark", "light"],
            index=0,
            key="widget_theme"
        )

# Construire la liste des symboles pour TradingView
if ticker_tape_options:
    import json
    symbols_tv1 = []
    symbols_tv2 = []
    
    for i, ticker in enumerate(ticker_tape_options[:20]):  # Limite à 20
        symbol_entry = None
        
        # Déterminer le format du symbole selon le ticker
        if ticker in ["BTCUSD", "ETHUSD"]:
            symbol_entry = {"proName": f"BINANCE:{ticker}", "title": ticker}
        elif ticker in ["EURUSD", "GBPUSD", "USDJPY"]:
            symbol_entry = {"proName": f"FX_IDC:{ticker}", "title": ticker}
        elif ticker in ["GOLD", "SILVER"]:
            symbol_entry = {"proName": f"TVC:{ticker}", "title": ticker}
        elif ticker == "CRUDE_OIL":
            symbol_entry = {"proName": "TVC:USOIL", "title": "OIL"}
        elif ticker == "SLHN":
            symbol_entry = {"proName": "SWX:SLHN", "title": ticker}
        elif ticker == "XOM":
            symbol_entry = {"proName": "NYSE:XOM", "title": ticker}
        elif ticker == "HSBC":
            symbol_entry = {"proName": "LSE:HSBA", "title": ticker}
        elif ticker == "QQQ":
            symbol_entry = {"proName": "NASDAQ:QQQ", "title": ticker}
        elif ticker == "VIX":
            symbol_entry = {"proName": "TVC:VIX", "title": "VIX"}
        else:
            # Par défaut, utiliser NASDAQ
            symbol_entry = {"proName": f"NASDAQ:{ticker}", "title": ticker}
        
        # Ajouter au bon tableau (max 10 par widget)
        if symbol_entry:
            if i < 10:
                symbols_tv1.append(symbol_entry)
            else:
                symbols_tv2.append(symbol_entry)
    
    # Ajouter les indices principaux au deuxième widget (toujours affichés)
    symbols_tv2.extend([
        {"proName": "NASDAQ:IXIC", "title": "Nasdaq"},
        {"proName": "CAPITALCOM:US500", "title": "S&P 500"},
        {"proName": "CAPITALCOM:US30", "title": "Dow Jones"},
        {"proName": "TVC:VIX", "title": "VIX"}
    ])

    symbols_json1 = json.dumps(symbols_tv1)
    symbols_json2 = json.dumps(symbols_tv2)

    tradingview_widget1 = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="margin-bottom: 0px;">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {{
      "symbols": {symbols_json1},
      "showSymbolLogo": true,
      "colorTheme": "{color_theme}",
      "isTransparent": false,
      "displayMode": "adaptive",
      "locale": "fr"
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """

    tradingview_widget2 = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="margin-top: 0px; margin-bottom: 20px;">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {{
      "symbols": {symbols_json2},
      "showSymbolLogo": true,
      "colorTheme": "{color_theme}",
      "isTransparent": false,
      "displayMode": "adaptive",
      "locale": "fr"
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """

    st.components.v1.html(tradingview_widget1, height=80)
    st.components.v1.html(tradingview_widget2, height=80)
else:
    st.info("Sélectionnez des tickers pour afficher le bandeau TradingView")

st.markdown('<hr style="border-color: #333; margin: 15px 0;">', unsafe_allow_html=True)


# =============================================
# BARRE DE COMMANDE BLOOMBERG
# À ajouter après le header, avant les données de marché
# =============================================

# Style pour la barre de commande
st.markdown("""
<style>
    .command-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 10px 15px;
        margin: 10px 0 20px 0;
    }
    .command-prompt {
        color: #FFAA00;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Dictionnaire des commandes et leurs pages
COMMANDS = {
    "EDGAR": "pages/EDGAR.py",
    "NEWS": "pages/NEWS.py",
    "PRICE": "pages/PRICING.py",
    "CHAT": "pages/CHATBOT.py",
    "BT": "pages/BACKTESTING.py",
    "ANA": "pages/COMPANY_ANALYSIS.py",
    "CRYPTO":"pages/CRYPTO_SCRAPER.py",
    "ECO":"pages/ECONOMICS.py", 
    "EU":"pages/EUROPE.py",
    "SIMU":"pages/PORTFOLIO_SIMU.py",
    "PY":"pages/PYTHON_EDITOR.py",
    "SQL":"pages/SQL_EDITOR.py",
    "BONDS":"pages/BONDS.py",
    "HOME":"pages/HOME.py",
}

# Affichage de la barre de commande
st.markdown('<div class="command-container">', unsafe_allow_html=True)

col_prompt, col_input = st.columns([1, 11])

with col_prompt:
    st.markdown('<span class="command-prompt">BBG&gt;</span>', unsafe_allow_html=True)

with col_input:
    command_input = st.text_input(
        "Command",
        placeholder="Tapez une commande: EDGAR, NEWS, CHATBOT, PRICING, HELP...",
        label_visibility="collapsed",
        key="bloomberg_command"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Traitement de la commande
if command_input:
    cmd = command_input.upper().strip()
    
    if cmd == "HELP" or cmd == "H":
        st.info("""
        **📋 COMMANDES DISPONIBLES:**
        - `EDGAR` → SEC Filings & Documents
        - `NEWS` → Market News Feed
        - `CHAT` → AI Assistant
        - `PRICE` → Options Pricing
        - `HELP` → Afficher cette aide
        - `BT` → Backesting de strategies
        - `ANA` → Analyse financière de sociétés côtées
        - `CRYPTO` → Scrapping et backtest de strategies liées aux cryptos
        - `ECO` → Données économiques
        - `EU` → Données Européennes
        - `SIMU` → Simulation de portefeuille
        - `PY` → Editeur de code python 
        - `SQL` → Editeur de code SQL
        - `BONDS` → Screener d'obligation
        - `HOME` → Menu
        """)
    elif cmd in COMMANDS:
        st.switch_page(COMMANDS[cmd])
    else:
        st.warning(f"⚠️ Commande '{cmd}' non reconnue. Tapez HELP pour voir les commandes disponibles.")

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

# ===== PUB HEADER =====
add_header_ad()

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
# WIDGET TRADINGVIEW - FOREX CROSS RATES
# =============================================
st.markdown("### 💹 FOREX CROSS RATES - TRADINGVIEW")

# Interface de configuration (discrète)
with st.expander("⚙️ CONFIGURE FOREX WIDGET", expanded=False):
    col_fx_widget1, col_fx_widget2, col_fx_widget3 = st.columns(3)
    
    with col_fx_widget1:
        forex_theme = st.selectbox(
            "Thème",
            options=["dark", "light"],
            index=0,
            key="forex_widget_theme"
        )
    
    with col_fx_widget2:
        forex_height = st.slider(
            "Hauteur (px)",
            min_value=300,
            max_value=800,
            value=500,
            step=50,
            key="forex_widget_height"
        )
    
    with col_fx_widget3:
        show_symbol_logo = st.checkbox(
            "Afficher les logos",
            value=True,
            key="forex_show_logo"
        )

# Widget TradingView Forex Cross Rates
forex_widget = f"""
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container" style="margin: 20px 0;">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-forex-cross-rates.js" async>
  {{
  "width": "100%",
  "height": {forex_height},
  "currencies": [
    "EUR",
    "USD",
    "JPY",
    "GBP",
    "CHF",
    "AUD",
    "CAD",
    "NZD",
    "CNY"
  ],
  "isTransparent": false,
  "colorTheme": "{forex_theme}",
  "locale": "fr",
  "backgroundColor": "#000000"
  }}
  </script>
</div>
<!-- TradingView Widget END -->
"""

# Afficher le widget
st.components.v1.html(forex_widget, height=forex_height + 20)

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
# GRAPHIQUE COMPARATIF + MATRICE CORRÉLATION
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("### 📈 COMPARATIVE CHART - PERFORMANCE %")

col_chart1, col_chart2 = st.columns([3, 1])

with col_chart1:
    popular_tickers = ['A', 'AA', 'AAAU', 'AACB', 'AACBR', 'AACBU', 'AACG', 'AAL', 'AAM', 'AAM-UN','AAM-WT', 'AAME', 'AAMI', 'AAMTF', 'AAOI', 'AAON', 'AAP', 'AAPG', 'AAPI', 'AAPL',
    'AAQL', 'AARD', 'AAS', 'AASP', 'AAT', 'AATC', 'AAUAF', 'AAUC', 'AAUGF', 'AAVXF',
    'AAWH', 'AB', 'ABAKF', 'ABAT', 'ABBNY', 'ABBV', 'ABCB', 'ABCFF', 'ABCL', 'ABCP',
    'ABEO', 'ABEV', 'ABG', 'ABIT', 'ABL', 'ABLLL', 'ABLV', 'ABLVW', 'ABLZF', 'ABM',
    'ABNB', 'ABOS', 'ABP', 'ABPWW', 'ABQQ', 'ABR', 'ABR-PD', 'ABR-PE', 'ABR-PF', 'ABSI',
    'ABT', 'ABTC', 'ABTS', 'ABUS', 'ABVC', 'ABVE', 'ABVEW', 'ABVX', 'ABXXF', 'ACA',
    'ACAD', 'ACATD', 'ACB', 'ACBM', 'ACCL', 'ACCO', 'ACCS', 'ACDC', 'ACEL', 'ACET',
    'ACFN', 'ACGL', 'ACGLN', 'ACGLO', 'ACGP', 'ACHC', 'ACHR', 'ACHR-WT', 'ACHV', 'ACI',
    'ACIC', 'ACIU', 'ACIW', 'ACLS', 'ACLX', 'ACM', 'ACMIF', 'ACMR', 'ACN', 'ACNB',
    'ACNT', 'ACOG', 'ACON', 'ACONW', 'ACP', 'ACP-PA', 'ACPS', 'ACR', 'ACR-PC', 'ACR-PD',
    'ACRE', 'ACRG', 'ACRL', 'ACRS', 'ACRV', 'ACT', 'ACTG', 'ACTU', 'ACU', 'ACUT',
    'ACV', 'ACVA', 'ACXP', 'AD', 'ADAG', 'ADAM', 'ADAMG', 'ADAMH', 'ADAMI', 'ADAML',
    'ADAMM', 'ADAMN', 'ADAMZ', 'ADAPY', 'ADBE', 'ADC', 'ADC-PA', 'ADCT', 'ADEA', 'ADGM',
    'ADI', 'ADIA', 'ADIL', 'ADM', 'ADMA', 'ADMG', 'ADMQ', 'ADMT', 'ADN', 'ADNH',
    'ADNT', 'ADNWW', 'ADOOY', 'ADP', 'ADPT', 'ADSE', 'ADSEW', 'ADSK', 'ADT', 'ADTI',
    'ADTN', 'ADTTF', 'ADTX', 'ADUR', 'ADUS', 'ADV', 'ADVB', 'ADVM', 'ADX', 'ADXN',
    'ADYYF', 'ADZCF', 'AEAE', 'AEAEU', 'AEAEW', 'AEBI', 'AEBMF', 'AEBMY', 'AEBZY', 'AEC',
    'AEE', 'AEF', 'AEFC', 'AEG', 'AEGOF', 'AEGXF', 'AEHGS', 'AEHL', 'AEHR', 'AEI',
    'AEIS', 'AEM', 'AEMD', 'AENT', 'AENTW', 'AEO', 'AEON', 'AEP', 'AER', 'AERG',
    'AERGP', 'AERO', 'AERT', 'AERTW', 'AES', 'AESI', 'AEVA', 'AEVAW', 'AEXA', 'AEYE',
    'AFA', 'AFB', 'AFBI', 'AFCG', 'AFG', 'AFGB', 'AFGC', 'AFGD', 'AFGE', 'AFJK',
    'AFJKR', 'AFJKU', 'AFL', 'AFRI', 'AFRIW', 'AFRM', 'AFYA', 'AG', 'AGAE', 'AGCC',
    'AGCO', 'AGD', 'AGEN', 'AGGI', 'AGH', 'AGI', 'AGIO', 'AGL', 'AGLY', 'AGM',
    'AGM-A', 'AGM-PD', 'AGM-PE', 'AGM-PF', 'AGM-PG', 'AGM-PH', 'AGMH', 'AGMRF', 'AGMWF', 'AGNC',
    'AGNCL', 'AGNCM', 'AGNCN', 'AGNCO', 'AGNCP', 'AGNCZ', 'AGNPF', 'AGO', 'AGQ', 'AGQPF',
    'AGRO', 'AGRZ', 'AGSS', 'AGTX', 'AGX', 'AGXPF', 'AGYS', 'AHCO', 'AHG', 'AHH',
    'AHH-PA', 'AHII', 'AHL', 'AHL-PD', 'AHL-PE', 'AHL-PF', 'AHMA', 'AHNRF', 'AHR', 'AHRO',
    'AHT', 'AHT-PD', 'AHT-PF', 'AHT-PG', 'AHT-PH', 'AHT-PI', 'AI', 'AIBRF', 'AIBT', 'AIDG',
    'AIEV', 'AIFF', 'AIFU', 'AIG', 'AIGO', 'AIHS', 'AII', 'AIIA', 'AIIA-RI', 'AIIA-UN',
    'AIIO', 'AIIOW', 'AIJTY', 'AIKCF', 'AILIH', 'AILIM', 'AILIN', 'AILIO', 'AILIP', 'AILLI',
    'AILLM', 'AILLN', 'AILLO', 'AILLP', 'AIM', 'AIMD', 'AIMDW', 'AIMI', 'AIN', 'AIO',
    'AIOT', 'AIP', 'AIPG', 'AIQUF', 'AIR', 'AIRE', 'AIRG', 'AIRI', 'AIRJ', 'AIRJW',
    'AIRO', 'AIRRF', 'AIRS', 'AIRT', 'AIRTP', 'AISP', 'AISPW', 'AIT', 'AITTF', 'AITX',
    'AIV', 'AIXC', 'AIXI', 'AIXN', 'AIZ', 'AIZN', 'AJG', 'AKA', 'AKAM', 'AKAN',
    'AKBA', 'AKCLY', 'AKO-A', 'AKO-B', 'AKPPS', 'AKR', 'AKRO', 'AKTX', 'AL', 'ALAB',
    'ALAR', 'ALB', 'ALB-PA', 'ALBT', 'ALC', 'ALCE', 'ALCO', 'ALCY', 'ALCYU', 'ALCYW',
    'ALD', 'ALDA', 'ALDF', 'ALDFU', 'ALDFW', 'ALDS', 'ALDX', 'ALE', 'ALEC', 'ALEGF',
    'ALEH', 'ALEUY', 'ALEX', 'ALF', 'ALFUU', 'ALFUW', 'ALG', 'ALGM', 'ALGN', 'ALGS',
    'ALGT', 'ALH', 'ALHC', 'ALIS', 'ALISR', 'ALISU', 'ALIT', 'ALK', 'ALKS', 'ALKT',
    'ALL', 'ALL-PB', 'ALL-PH', 'ALL-PI', 'ALL-PJ', 'ALLE', 'ALLO', 'ALLR', 'ALLT', 'ALLY',
    'ALM', 'ALMMF', 'ALMP', 'ALMS', 'ALMU', 'ALNT', 'ALNY', 'ALOIS', 'ALOT', 'ALPS',
    'ALPWF', 'ALRM', 'ALRS', 'ALRTF', 'ALSAF', 'ALSN', 'ALSTF', 'ALSUF', 'ALSWF', 'ALT',
    'ALTB', 'ALTG', 'ALTG-PA', 'ALTI', 'ALTO', 'ALTS', 'ALTX', 'ALUB', 'ALUB-UN', 'ALUR',
    'ALUR-WT', 'ALV', 'ALVO', 'ALVOW', 'ALX', 'ALXO', 'ALXY', 'ALXYD', 'ALYAF', 'ALZN',
    'AM', 'AMAL', 'AMAT', 'AMBA', 'AMBI', 'AMBIQ', 'AMBO', 'AMBP', 'AMBP-WT', 'AMBQ',
    'AMBR', 'AMBWQ', 'AMC', 'AMCCF', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMG', 'AMGN',
    'AMH', 'AMH-PG', 'AMH-PH', 'AMIX', 'AMJB', 'AMKR', 'AMLIF', 'AMLX', 'AMN', 'AMOD',
    'AMODW', 'AMP', 'AMPG', 'AMPGW', 'AMPH', 'AMPL', 'AMPM', 'AMPX', 'AMPX-WT', 'AMPY',
    'AMR', 'AMRC', 'AMRK', 'AMRN', 'AMRQF', 'AMRX', 'AMRZ', 'AMS', 'AMSC', 'AMSF',
    'AMST', 'AMSYF', 'AMT', 'AMTB', 'AMTD', 'AMTM', 'AMTU', 'AMTX', 'AMUB', 'AMVIF',
    'AMVOY', 'AMWD', 'AMWL', 'AMX', 'AMXOF', 'AMZE', 'AMZN', 'AN', 'ANAB', 'ANDE',
    'ANDG', 'ANEB', 'ANET', 'ANF', 'ANG-PD', 'ANGH', 'ANGHW', 'ANGI', 'ANGO', 'ANGX',
    'ANIK', 'ANIP', 'ANIX', 'ANKM', 'ANL', 'ANNA', 'ANNAW', 'ANNSF', 'ANNX', 'ANPA',
    'ANPCF', 'ANPCY', 'ANRO', 'ANSC', 'ANSCU', 'ANSCW', 'ANTA', 'ANTX', 'ANVI', 'ANVS',
    'ANY', 'AOAO', 'AOD', 'AOMD', 'AOMFF', 'AOMN', 'AOMR', 'AON', 'AONC', 'AONCW',
    'AORT', 'AOS', 'AOSL', 'AOUT', 'AOXY', 'AP', 'APA', 'APAAF', 'APAC', 'APACR',
    'APACU', 'APAD', 'APADR', 'APADU', 'APAM', 'APCX', 'APCXW', 'APD', 'APEI', 'APEX',
    'APG', 'APGE', 'APGOF', 'APH', 'APHD', 'APHP', 'API', 'APLD', 'APLE', 'APLM',
    'APLMW', 'APLS', 'APLT', 'APM', 'APO', 'APO-PA', 'APOG', 'APOS', 'APP', 'APPF',
    'APPN', 'APPS', 'APRE', 'APSI', 'APT', 'APTOF', 'APTV', 'APUR', 'APUS', 'APVO',
    'APWC', 'APXC', 'APXCF', 'APXIF', 'APXT', 'APXTU', 'APXTW', 'APYP', 'APYX', 'AQB',
    'AQLS', 'AQMS', 'AQN', 'AQNB', 'AQST', 'AQUC', 'AR', 'ARAI', 'ARAY', 'ARBB',
    'ARBE', 'ARBEW', 'ARBK', 'ARBKF', 'ARBKL', 'ARCB', 'ARCC', 'ARCO', 'ARCT', 'ARCXF',
    'ARDC', 'ARDT', 'ARDX', 'ARE', 'AREB', 'AREBW', 'AREC', 'AREN', 'ARES', 'ARES-PB',
    'ARGX', 'ARHS', 'ARHUF', 'ARI', 'ARKB', 'ARKO', 'ARKOW', 'ARKR', 'ARL', 'ARLO',
    'ARLP', 'ARM', 'ARMK', 'ARMN', 'ARMP', 'ARNNY', 'AROC', 'AROW', 'ARQ', 'ARQQ',
    'ARQQW', 'ARQT', 'ARR', 'ARR-PC', 'ARRKF', 'ARRT', 'ARRY', 'ARSMF', 'ARSTY', 'ARTL',
    'ARTNA', 'ARTNB', 'ARTV', 'ARTW', 'ARVN', 'ARW', 'ARWR', 'ARX', 'ARZTF', 'AS',
    'ASA', 'ASAIY', 'ASAN', 'ASB', 'ASB-PE', 'ASB-PF', 'ASBA', 'ASBHY', 'ASBP', 'ASBPW',
    'ASBRF', 'ASC', 'ASCBF', 'ASCIX', 'ASCRF', 'ASCWF', 'ASFH', 'ASFT', 'ASG', 'ASGI',
    'ASGN', 'ASH', 'ASIC', 'ASII', 'ASIX', 'ASKRY', 'ASLE', 'ASM', 'ASMB', 'ASML',
    'ASMLF', 'ASND', 'ASNS', 'ASO', 'ASPC', 'ASPCR', 'ASPCU', 'ASPI', 'ASPN', 'ASPS',
    'ASPSW', 'ASPSZ', 'ASPU', 'ASR', 'ASRE', 'ASRMF', 'ASRRF', 'ASRT', 'ASRV', 'ASST',
    'ASTC', 'ASTE', 'ASTH', 'ASTI', 'ASTL', 'ASTLW', 'ASTS', 'ASUR', 'ASUUF', 'ASX',
    'ASYS', 'ATAI', 'ATAT', 'ATCH', 'ATCHW', 'ATCMF', 'ATCX', 'ATDS', 'ATEC', 'ATEK',
    'ATEN', 'ATER', 'ATEX', 'ATEYY', 'ATGAF', 'ATGE', 'ATGFF', 'ATGL', 'ATH-PA', 'ATH-PB',
    'ATH-PD', 'ATH-PE', 'ATHA', 'ATHE', 'ATHM', 'ATHNF', 'ATHNY', 'ATHR', 'ATHS', 'ATI',
    'ATII', 'ATIIU', 'ATIIW', 'ATKR', 'ATLC', 'ATLCL', 'ATLCP', 'ATLCZ', 'ATLN', 'ATLO',
    'ATLX', 'ATMC', 'ATMCR', 'ATMCU', 'ATMCW', 'ATMH', 'ATMP', 'ATMU', 'ATMV', 'ATMVR',
    'ATMVU', 'ATNI', 'ATNM', 'ATO', 'ATOM', 'ATON', 'ATOS', 'ATPC', 'ATR', 'ATRA',
    'ATRC', 'ATRO', 'ATROB', 'ATS', 'ATVK', 'ATXG', 'ATXI', 'ATXS', 'ATYR', 'AU',
    'AUB', 'AUB-PA', 'AUBN', 'AUDC', 'AUGG', 'AUGO', 'AUIAF', 'AUID', 'AUIWF', 'AUMN',
    'AUNA', 'AUPH', 'AUR', 'AURA', 'AURE', 'AUROW', 'AURX', 'AUSI', 'AUST', 'AUTL',
    'AUUD', 'AUUDW', 'AUXXF', 'AVA', 'AVAH', 'AVAI', 'AVAL', 'AVAV', 'AVAX', 'AVB',
    'AVBC', 'AVBH', 'AVBP', 'AVCRF', 'AVD', 'AVDL', 'AVG', 'AVGO', 'AVHHL', 'AVIR',
    'AVK', 'AVLNF', 'AVNI', 'AVNS', 'AVNT', 'AVNW', 'AVO', 'AVPKS', 'AVPMF', 'AVPT',
    'AVR', 'AVSBS', 'AVT', 'AVTR', 'AVTX', 'AVVOF', 'AVVSY', 'AVX', 'AVXL', 'AVY',
    'AWATY', 'AWCA', 'AWF', 'AWHL', 'AWI', 'AWK', 'AWP', 'AWR', 'AWRE', 'AWTRF',
    'AWX', 'AX', 'AXG', 'AXGN', 'AXIA', 'AXIA-P', 'AXIL', 'AXIM', 'AXIN', 'AXINR',
    'AXINU', 'AXL', 'AXON', 'AXP', 'AXR', 'AXREF', 'AXS', 'AXS-PE', 'AXSM', 'AXTA',
    'AXTI', 'AYI', 'AYR', 'AYRWF', 'AYTU', 'AYWWF', 'AZ', 'AZASF', 'AZI', 'AZN',
    'AZNCF', 'AZO', 'AZTA', 'AZTR', 'AZULQ', 'AZZ', 'AZZTF', 'B', 'BA', 'BA-PA',
    'BABA', 'BABAF', 'BABB', 'BAC', 'BAC-PB', 'BAC-PE', 'BAC-PK', 'BAC-PL', 'BAC-PM', 'BAC-PN',
    'BAC-PO', 'BAC-PP', 'BAC-PQ', 'BAC-PS', 'BACC', 'BACCR', 'BACCU', 'BACK', 'BACQ', 'BACQR',
    'BACQU', 'BACRP', 'BAER', 'BAERW', 'BAESF', 'BAESY', 'BAFBF', 'BAFN', 'BAH', 'BAIDF',
    'BAK', 'BAKR', 'BALL', 'BALY', 'BAM', 'BAMGF', 'BAMKF', 'BAMXF', 'BANC', 'BANC-PF',
    'BAND', 'BANF', 'BANFP', 'BANL', 'BANR', 'BANX', 'BAO', 'BAOS', 'BAP', 'BAR',
    'BARK', 'BARK-WT', 'BASA', 'BATL', 'BATRA', 'BATRB', 'BATRK', 'BAX', 'BAYA', 'BAYAR',
    'BAYAU', 'BB', 'BBAAY', 'BBAI', 'BBAI-WT', 'BBAR', 'BBBMF', 'BBBXF', 'BBBY', 'BBBY-WT',
    'BBCP', 'BBD', 'BBDC', 'BBDO', 'BBGI', 'BBIO', 'BBLG', 'BBLGW', 'BBLR', 'BBN',
    'BBNX', 'BBOT', 'BBSI', 'BBT', 'BBU', 'BBUC', 'BBVA', 'BBVXF', 'BBW', 'BBWI',
    'BBXIA', 'BBXIB', 'BBY', 'BC', 'BC-PA', 'BC-PC', 'BCAB', 'BCAEF', 'BCAL', 'BCAR',
    'BCARU', 'BCARW', 'BCAT', 'BCAX', 'BCBP', 'BCC', 'BCDA', 'BCDRF', 'BCE', 'BCEFF',
    'BCENF', 'BCEPF', 'BCEXF', 'BCG', 'BCGWW', 'BCH', 'BCHG', 'BCHT', 'BCIC', 'BCKIF',
    'BCLI', 'BCLYF', 'BCML', 'BCO', 'BCOW', 'BCPC', 'BCPPF', 'BCRD', 'BCRX', 'BCS',
    'BCSF', 'BCSS', 'BCSS-UN', 'BCSS-WT', 'BCTF', 'BCTX', 'BCTXW', 'BCTXZ', 'BCUCF', 'BCUFF',
    'BCV', 'BCV-PA', 'BCX', 'BCYC', 'BDC', 'BDCC', 'BDCI', 'BDCIU', 'BDCIW', 'BDCO',
    'BDCTF', 'BDCX', 'BDCZ', 'BDJ', 'BDL', 'BDMD', 'BDMDW', 'BDN', 'BDPT', 'BDRX',
    'BDRY', 'BDSX', 'BDTB', 'BDTX', 'BDX', 'BE', 'BEAG', 'BEAGR', 'BEAGU', 'BEAM',
    'BEAT', 'BEATW', 'BEBE', 'BECEF', 'BEDU', 'BEEM', 'BEEP', 'BEIGF', 'BEKE', 'BELFA',
    'BELFB', 'BEN', 'BENF', 'BENFW', 'BENN', 'BEOB', 'BEP', 'BEP-PA', 'BEPC', 'BEPH',
    'BEPI', 'BEPJ', 'BERZ', 'BESS', 'BETA', 'BETR', 'BETRF', 'BETRW', 'BEWFF', 'BF-A',
    'BF-B', 'BFAM', 'BFC', 'BFGFF', 'BFH', 'BFH-PA', 'BFHIP', 'BFIN', 'BFK', 'BFLY',
    'BFLY-WT', 'BFNH', 'BFRG', 'BFRGW', 'BFRI', 'BFRIW', 'BFS', 'BFS-PD', 'BFS-PE', 'BFST',
    'BFZ', 'BG', 'BGAVF', 'BGB', 'BGC', 'BGFR', 'BGH', 'BGHL', 'BGI', 'BGIN',
    'BGL', 'BGLC', 'BGLWW', 'BGM', 'BGMS', 'BGMSP', 'BGR', 'BGS', 'BGSF', 'BGSI',
    'BGT', 'BGX', 'BGY', 'BH', 'BH-A', 'BHAT', 'BHB', 'BHC', 'BHE', 'BHF',
    'BHFAL', 'BHFAM', 'BHFAN', 'BHFAO', 'BHFAP', 'BHIC', 'BHK', 'BHLL', 'BHM', 'BHP',
    'BHPLF', 'BHR', 'BHR-PB', 'BHR-PD', 'BHRB', 'BHSIF', 'BHST', 'BHV', 'BHVN', 'BIAF',
    'BIAFW', 'BICX', 'BIDU', 'BIIB', 'BILI', 'BILL', 'BINI', 'BIO', 'BIO-B', 'BIOA',
    'BIOE', 'BIOF', 'BIOT', 'BIOX', 'BIP', 'BIP-PA', 'BIP-PB', 'BIPC', 'BIPH', 'BIPI',
    'BIPJ', 'BIRD', 'BIREF', 'BIREY', 'BIRK', 'BIT', 'BITB', 'BITF', 'BITTF', 'BITW',
    'BIVI', 'BIVIW', 'BIXI', 'BIXT', 'BIYA', 'BJ', 'BJDX', 'BJRI', 'BK', 'BK-PK',
    'BKAMF', 'BKD', 'BKE', 'BKFAF', 'BKFDF', 'BKFOF', 'BKFPF', 'BKFSF', 'BKH', 'BKHA',
    'BKHAR', 'BKHAU', 'BKKT', 'BKKT-WT', 'BKLPF', 'BKN', 'BKNG', 'BKPA', 'BKPKF', 'BKPOY',
    'BKR', 'BKRRF', 'BKSC', 'BKSY', 'BKSY-WT', 'BKT', 'BKTH', 'BKTI', 'BKU', 'BKUCF',
    'BKV', 'BKYI', 'BL', 'BLBD', 'BLBLF', 'BLBX', 'BLCO', 'BLD', 'BLDP', 'BLDR',
    'BLE', 'BLFBY', 'BLFS', 'BLFY', 'BLGO', 'BLIN', 'BLIS', 'BLIV', 'BLK', 'BLKB',
    'BLLN', 'BLMH', 'BLMN', 'BLMZ', 'BLNC', 'BLND', 'BLNE', 'BLNK', 'BLPG', 'BLRX',
    'BLSH', 'BLTE', 'BLTH', 'BLUW', 'BLUWU', 'BLUWW', 'BLW', 'BLX', 'BLZE', 'BLZR',
    'BLZRU', 'BLZRW', 'BMA', 'BMBL', 'BMBN', 'BME', 'BMEA', 'BMEZ', 'BMGL', 'BMHL',
    'BMI', 'BML-PG', 'BML-PH', 'BML-PJ', 'BML-PL', 'BMN', 'BMNM', 'BMNR', 'BMO', 'BMOK',
    'BMOOF', 'BMPA', 'BMR', 'BMRA', 'BMRC', 'BMRN', 'BMTM', 'BMWKY', 'BMXI', 'BMY',
    'BMYMP', 'BN', 'BNAI', 'BNAIW', 'BNBX', 'BNC', 'BNCWW', 'BNED', 'BNET', 'BNGO',
    'BNH', 'BNIGF', 'BNJ', 'BNKD', 'BNKK', 'BNKU', 'BNL', 'BNO', 'BNR', 'BNRG',
    'BNS', 'BNSOF', 'BNT', 'BNTC', 'BNTX', 'BNY', 'BNZI', 'BNZIW', 'BOC', 'BODI',
    'BODYW', 'BOE', 'BOF', 'BOH', 'BOH-PA', 'BOH-PB', 'BOIL', 'BOKF', 'BOLD', 'BOLT',
    'BON', 'BONXF', 'BOOM', 'BOOT', 'BORMF', 'BORR', 'BOSC', 'BOTJ', 'BOTY', 'BOW',
    'BOWN', 'BOX', 'BOXL', 'BP', 'BPAC', 'BPACU', 'BPAQF', 'BPOP', 'BPOPM', 'BPOPO',
    'BPPFF', 'BPPTU', 'BPRN', 'BPTH', 'BPYPM', 'BPYPN', 'BPYPO', 'BPYPP', 'BQ', 'BQST',
    'BR', 'BRAG', 'BRAI', 'BRBF', 'BRBI', 'BRBR', 'BRBS', 'BRC', 'BRCB', 'BRCC',
    'BRCFF', 'BRCNF', 'BRELY', 'BRENF', 'BRETF', 'BREZ', 'BRFCF', 'BRFH', 'BRGC', 'BRGX',
    'BRIA', 'BRID', 'BRIPF', 'BRK-A', 'BRK-B', 'BRKCF', 'BRKK', 'BRKR', 'BRKRP', 'BRLS',
    'BRLSW', 'BRLT', 'BRN', 'BRNS', 'BRO', 'BROS', 'BROXF', 'BRPSF', 'BRQL', 'BRQSF',
    'BRR', 'BRRN', 'BRRR', 'BRRWU', 'BRRWW', 'BRSL', 'BRSP', 'BRST', 'BRSYF', 'BRT',
    'BRTX', 'BRVO', 'BRW', 'BRWC', 'BRX', 'BRY', 'BRZE', 'BSAA', 'BSAAR', 'BSAAU',
    'BSAC', 'BSAI', 'BSBK', 'BSBR', 'BSEM', 'BSET', 'BSFC', 'BSL', 'BSLK', 'BSLKW',
    'BSM', 'BSMLP', 'BSOL', 'BSPK', 'BSQKZ', 'BSRR', 'BST', 'BSTT', 'BSTZ', 'BSVN',
    'BSX', 'BSY', 'BTA', 'BTAB', 'BTAFF', 'BTAI', 'BTBD', 'BTBDW', 'BTBT', 'BTC',
    'BTCO', 'BTCS', 'BTCT', 'BTCW', 'BTCY', 'BTDPF', 'BTDPY', 'BTDR', 'BTE', 'BTFT',
    'BTG', 'BTGO', 'BTGRF', 'BTI', 'BTLWF', 'BTM', 'BTMD', 'BTMWW', 'BTO', 'BTOC',
    'BTOG', 'BTQ', 'BTSG', 'BTSGU', 'BTT', 'BTTC', 'BTU', 'BTX', 'BTZ', 'BUD',
    'BUDA', 'BUDFF', 'BUDZ', 'BUGDF', 'BUHHF', 'BUHPF', 'BUHPY', 'BUI', 'BUKS', 'BULL',
    'BULLW', 'BULZ', 'BUR', 'BURL', 'BURU', 'BURUW', 'BUSE', 'BUSEP', 'BUUU', 'BV',
    'BVAXF', 'BVFL', 'BVN', 'BVS', 'BW', 'BW-PA', 'BWA', 'BWAY', 'BWB', 'BWBBP',
    'BWEN', 'BWET', 'BWFG', 'BWG', 'BWIN', 'BWLP', 'BWMG', 'BWMN', 'BWMX', 'BWNB',
    'BWOW', 'BWSN', 'BWVTF', 'BWXT', 'BX', 'BXC', 'BXDIF', 'BXMT', 'BXMX', 'BXP',
    'BXRLY', 'BXSL', 'BXSY', 'BY', 'BYAH', 'BYCBF', 'BYD', 'BYDDF', 'BYDIF', 'BYFC',
    'BYM', 'BYMOF', 'BYND', 'BYNO', 'BYNOU', 'BYNOW', 'BYOC', 'BYRN', 'BYSI', 'BZ',
    'BZAI', 'BZAIW', 'BZFD', 'BZFDW', 'BZH', 'BZLFF', 'BZLFY', 'BZRD', 'BZUN', 'BZYR',
    'C', 'C-PN', 'CAAP', 'CAAS', 'CABA', 'CABO', 'CABR', 'CAC', 'CACC', 'CACI',
    'CADE', 'CADE-PA', 'CADL', 'CADV', 'CAE', 'CAEA', 'CAEP', 'CAF', 'CAG', 'CAH',
    'CAHO', 'CAHPF', 'CAI', 'CAJFF', 'CAJPY', 'CAKE', 'CAL', 'CALC', 'CALM', 'CALX',
    'CAMG', 'CAMP', 'CAMT', 'CAN', 'CANE', 'CANF', 'CANG', 'CANN', 'CAPC', 'CAPL',
    'CAPN', 'CAPNR', 'CAPNU', 'CAPR', 'CAPS', 'CAPT', 'CAPTW', 'CAR', 'CARCY', 'CARD',
    'CARE', 'CARG', 'CARL', 'CARM', 'CARR', 'CARS', 'CART', 'CARU', 'CARV', 'CASH',
    'CASI', 'CASS', 'CAST', 'CASY', 'CAT', 'CATO', 'CATX', 'CATY', 'CAVA', 'CAZGF',
    'CB', 'CBAN', 'CBAT', 'CBC', 'CBCY', 'CBDBY', 'CBDL', 'CBDW', 'CBDY', 'CBFV',
    'CBGGF', 'CBIH', 'CBIO', 'CBK', 'CBKM', 'CBL', 'CBLL', 'CBLO', 'CBMJ', 'CBNA',
    'CBNK', 'CBOE', 'CBPHF', 'CBRA', 'CBRE', 'CBRGF', 'CBRL', 'CBRRF', 'CBSH', 'CBSTF',
    'CBT', 'CBU', 'CBUS', 'CBZ', 'CC', 'CCAP', 'CCB', 'CCBG', 'CCC', 'CCCC',
    'CCCP', 'CCCX', 'CCCXU', 'CCCXW', 'CCD', 'CCEC', 'CCEL', 'CCEP', 'CCFN', 'CCG',
    'CCGWW', 'CCHH', 'CCI', 'CCID', 'CCIF', 'CCII', 'CCIIU', 'CCIIW', 'CCIX', 'CCIXU',
    'CCIXW', 'CCJ', 'CCK', 'CCL', 'CCLD', 'CCLDO', 'CCM', 'CCNB', 'CCNE', 'CCNEP',
    'CCO', 'CCOI', 'CCOOF', 'CCRN', 'CCS', 'CCSI', 'CCTC', 'CCTG', 'CCTSF', 'CCU',
    'CCXI', 'CCZ', 'CD', 'CDAQF', 'CDAUF', 'CDAWF', 'CDE', 'CDGLF', 'CDIO', 'CDIOW',
    'CDIX', 'CDLR', 'CDLX', 'CDNA', 'CDNS', 'CDP', 'CDR-PB', 'CDR-PC', 'CDRE', 'CDRO',
    'CDROW', 'CDSG', 'CDT', 'CDTG', 'CDTHY', 'CDTTW', 'CDTX', 'CDW', 'CDXS', 'CDZI',
    'CDZIP', 'CE', 'CECO', 'CEE', 'CEF', 'CEFD', 'CEFZ', 'CEG', 'CEHCF', 'CEIN',
    'CELC', 'CELG-RI', 'CELH', 'CELU', 'CELUW', 'CELZ', 'CENN', 'CENT', 'CENTA', 'CENX',
    'CEP', 'CEPF', 'CEPO', 'CEPT', 'CEPU', 'CEPV', 'CERO', 'CEROW', 'CERS', 'CERT',
    'CET', 'CETI', 'CETX', 'CETXP', 'CETY', 'CEV', 'CEVA', 'CF', 'CFAC', 'CFBK',
    'CFFI', 'CFFN', 'CFG', 'CFG-PE', 'CFG-PH', 'CFG-PI', 'CFLT', 'CFNB', 'CFND', 'CFOO',
    'CFR', 'CFR-PB', 'CFSB', 'CFSU', 'CG', 'CGABL', 'CGAU', 'CGBD', 'CGBDL', 'CGBSF',
    'CGC', 'CGCT', 'CGCTU', 'CGCTW', 'CGDXF', 'CGEH', 'CGEM', 'CGEN', 'CGL', 'CGNT',
    'CGNX', 'CGO', 'CGON', 'CGTL', 'CGTX', 'CHA', 'CHAC', 'CHACR', 'CHACU', 'CHAI',
    'CHAR', 'CHARR', 'CHARU', 'CHCI', 'CHCO', 'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEAF',
    'CHEC', 'CHECU', 'CHECW', 'CHEF', 'CHEK', 'CHEV', 'CHGG', 'CHH', 'CHI', 'CHKIF',
    'CHKKF', 'CHKP', 'CHKR', 'CHLSY', 'CHMG', 'CHMI', 'CHMI-PA', 'CHMI-PB', 'CHMX', 'CHNEY',
    'CHNR', 'CHOW', 'CHPG', 'CHPGR', 'CHPGU', 'CHPT', 'CHR', 'CHRD', 'CHRS', 'CHRW',
    'CHSCL', 'CHSCM', 'CHSCN', 'CHSCO', 'CHSCP', 'CHSN', 'CHT', 'CHTH', 'CHTR', 'CHUC',
    'CHW', 'CHWRF', 'CHWY', 'CHY', 'CHYM', 'CI', 'CIA', 'CIB', 'CICB', 'CIEN',
    'CIF', 'CIFR', 'CIFRW', 'CIG', 'CIG-C', 'CIGI', 'CIGL', 'CII', 'CIIT', 'CIK',
    'CILJF', 'CIM', 'CIM-PA', 'CIM-PB', 'CIM-PC', 'CIM-PD', 'CIMN', 'CIMO', 'CIMP', 'CINF',
    'CING', 'CINGW', 'CINT', 'CIO', 'CIO-PA', 'CION', 'CIRX', 'CISO', 'CISS', 'CITAF',
    'CIVB', 'CIVI', 'CIVII', 'CIX', 'CIXPF', 'CIZN', 'CJAX', 'CJET', 'CJMB', 'CJRCF',
    'CKDXF', 'CKHGF', 'CKX', 'CL', 'CLAR', 'CLB', 'CLBK', 'CLBT', 'CLBZ', 'CLCO',
    'CLCS', 'CLDI', 'CLDT', 'CLDT-PA', 'CLDVF', 'CLDWW', 'CLDX', 'CLEV', 'CLF', 'CLFD',
    'CLGN', 'CLH', 'CLIK', 'CLIR', 'CLLS', 'CLM', 'CLMB', 'CLMT', 'CLNE', 'CLNN',
    'CLNNW', 'CLNV', 'CLOQ', 'CLOV', 'CLOW', 'CLPR', 'CLPS', 'CLPT', 'CLRB', 'CLRCF',
    'CLRI', 'CLRLY', 'CLRO', 'CLRRF', 'CLRUF', 'CLRWF', 'CLS', 'CLSD', 'CLSK', 'CLSKW',
    'CLSO', 'CLST', 'CLUS', 'CLVT', 'CLW', 'CLWT', 'CLX', 'CLYM', 'CM', 'CMA',
    'CMA-PB', 'CMBM', 'CMBT', 'CMC', 'CMCL', 'CMCM', 'CMCO', 'CMCSA', 'CMCT', 'CMDB',
    'CME', 'CMG', 'CMGHF', 'CMGMF', 'CMHSF', 'CMI', 'CMLS', 'CMMB', 'CMND', 'CMP',
    'CMPO', 'CMPOW', 'CMPR', 'CMPS', 'CMPX', 'CMRC', 'CMRE', 'CMRE-PB', 'CMRE-PC', 'CMRE-PD',
    'CMRF', 'CMS', 'CMS-PB', 'CMS-PC', 'CMSA', 'CMSC', 'CMSD', 'CMT', 'CMTG', 'CMTL',
    'CMTV', 'CMU', 'CNA', 'CNBX', 'CNC', 'CNCK', 'CNCKW', 'CNDA', 'CNDAU', 'CNDAW',
    'CNDHF', 'CNDIF', 'CNDT', 'CNET', 'CNEY', 'CNF', 'CNFN', 'CNH', 'CNI', 'CNK',
    'CNL', 'CNLHN', 'CNLHO', 'CNLHP', 'CNLPL', 'CNLPM', 'CNLTL', 'CNLTN', 'CNLTP', 'CNM',
    'CNMD', 'CNNE', 'CNO', 'CNO-PA', 'CNOB', 'CNOBF', 'CNOBP', 'CNP', 'CNPWM', 'CNPWP',
    'CNQ', 'CNR', 'CNRCF', 'CNS', 'CNSP', 'CNTA', 'CNTB', 'CNTHN', 'CNTHO', 'CNTHP',
    'CNTM', 'CNTMF', 'CNTX', 'CNTY', 'CNVEF', 'CNVS', 'CNX', 'CNXC', 'CNXN', 'COBA',
    'COCH', 'COCHW', 'COCO', 'COCP', 'COCSF', 'CODA', 'CODGF', 'CODI', 'CODI-PA', 'CODI-PB',
    'CODI-PC', 'CODQL', 'CODX', 'COE', 'COEP', 'COEPW', 'COF', 'COF-PI', 'COF-PJ', 'COF-PK',
    'COF-PL', 'COF-PN', 'COFS', 'COGT', 'COHN', 'COHR', 'COHU', 'COIN', 'COKE', 'COLA',
    'COLAR', 'COLAU', 'COLB', 'COLD', 'COLL', 'COLM', 'COMM', 'COMP', 'CON', 'CONC',
    'COO', 'COOK', 'COOT', 'COOTW', 'COP', 'COPL', 'COPL-UN', 'COPL-WT', 'COPR', 'COR',
    'CORN', 'CORT', 'CORZ', 'CORZR', 'CORZW', 'CORZZ', 'COSG', 'COSM', 'COSO', 'COST',
    'COTY', 'COUR', 'COYA', 'CP', 'CPA', 'CPAC', 'CPAY', 'CPB', 'CPBI', 'CPCPF',
    'CPER', 'CPF', 'CPHC', 'CPHI', 'CPIVF', 'CPIX', 'CPK', 'CPKF', 'CPMD', 'CPMV',
    'CPNG', 'CPNNF', 'CPOP', 'CPPBY', 'CPPMF', 'CPPTL', 'CPRHF', 'CPRI', 'CPRT', 'CPRX',
    'CPS', 'CPSH', 'CPSS', 'CPT', 'CPTP', 'CPWPF', 'CPXWF', 'CPXXY', 'CPZ', 'CQP',
    'CR', 'CRAC', 'CRACU', 'CRAI', 'CRAQ', 'CRAQR', 'CRAQU', 'CRARF', 'CRAUY', 'CRAWA',
    'CRBD', 'CRBG', 'CRBP', 'CRBU', 'CRC', 'CRCE', 'CRCL', 'CRCT', 'CRCUF', 'CRCW',
    'CRD-A', 'CRD-B', 'CRDF', 'CRDL', 'CRDO', 'CRDV', 'CRE', 'CREG', 'CRERF', 'CRESW',
    'CRESY', 'CREV', 'CREVW', 'CREX', 'CRF', 'CRGO', 'CRGOW', 'CRGT', 'CRGY', 'CRH',
    'CRI', 'CRIS', 'CRK', 'CRKN', 'CRL', 'CRLBF', 'CRM', 'CRMD', 'CRML', 'CRMLW',
    'CRMT', 'CRMZ', 'CRNC', 'CRNT', 'CRNX', 'CRON', 'CROX', 'CRRFY', 'CRS', 'CRSF',
    'CRSP', 'CRSR', 'CRT', 'CRTAF', 'CRTD', 'CRTMF', 'CRTO', 'CRTUF', 'CRTWF', 'CRUCF',
    'CRUS', 'CRVL', 'CRVO', 'CRVS', 'CRVW', 'CRWD', 'CRWE', 'CRWS', 'CRWV', 'CSAI',
    'CSAN', 'CSBB', 'CSBR', 'CSC', 'CSCCF', 'CSCCY', 'CSCIF', 'CSCO', 'CSDX', 'CSGP',
    'CSGS', 'CSIQ', 'CSL', 'CSLMF', 'CSPI', 'CSQ', 'CSR', 'CSRIF', 'CSTAF', 'CSTE',
    'CSTL', 'CSTM', 'CSTUF', 'CSTWF', 'CSUI', 'CSV', 'CSW', 'CSWC', 'CSX', 'CTA-PA',
    'CTA-PB', 'CTAS', 'CTATF', 'CTBB', 'CTBI', 'CTDD', 'CTEV', 'CTGL', 'CTGO', 'CTKB',
    'CTKYY', 'CTLP', 'CTLPP', 'CTM', 'CTMX', 'CTNM', 'CTNT', 'CTO', 'CTO-PA', 'CTOR',
    'CTOS', 'CTPUF', 'CTRA', 'CTRE', 'CTRI', 'CTRM', 'CTRN', 'CTRVP', 'CTS', 'CTSH',
    'CTSO', 'CTSUF', 'CTSWF', 'CTTRF', 'CTVA', 'CTW', 'CTWO', 'CTXR', 'CUAUF', 'CUB',
    'CUBB', 'CUBE', 'CUBI', 'CUBI-PF', 'CUBT', 'CUBWU', 'CUBWW', 'CUE', 'CUK', 'CUKPF',
    'CULL', 'CULP', 'CUPPF', 'CUPR', 'CURB', 'CURI', 'CURLF', 'CURR', 'CURV', 'CURX',
    'CUZ', 'CV', 'CVAC', 'CVAT', 'CVBF', 'CVCO', 'CVE', 'CVE-WT', 'CVEO', 'CVGI',
    'CVGW', 'CVI', 'CVKD', 'CVLG', 'CVLT', 'CVM', 'CVNA', 'CVR', 'CVRX', 'CVS',
    'CVSI', 'CVU', 'CVV', 'CVVUF', 'CVX', 'CW', 'CWAN', 'CWBC', 'CWBHF', 'CWCO',
    'CWD', 'CWEN', 'CWEN-A', 'CWGL', 'CWH', 'CWK', 'CWLXF', 'CWNOF', 'CWPE', 'CWST',
    'CWT', 'CX', 'CXAI', 'CXAIW', 'CXDO', 'CXE', 'CXH', 'CXM', 'CXMSF', 'CXT',
    'CXW', 'CXXIF', 'CYAN', 'CYATY', 'CYBN', 'CYBR', 'CYCA', 'CYCN', 'CYCU', 'CYCUW',
    'CYD', 'CYDY', 'CYH', 'CYJBF', 'CYN', 'CYPH', 'CYRX', 'CYTK', 'CYTOF', 'CZFS',
    'CZNC', 'CZR', 'CZWI', 'D', 'DAAQ', 'DAAQU', 'DAAQW', 'DAC', 'DAIC', 'DAICW',
    'DAIO', 'DAKT', 'DAL', 'DAN', 'DAO', 'DAR', 'DARE', 'DASH', 'DAVA', 'DAVE',
    'DAVEW', 'DAVI', 'DAWN', 'DAY', 'DAZSF', 'DB', 'DBA', 'DBB', 'DBC', 'DBCA',
    'DBD', 'DBE', 'DBGI', 'DBI', 'DBIM', 'DBL', 'DBMM', 'DBO', 'DBP', 'DBRG',
    'DBRG-PH', 'DBRG-PI', 'DBRG-PJ', 'DBVT', 'DBVTF', 'DBX', 'DC', 'DC-WT', 'DCBO', 'DCFBS',
    'DCGO', 'DCI', 'DCO', 'DCOM', 'DCOMG', 'DCOMP', 'DCTH', 'DD', 'DDC', 'DDD',
    'DDHLY', 'DDI', 'DDL', 'DDOG', 'DDS', 'DDT', 'DE', 'DEA', 'DEC', 'DECK',
    'DEENF', 'DEFG', 'DEFI', 'DEFT', 'DEI', 'DELL', 'DENN', 'DEO', 'DERM', 'DEVS',
    'DFDV', 'DFDVW', 'DFH', 'DFIN', 'DFLI', 'DFLIW', 'DFP', 'DFPH', 'DFSC', 'DFSCW',
    'DG', 'DGEAF', 'DGICA', 'DGICB', 'DGII', 'DGLY', 'DGMDF', 'DGNX', 'DGP', 'DGTEF',
    'DGX', 'DGXX', 'DGZ', 'DH', 'DHAI', 'DHC', 'DHCNI', 'DHCNL', 'DHF', 'DHI',
    'DHIL', 'DHR', 'DHT', 'DHTI', 'DHX', 'DHY', 'DIA', 'DIAX', 'DIBS', 'DIDIY',
    'DIN', 'DINO', 'DINRF', 'DIOD', 'DIS', 'DIT', 'DJCO', 'DJP', 'DJT', 'DJTWW',
    'DK', 'DKI', 'DKILF', 'DKL', 'DKNG', 'DKS', 'DLAKF', 'DLAKY', 'DLB', 'DLHC',
    'DLMAF', 'DLMAY', 'DLNG', 'DLNG-PA', 'DLO', 'DLPN', 'DLR', 'DLR-PJ', 'DLR-PK', 'DLR-PL',
    'DLTH', 'DLTR', 'DLX', 'DLXY', 'DLY', 'DMA', 'DMAA', 'DMAAR', 'DMAAU', 'DMAC',
    'DMB', 'DMII', 'DMIIU', 'DMLP', 'DMNIF', 'DMO', 'DMRC', 'DMXCF', 'DMYY', 'DMYYU',
    'DMYYW', 'DNA', 'DNABW', 'DNACF', 'DNCLY', 'DNLI', 'DNMX', 'DNMXU', 'DNMXW', 'DNN',
    'DNOPF', 'DNOW', 'DNP', 'DNTH', 'DNUT', 'DOC', 'DOCN', 'DOCS', 'DOCU', 'DOGP',
    'DOGZ', 'DOLE', 'DOMH', 'DOMO', 'DOOO', 'DORM', 'DOT', 'DOUG', 'DOV', 'DOW',
    'DOWAY', 'DOX', 'DOYU', 'DPCDF', 'DPDSY', 'DPG', 'DPLMF', 'DPLS', 'DPLSD', 'DPRO',
    'DPUI', 'DPZ', 'DQ', 'DQJCF', 'DQWS', 'DRCT', 'DRD', 'DRDB', 'DRDBU', 'DRDBW',
    'DRDGF', 'DREM', 'DRH', 'DRH-PA', 'DRI', 'DRIO', 'DRMA', 'DRMAW', 'DROR', 'DRRKD',
    'DRS', 'DRSHF', 'DRTS', 'DRTSW', 'DRTTF', 'DRUG', 'DRVN', 'DRYGF', 'DSAC', 'DSECF',
    'DSEEY', 'DSFIY', 'DSGN', 'DSGR', 'DSGX', 'DSL', 'DSM', 'DSMFF', 'DSNY', 'DSP',
    'DSRNY', 'DSS', 'DSU', 'DSWL', 'DSX', 'DSX-PB', 'DSX-WT', 'DSY', 'DSYWW', 'DT',
    'DTB', 'DTCK', 'DTDT', 'DTE', 'DTEAF', 'DTEGF', 'DTEGY', 'DTF', 'DTG', 'DTGI',
    'DTI', 'DTII', 'DTIL', 'DTK', 'DTLAP', 'DTM', 'DTSQ', 'DTSQR', 'DTSQU', 'DTSS',
    'DTST', 'DTSTW', 'DTW', 'DTZNY', 'DTZZF', 'DUK', 'DUK-PA', 'DUKB', 'DUKR', 'DULL',
    'DUO', 'DUOL', 'DUOT', 'DUTBF', 'DV', 'DVA', 'DVAX', 'DVLT', 'DVN', 'DVS',
    'DWAY', 'DWNX', 'DWSN', 'DWTX', 'DWWYF', 'DX', 'DX-PC', 'DXC', 'DXCM', 'DXF',
    'DXG', 'DXLG', 'DXPE', 'DXR', 'DXST', 'DXYN', 'DXYZ', 'DY', 'DYAI', 'DYBTY',
    'DYN', 'DYNR', 'DYNT', 'DYOR', 'DYORU', 'DYORW', 'DZZ', 'E', 'EA', 'EACO',
    'EAD', 'EADSF', 'EAF', 'EAI', 'EARN', 'EAT', 'EAXR', 'EB', 'EBAY', 'EBBGF',
    'EBBNF', 'EBC', 'EBCRY', 'EBF', 'EBFI', 'EBGEF', 'EBMT', 'EBON', 'EBOSF', 'EBOSY',
    'EBRCZ', 'EBRGF', 'EBRZF', 'EBS', 'EBZT', 'EC', 'ECAT', 'ECBK', 'ECC', 'ECC-PD',
    'ECCC', 'ECCF', 'ECCU', 'ECCV', 'ECCW', 'ECCX', 'ECDA', 'ECDAW', 'ECF', 'ECF-PA',
    'ECG', 'ECIA', 'ECL', 'ECO', 'ECOR', 'ECOX', 'ECPG', 'ECPL', 'ECRO', 'ECST',
    'ECTM', 'ECVT', 'ECX', 'ECXJ', 'ECXWW', 'ED', 'EDAP', 'EDBL', 'EDBLW', 'EDD',
    'EDF', 'EDGM', 'EDHL', 'EDIT', 'EDN', 'EDPFY', 'EDRY', 'EDSA', 'EDTK', 'EDU',
    'EDUC', 'EDVGF', 'EDVLY', 'EDVR', 'EDXC', 'EE', 'EEA', 'EEFT', 'EEGI', 'EEIQ',
    'EESH', 'EEX', 'EFC', 'EFC-PA', 'EFC-PB', 'EFC-PC', 'EFC-PD', 'EFOI', 'EFR', 'EFSC',
    'EFSCP', 'EFSI', 'EFT', 'EFTY', 'EFX', 'EFXT', 'EG', 'EGAN', 'EGBN', 'EGG',
    'EGHA', 'EGHAR', 'EGHAU', 'EGHT', 'EGLXF', 'EGMCF', 'EGO', 'EGP', 'EGTYF', 'EGY',
    'EH', 'EHAB', 'EHC', 'EHGO', 'EHI', 'EHLD', 'EHSI', 'EHTH', 'EHVVF', 'EIC',
    'EICA', 'EICB', 'EICC', 'EIG', 'EIIA', 'EIM', 'EIPAF', 'EIX', 'EJH', 'EJPRF',
    'EKSO', 'EL', 'ELA', 'ELAB', 'ELAN', 'ELBM', 'ELC', 'ELCG', 'ELCPF', 'ELDCF',
    'ELDN', 'ELE', 'ELECF', 'ELEEF', 'ELEMF', 'ELF', 'ELFTY', 'ELLO', 'ELMD', 'ELME',
    'ELOG', 'ELP', 'ELPC', 'ELPW', 'ELRA', 'ELRE', 'ELS', 'ELSE', 'ELTK', 'ELTP',
    'ELTX', 'ELUT', 'ELV', 'ELVA', 'ELVG', 'ELVN', 'ELVR', 'ELWS', 'ELWT', 'EM',
    'EMA', 'EMBC', 'EMBJ', 'EMBYF', 'EMCGF', 'EMCRF', 'EMCWF', 'EMD', 'EME', 'EMED',
    'EMF', 'EMGDF', 'EMI', 'EMICF', 'EMIS', 'EMISR', 'EML', 'EMMA', 'EMN', 'EMO',
    'EMP', 'EMPD', 'EMPG', 'EMR', 'EMRH', 'EMRJF', 'EMYB', 'ENAKF', 'ENB', 'ENBFF',
    'ENBGF', 'ENBHF', 'ENBMF', 'ENBNF', 'ENBOF', 'ENBP', 'ENBRF', 'ENBSF', 'ENDI', 'ENDMF',
    'ENDV', 'ENFY', 'ENGN', 'ENGNW', 'ENGS', 'ENIC', 'ENJ', 'ENLT', 'ENLV', 'ENNPF',
    'ENO', 'ENOV', 'ENPH', 'ENR', 'ENRT', 'ENS', 'ENSC', 'ENSCW', 'ENSG', 'ENTA',
    'ENTG', 'ENTO', 'ENTX', 'ENTXW', 'ENVA', 'ENVB', 'ENVX', 'ENZN', 'EOD', 'EOG',
    'EOI', 'EOLS', 'EONGY', 'EONR', 'EONR-WT', 'EOS', 'EOSE', 'EOT', 'EP', 'EP-PC',
    'EPAC', 'EPAM', 'EPC', 'EPD', 'EPDU', 'EPM', 'EPOW', 'EPR', 'EPR-PC', 'EPR-PE',
    'EPR-PG', 'EPRT', 'EPRX', 'EPSM', 'EPSN', 'EPWK', 'EQ', 'EQBK', 'EQH', 'EQH-PA',
    'EQH-PC', 'EQIX', 'EQNR', 'EQR', 'EQS', 'EQT', 'EQTRF', 'EQX', 'ERAS', 'ERC',
    'ERELY', 'ERH', 'ERIC', 'ERIE', 'ERII', 'ERIXF', 'ERLFF', 'ERNA', 'ERO', 'ERRAF',
    'ES', 'ESAB', 'ESAUF', 'ESBA', 'ESCA', 'ESE', 'ESEA', 'ESGH', 'ESGL', 'ESGLW',
    'ESHA', 'ESHAR', 'ESHSF', 'ESI', 'ESLA', 'ESLAW', 'ESLT', 'ESMC', 'ESNT', 'ESOA',
    'ESP', 'ESPR', 'ESQ', 'ESRT', 'ESS', 'ESSI', 'ESTA', 'ESTC', 'ET', 'ET-PI',
    'ETB', 'ETCG', 'ETD', 'ETG', 'ETH', 'ETHA', 'ETHE', 'ETHM', 'ETHMU', 'ETHMW',
    'ETHV', 'ETHW', 'ETHZ', 'ETI-P', 'ETJ', 'ETN', 'ETO', 'ETON', 'ETOR', 'ETR',
    'ETS', 'ETST', 'ETSY', 'ETUGF', 'ETV', 'ETW', 'ETX', 'ETY', 'EU', 'EUBG',
    'EUDA', 'EUDAW', 'EUEMF', 'EUO', 'EURK', 'EURKR', 'EURKU', 'EVAC', 'EVAC-UN', 'EVAC-WT',
    'EVAX', 'EVC', 'EVCM', 'EVER', 'EVEX', 'EVEX-WT', 'EVF', 'EVFM', 'EVG', 'EVGN',
    'EVGO', 'EVGOW', 'EVH', 'EVI', 'EVLLF', 'EVLV', 'EVLVW', 'EVMN', 'EVN', 'EVO',
    'EVOH', 'EVOK', 'EVON', 'EVOTF', 'EVOX', 'EVOXU', 'EVR', 'EVRG', 'EVT', 'EVTC',
    'EVTK', 'EVTL', 'EVTV', 'EVTWF', 'EVV', 'EW', 'EWBC', 'EWCZ', 'EWSB', 'EWTX',
    'EXAS', 'EXC', 'EXDW', 'EXE', 'EXEEL', 'EXEEW', 'EXEEZ', 'EXEL', 'EXFY', 'EXG',
    'EXK', 'EXLS', 'EXNRF', 'EXOD', 'EXOZ', 'EXP', 'EXPD', 'EXPE', 'EXPI', 'EXPO',
    'EXR', 'EXTR', 'EYE', 'EYGPF', 'EYPT', 'EYUBY', 'EYUUF', 'EZBC', 'EZET', 'EZGO',
    'EZOO', 'EZPW', 'EZPZ', 'F', 'F-PB', 'F-PC', 'F-PD', 'FA', 'FACT', 'FACTU',
    'FACTW', 'FAF', 'FALC', 'FAMI', 'FANG', 'FANUF', 'FARM', 'FAST', 'FAT', 'FATBB',
    'FATBP', 'FATE', 'FATN', 'FAX', 'FBCD', 'FBDS', 'FBGL', 'FBIN', 'FBIO', 'FBIOP',
    'FBIZ', 'FBK', 'FBLA', 'FBLG', 'FBNC', 'FBP', 'FBRT', 'FBRT-PE', 'FBRX', 'FBSI',
    'FBTC', 'FBYD', 'FBYDW', 'FC', 'FCAP', 'FCBC', 'FCCI', 'FCCN', 'FCCO', 'FCEL',
    'FCELB', 'FCF', 'FCFS', 'FCHDF', 'FCHL', 'FCHRF', 'FCHS', 'FCN', 'FCNCA', 'FCNCB',
    'FCNCO', 'FCNCP', 'FCO', 'FCPT', 'FCRS', 'FCRS-UN', 'FCRS-WT', 'FCRX', 'FCT', 'FCUV',
    'FCX', 'FDBC', 'FDCT', 'FDMT', 'FDP', 'FDS', 'FDSB', 'FDUS', 'FDX', 'FDXTF',
    'FE', 'FEAM', 'FEAV', 'FEBO', 'FECOF', 'FEDU', 'FEIM', 'FELE', 'FEMY', 'FENC',
    'FENG', 'FER', 'FERA', 'FERAR', 'FERAU', 'FERG', 'FET', 'FETH', 'FF', 'FFA',
    'FFAI', 'FFAIW', 'FFBC', 'FFC', 'FFIC', 'FFIN', 'FFIV', 'FFLO', 'FFMGF', 'FFWM',
    'FG', 'FGBI', 'FGBIP', 'FGCO', 'FGDL', 'FGEN', 'FGFH', 'FGHFF', 'FGI', 'FGIWW',
    'FGL', 'FGMC', 'FGMCR', 'FGMCU', 'FGN', 'FGNV', 'FGNX', 'FGNXP', 'FGO', 'FGOVF',
    'FGPR', 'FGPRB', 'FGSN', 'FGXC', 'FHB', 'FHFFY', 'FHI', 'FHLD', 'FHN', 'FHN-PC',
    'FHN-PE', 'FHN-PF', 'FHTX', 'FIBK', 'FICO', 'FIEE', 'FIG', 'FIGP', 'FIGR', 'FIGS',
    'FIGX', 'FIGXU', 'FIGXW', 'FIHL', 'FIISO', 'FIISP', 'FILG', 'FINGF', 'FINS', 'FINV',
    'FINW', 'FIP', 'FIS', 'FISI', 'FISK', 'FISV', 'FIT', 'FITB', 'FITBI', 'FITBO',
    'FITBP', 'FIVE', 'FIVN', 'FIX', 'FIZZ', 'FKST', 'FKURF', 'FKWL', 'FKYS', 'FLC',
    'FLD', 'FLDDW', 'FLEX', 'FLG', 'FLG-PA', 'FLG-PU', 'FLGC', 'FLGT', 'FLL', 'FLNC',
    'FLNCF', 'FLNG', 'FLNT', 'FLO', 'FLOC', 'FLR', 'FLS', 'FLUT', 'FLUX', 'FLWS',
    'FLX', 'FLXS', 'FLY', 'FLYD', 'FLYE', 'FLYU', 'FLYW', 'FLYX', 'FLYX-WT', 'FLYYQ',
    'FMAO', 'FMBH', 'FMBM', 'FMC', 'FMCB', 'FMCC', 'FMCCG', 'FMCCH', 'FMCCI', 'FMCCJ',
    'FMCCK', 'FMCCL', 'FMCCM', 'FMCCN', 'FMCCO', 'FMCCP', 'FMCCS', 'FMCCT', 'FMCKI', 'FMCKJ',
    'FMCKK', 'FMCKL', 'FMCKM', 'FMCKN', 'FMCKO', 'FMCKP', 'FMCQF', 'FMCXF', 'FMFC', 'FMFG',
    'FMHS', 'FMN', 'FMNB', 'FMS', 'FMST', 'FMSTW', 'FMTOF', 'FMX', 'FMY', 'FN',
    'FNB', 'FNBT', 'FNCH', 'FNCTF', 'FND', 'FNF', 'FNFI', 'FNFPA', 'FNGD', 'FNGO',
    'FNGR', 'FNGS', 'FNGU', 'FNIGY', 'FNKO', 'FNLC', 'FNMA', 'FNMAG', 'FNMAH', 'FNMAI',
    'FNMAJ', 'FNMAK', 'FNMAL', 'FNMAM', 'FNMAN', 'FNMAO', 'FNMAP', 'FNMAS', 'FNMAT', 'FNMCF',
    'FNMFM', 'FNMFN', 'FNMFO', 'FNRN', 'FNV', 'FNWB', 'FNWD', 'FNXMF', 'FOA', 'FOACW',
    'FOF', 'FOFA', 'FOFO', 'FOLD', 'FONR', 'FOR', 'FORA', 'FORFF', 'FORL', 'FORLU',
    'FORLW', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOUR', 'FOUR-PA', 'FOX', 'FOXA', 'FOXF',
    'FOXO', 'FOXOW', 'FOXX', 'FOXXW', 'FPF', 'FPH', 'FPI', 'FPRGF', 'FR', 'FRA',
    'FRAF', 'FRBP', 'FRD', 'FRECF', 'FREGP', 'FREJN', 'FREJO', 'FREJP', 'FREVS', 'FRFAF',
    'FRFFF', 'FRFHF', 'FRFXF', 'FRGE', 'FRGT', 'FRHC', 'FRLCY', 'FRME', 'FRMEP', 'FRMI',
    'FRO', 'FROG', 'FRPH', 'FRPT', 'FRQN', 'FRSH', 'FRSPF', 'FRST', 'FRSX', 'FRT',
    'FRT-PC', 'FRTSF', 'FRZT', 'FSBC', 'FSBW', 'FSCO', 'FSEA', 'FSFG', 'FSHP', 'FSHPR',
    'FSHPU', 'FSI', 'FSK', 'FSLR', 'FSLY', 'FSM', 'FSNUF', 'FSNUY', 'FSOL', 'FSP',
    'FSS', 'FSSL', 'FSTJ', 'FSTR', 'FSTWF', 'FSUN', 'FSV', 'FT', 'FTAI', 'FTAIM',
    'FTAIN', 'FTCI', 'FTCO', 'FTDR', 'FTEK', 'FTEL', 'FTF', 'FTFT', 'FTHM', 'FTHY',
    'FTI', 'FTII', 'FTIIU', 'FTK', 'FTLF', 'FTNT', 'FTPSF', 'FTRE', 'FTRK', 'FTRSF',
    'FTS', 'FTSP', 'FTV', 'FTW', 'FTW-UN', 'FTW-WT', 'FTY', 'FTZFF', 'FUBO', 'FUEMF',
    'FUFU', 'FUFUW', 'FUJIF', 'FUJIY', 'FUL', 'FULC', 'FULO', 'FULT', 'FULTP', 'FUN',
    'FUNC', 'FUND', 'FUNI', 'FURCF', 'FURY', 'FUSB', 'FUSE', 'FUSEW', 'FUST', 'FUTU',
    'FVCB', 'FVGPY', 'FVN', 'FVNNR', 'FVNNU', 'FVR', 'FVRR', 'FVTI', 'FWDI', 'FWFW',
    'FWONA', 'FWONB', 'FWONK', 'FWRD', 'FWRG', 'FXA', 'FXB', 'FXBY', 'FXC', 'FXCND',
    'FXCXD', 'FXE', 'FXF', 'FXNC', 'FXY', 'FYBR', 'FYNN', 'FZMD', 'G', 'GAB',
    'GAB-PG', 'GAB-PH', 'GAB-PK', 'GABC', 'GACW', 'GADA', 'GAERF', 'GAFC', 'GAIA', 'GAIN',
    'GAINI', 'GAINL', 'GAINN', 'GAINZ', 'GALDY', 'GALT', 'GAM', 'GAM-PB', 'GAMB', 'GAME',
    'GANX', 'GAP', 'GARLF', 'GARWF', 'GASS', 'GATX', 'GAU', 'GAUZ', 'GBAB', 'GBAT',
    'GBCI', 'GBCS', 'GBDC', 'GBFH', 'GBH', 'GBIO', 'GBLI', 'GBNXF', 'GBNXY', 'GBNY',
    'GBR', 'GBTC', 'GBTG', 'GBUG', 'GBUX', 'GBX', 'GCAN', 'GCBC', 'GCGJ', 'GCL',
    'GCLWW', 'GCMG', 'GCO', 'GCT', 'GCTK', 'GCTS', 'GCTS-WT', 'GCUMF', 'GCV', 'GD',
    'GDC', 'GDDY', 'GDEL', 'GDEN', 'GDERF', 'GDEV', 'GDEVW', 'GDHG', 'GDHLF', 'GDL',
    'GDLC', 'GDLG', 'GDNT', 'GDO', 'GDOG', 'GDOT', 'GDRX', 'GDRZF', 'GDS', 'GDST',
    'GDSTR', 'GDSTU', 'GDSTW', 'GDTC', 'GDV', 'GDV-PH', 'GDV-PK', 'GDXD', 'GDXU', 'GDYN',
    'GE', 'GEBRF', 'GECC', 'GECCG', 'GECCH', 'GECCI', 'GECCO', 'GEDC', 'GEF', 'GEF-B',
    'GEG', 'GEGGL', 'GEGP', 'GEHC', 'GEHDF', 'GEL', 'GELHY', 'GELS', 'GELYF', 'GEMI',
    'GEN', 'GENC', 'GENI', 'GENK', 'GENVR', 'GEO', 'GEOS', 'GERN', 'GERNW', 'GES',
    'GETY', 'GEV', 'GEVI', 'GEVO', 'GF', 'GFAI', 'GFAIW', 'GFASY', 'GFCHY', 'GFF',
    'GFI', 'GFIOF', 'GFL', 'GFLT', 'GFMH', 'GFR', 'GFR-RI', 'GFRWF', 'GFS', 'GFSAY',
    'GFTTY', 'GGAL', 'GGAZF', 'GGB', 'GGBY', 'GGG', 'GGLXF', 'GGN', 'GGN-PB', 'GGPSF',
    'GGR', 'GGROU', 'GGROW', 'GGRPY', 'GGT', 'GGT-PE', 'GGT-PG', 'GGTKY', 'GGZ', 'GH',
    'GHBWF', 'GHC', 'GHG', 'GHI', 'GHLD', 'GHM', 'GHRS', 'GHRTF', 'GHST', 'GHY',
    'GIB', 'GIBO', 'GIBOW', 'GIC', 'GIDMF', 'GIFI', 'GIFLF', 'GIFOF', 'GIFT', 'GIG',
    'GIGGF', 'GIGGU', 'GIGGW', 'GIGM', 'GIII', 'GIKLY', 'GIL', 'GILD', 'GILT', 'GINT',
    'GIPL', 'GIPR', 'GIPRW', 'GIS', 'GITS', 'GIW', 'GIWWR', 'GIWWU', 'GJH', 'GJO',
    'GJP', 'GJR', 'GJS', 'GJT', 'GKOR', 'GKOS', 'GL', 'GL-PD', 'GLABF', 'GLAD',
    'GLAI', 'GLASF', 'GLATF', 'GLAXF', 'GLBE', 'GLBS', 'GLBZ', 'GLCP', 'GLD', 'GLDD',
    'GLDG', 'GLDI', 'GLDM', 'GLDW', 'GLE', 'GLEI', 'GLGI', 'GLIBA', 'GLIBB', 'GLIBK',
    'GLIBR', 'GLIV', 'GLL', 'GLLI', 'GLMD', 'GLNG', 'GLNK', 'GLNS', 'GLO', 'GLOB',
    'GLOO', 'GLOP-PA', 'GLOP-PB', 'GLOP-PC', 'GLP', 'GLP-PB', 'GLPG', 'GLPGF', 'GLPI', 'GLQ',
    'GLRE', 'GLSA', 'GLSI', 'GLTK', 'GLTO', 'GLTR', 'GLU', 'GLU-PA', 'GLU-PB', 'GLUC',
    'GLUE', 'GLV', 'GLVHF', 'GLVPY', 'GLVT', 'GLW', 'GLXG', 'GLXY', 'GLXZ', 'GM',
    'GMAB', 'GMBL', 'GMBLP', 'GMBLZ', 'GME', 'GME-WT', 'GMED', 'GMER', 'GMGI', 'GMHS',
    'GMM', 'GMPW', 'GMRE', 'GMRE-PA', 'GMRE-PB', 'GMTH', 'GMWKF', 'GMZP', 'GNE', 'GNFTY',
    'GNK', 'GNL', 'GNL-PA', 'GNL-PB', 'GNL-PD', 'GNL-PE', 'GNLN', 'GNLX', 'GNMSF', 'GNOLF',
    'GNPX', 'GNRC', 'GNS', 'GNSS', 'GNT', 'GNT-PA', 'GNTA', 'GNTOF', 'GNTX', 'GNVR',
    'GNW', 'GO', 'GOAI', 'GOCO', 'GOEVQ', 'GOF', 'GOGO', 'GOGR', 'GOLF', 'GONYF',
    'GOOD', 'GOODN', 'GOODO', 'GOOG', 'GOOGL', 'GOOS', 'GORO', 'GORV', 'GOSS', 'GOTU',
    'GOVB', 'GOVX', 'GP', 'GPAEF', 'GPAT', 'GPATU', 'GPATW', 'GPC', 'GPCR', 'GPGCF',
    'GPI', 'GPJA', 'GPK', 'GPLB', 'GPLL', 'GPMT', 'GPMT-PA', 'GPMTF', 'GPN', 'GPOR',
    'GPOX', 'GPRE', 'GPRK', 'GPRO', 'GPTGF', 'GPUS', 'GPUS-PD', 'GPUSF', 'GRAB', 'GRABW',
    'GRAF', 'GRAF-UN', 'GRAF-WT', 'GRAL', 'GRAN', 'GRAY', 'GRBK', 'GRBK-PA', 'GRC', 'GRCE',
    'GRCPY', 'GRDAF', 'GRDN', 'GREE', 'GREEL', 'GREH', 'GRF', 'GRFS', 'GRFXF', 'GRFXY',
    'GRHI', 'GRI', 'GRLF', 'GRLMF', 'GRMN', 'GRN', 'GRND', 'GRNL', 'GRNQ', 'GRNT',
    'GRO', 'GROV', 'GROVW', 'GROW', 'GROY', 'GROY-WT', 'GRP-UN', 'GRPFF', 'GRPN', 'GRPS',
    'GRRR', 'GRRRW', 'GRTX', 'GRUSF', 'GRVE', 'GRVY', 'GRWFF', 'GRWG', 'GRWLF', 'GRWTF',
    'GRX', 'GS', 'GS-PA', 'GS-PC', 'GS-PD', 'GSAC', 'GSAT', 'GSBC', 'GSBD', 'GSCCF',
    'GSCE', 'GSG', 'GSHD', 'GSHR', 'GSHRU', 'GSHRW', 'GSIT', 'GSIW', 'GSK', 'GSL',
    'GSL-PB', 'GSM', 'GSNR', 'GSOL', 'GSPI', 'GSRF', 'GSRFR', 'GSRFU', 'GST', 'GSTK',
    'GSTX', 'GSUI', 'GSUN', 'GSVRF', 'GT', 'GTBIF', 'GTBP', 'GTCH', 'GTE', 'GTEC',
    'GTEN', 'GTENU', 'GTENW', 'GTERA', 'GTERR', 'GTERU', 'GTERW', 'GTES', 'GTHP', 'GTIC',
    'GTII', 'GTIJF', 'GTIM', 'GTLB', 'GTLL', 'GTLRY', 'GTLS', 'GTLS-PB', 'GTM', 'GTMAY',
    'GTN', 'GTN-A', 'GTSG', 'GTX', 'GTY', 'GUER', 'GUG', 'GULTU', 'GURE', 'GUT',
    'GUT-PC', 'GUTS', 'GUYGF', 'GV', 'GVA', 'GVH', 'GVSE', 'GWAV', 'GWH', 'GWH-WT',
    'GWKSY', 'GWLIF', 'GWLL', 'GWLPF', 'GWRE', 'GWRS', 'GWSO', 'GWTI', 'GWW', 'GXAI',
    'GXLM', 'GXO', 'GXRP', 'GXXM', 'GYGLF', 'GYRE', 'GYRO', 'H', 'HACBF', 'HAE',
    'HAFC', 'HAFG', 'HAFN', 'HAIAF', 'HAIN', 'HAL', 'HALO', 'HAMA', 'HAMRF', 'HANNF',
    'HAO', 'HAPVD', 'HAS', 'HASI', 'HAVA', 'HAVAU', 'HAYW', 'HBAN', 'HBANL', 'HBANM',
    'HBANP', 'HBAR', 'HBB', 'HBCP', 'HBCYF', 'HBI', 'HBIA', 'HBIO', 'HBM', 'HBNB',
    'HBNC', 'HBT', 'HBUV', 'HCA', 'HCAC', 'HCACU', 'HCAI', 'HCAT', 'HCC', 'HCHL',
    'HCI', 'HCIIP', 'HCIL', 'HCKT', 'HCM', 'HCMA', 'HCMAU', 'HCMAW', 'HCMC', 'HCSG',
    'HCTI', 'HCWB', 'HCWC', 'HCXY', 'HD', 'HDB', 'HDL', 'HDLB', 'HDSN', 'HE',
    'HEI', 'HEI-A', 'HELE', 'HEPA', 'HEPS', 'HEQ', 'HERE', 'HERZ', 'HESM', 'HFBL',
    'HFFG', 'HFRO', 'HFRO-PA', 'HFRO-PB', 'HFUS', 'HFWA', 'HG', 'HGAS', 'HGASW', 'HGBL',
    'HGLB', 'HGMCF', 'HGTXU', 'HGTY', 'HGV', 'HGYN', 'HHH', 'HHHEF', 'HHLKF', 'HHS',
    'HI', 'HIFI', 'HIG', 'HIG-PG', 'HIGR', 'HIHO', 'HII', 'HIMS', 'HIMX', 'HIND',
    'HIO', 'HIPO', 'HIPOW', 'HIRU', 'HIT', 'HITI', 'HIVE', 'HIW', 'HIX', 'HKD',
    'HKHC', 'HKHHF', 'HKHHY', 'HKIT', 'HKPD', 'HL', 'HL-PB', 'HLEO', 'HLF', 'HLI',
    'HLIO', 'HLIT', 'HLLK', 'HLLY', 'HLLY-WT', 'HLMN', 'HLN', 'HLNCF', 'HLNE', 'HLP',
    'HLRTF', 'HLSCF', 'HLT', 'HLTC', 'HLX', 'HLYK', 'HMBL', 'HMC', 'HMDCF', 'HMELF',
    'HMH', 'HMKIY', 'HMMR', 'HMN', 'HMR', 'HMY', 'HNDAF', 'HNGE', 'HNI', 'HNIT',
    'HNNA', 'HNNAZ', 'HNOI', 'HNPHY', 'HNRG', 'HNSDF', 'HNSPF', 'HNST', 'HNVR', 'HODL',
    'HOFT', 'HOFV', 'HOG', 'HOLO', 'HOLOW', 'HOLX', 'HOMB', 'HON', 'HONE', 'HOOD',
    'HOOK', 'HOPE', 'HOTH', 'HOUR', 'HOUS', 'HOV', 'HOVNP', 'HOVR', 'HOVRW', 'HOVVB',
    'HOWL', 'HOYFF', 'HP', 'HPAI', 'HPAIW', 'HPE', 'HPE-PC', 'HPF', 'HPHTF', 'HPHTY',
    'HPI', 'HPK', 'HPP', 'HPP-PC', 'HPQ', 'HPS', 'HQH', 'HQI', 'HQL', 'HQY',
    'HR', 'HRB', 'HRBAY', 'HRGN', 'HRI', 'HRIBF', 'HRL', 'HRMY', 'HRNNF', 'HROW',
    'HRTG', 'HRTX', 'HRZN', 'HRZRF', 'HSAI', 'HSBC', 'HSCS', 'HSCSW', 'HSDT', 'HSDTW',
    'HSHP', 'HSIC', 'HSII', 'HSPO', 'HSPOR', 'HSPOU', 'HSPOW', 'HSPT', 'HSPTR', 'HSPTU',
    'HST', 'HSTA', 'HSTC', 'HSTI', 'HSTM', 'HSY', 'HTB', 'HTBK', 'HTCO', 'HTCR',
    'HTD', 'HTFB', 'HTFC', 'HTFL', 'HTGC', 'HTH', 'HTHIF', 'HTHIY', 'HTHT', 'HTLD',
    'HTLM', 'HTO', 'HTOO', 'HTOOW', 'HTZ', 'HTZWW', 'HUBB', 'HUBC', 'HUBCW', 'HUBCZ',
    'HUBG', 'HUBS', 'HUDI', 'HUHU', 'HUIZ', 'HUM', 'HUMA', 'HUMAW', 'HUN', 'HURA',
    'HURC', 'HURN', 'HUSA', 'HUT', 'HUYA', 'HVGDF', 'HVII', 'HVIIR', 'HVIIU', 'HVMC',
    'HVMCU', 'HVMCW', 'HVT', 'HVT-A', 'HWAIF', 'HWAUF', 'HWBK', 'HWC', 'HWCPZ', 'HWEP',
    'HWH', 'HWKE', 'HWKN', 'HWLDF', 'HWM', 'HWM-P', 'HWNI', 'HXHX', 'HXL', 'HY',
    'HYAC', 'HYAC-UN', 'HYAC-WT', 'HYDTF', 'HYEX', 'HYFM', 'HYFT', 'HYI', 'HYLN', 'HYMC',
    'HYMCW', 'HYNE', 'HYNLY', 'HYOR', 'HYPD', 'HYPMY', 'HYPR', 'HYPS', 'HYSR', 'HYT',
    'HZEN', 'HZO', 'HZRBY', 'IAC', 'IAE', 'IAF', 'IAG', 'IART', 'IAS', 'IAU',
    'IAUM', 'IAUX', 'IAUX-WT', 'IBAC', 'IBACR', 'IBATF', 'IBCP', 'IBEX', 'IBG', 'IBIDF',
    'IBIO', 'IBIT', 'IBKR', 'IBM', 'IBN', 'IBO', 'IBOC', 'IBP', 'IBRX', 'IBTA',
    'ICCC', 'ICCM', 'ICE', 'ICFI', 'ICG', 'ICHGF', 'ICHR', 'ICL', 'ICLR', 'ICMB',
    'ICON', 'ICR-PA', 'ICRP', 'ICTSF', 'ICU', 'ICUCW', 'ICUI', 'IDA', 'IDAI', 'IDCC',
    'IDE', 'IDEXF', 'IDKOF', 'IDKOY', 'IDN', 'IDPEY', 'IDPUF', 'IDR', 'IDT', 'IDTL',
    'IDVV', 'IDWM', 'IDXG', 'IDXMF', 'IDXX', 'IDYA', 'IE', 'IEAG', 'IEGCF', 'IEHC',
    'IEP', 'IESC', 'IEX', 'IFBD', 'IFCNF', 'IFED', 'IFF', 'IFHLY', 'IFN', 'IFNNF',
    'IFNNY', 'IFRX', 'IFS', 'IGA', 'IGAC', 'IGACU', 'IGC', 'IGD', 'IGI', 'IGIC',
    'IGR', 'IGTA', 'IGTAR', 'IGTAU', 'IGTAW', 'IH', 'IHD', 'IHETW', 'IHG', 'IHICF',
    'IHRT', 'IHRTB', 'IHS', 'IHT', 'IIF', 'III', 'IIIN', 'IIIV', 'IIM', 'IINN',
    'IINNW', 'IIPR', 'IIPR-PA', 'IKT', 'ILAG', 'ILAL', 'ILLMF', 'ILLR', 'ILLRW', 'ILMN',
    'ILPLF', 'ILPT', 'ILST', 'ILUS', 'IMA', 'IMAA', 'IMAQ', 'IMAQR', 'IMAQU', 'IMAQW',
    'IMAX', 'IMCC', 'IMCR', 'IMDX', 'IMG', 'IMIMF', 'IMKTA', 'IMMP', 'IMMR', 'IMMX',
    'IMNM', 'IMNN', 'IMO', 'IMOS', 'IMPM', 'IMPP', 'IMPPP', 'IMRN', 'IMRX', 'IMSR',
    'IMSRW', 'IMTCF', 'IMTE', 'IMTH', 'IMTX', 'IMUC', 'IMUX', 'IMVT', 'IMXI', 'INAB',
    'INAC', 'INACR', 'INACU', 'INBK', 'INBKZ', 'INBP', 'INBS', 'INBX', 'INCR', 'INCY',
    'INDB', 'INDI', 'INDO', 'INDP', 'INDV', 'INEO', 'INFT', 'INFU', 'INFY', 'ING',
    'INGM', 'INGN', 'INGR', 'INGVF', 'INHD', 'INHI', 'INIKF', 'INIS', 'INKS', 'INKT',
    'INKW', 'INLF', 'INLX', 'INM', 'INMB', 'INMD', 'INN', 'INN-PE', 'INN-PF', 'INND',
    'INNI', 'INNP', 'INNPF', 'INNV', 'INO', 'INOD', 'INPAP', 'INR', 'INRE', 'INSE',
    'INSG', 'INSM', 'INSP', 'INSW', 'INTA', 'INTC', 'INTG', 'INTI', 'INTJ', 'INTR',
    'INTS', 'INTT', 'INTU', 'INTZ', 'INUV', 'INV', 'INVA', 'INVE', 'INVH', 'INVLW',
    'INVU', 'INVUP', 'INVX', 'INVZ', 'INVZW', 'IOBT', 'IONI', 'IONQ', 'IONQ-WT', 'IONR',
    'IONS', 'IOR', 'IOSP', 'IOT', 'IOTR', 'IOVA', 'IP', 'IPAR', 'IPB', 'IPCX',
    'IPCXR', 'IPCXU', 'IPDN', 'IPEX', 'IPEXR', 'IPEXU', 'IPG', 'IPGP', 'IPHA', 'IPHYF',
    'IPI', 'IPM', 'IPOD', 'IPODU', 'IPODW', 'IPSAY', 'IPSC', 'IPSI', 'IPSOF', 'IPST',
    'IPTNF', 'IPW', 'IPWR', 'IPX', 'IQ', 'IQI', 'IQST', 'IQV', 'IR', 'IRBT',
    'IRD', 'IRDM', 'IREN', 'IRHO', 'IRIX', 'IRM', 'IRMD', 'IRME', 'IRON', 'IROQ',
    'IRRX', 'IRRXU', 'IRRXW', 'IRS', 'IRS-WT', 'IRT', 'IRTC', 'IRVRF', 'IRWD', 'ISBA',
    'ISCO', 'ISD', 'ISMCF', 'ISOU', 'ISPC', 'ISPO', 'ISPOW', 'ISPR', 'ISRG', 'ISRL',
    'ISRLU', 'ISRLW', 'ISSC', 'ISTKF', 'ISTR', 'IT', 'ITGR', 'ITHA', 'ITHR', 'ITHUF',
    'ITIC', 'ITMSF', 'ITOX', 'ITP', 'ITRG', 'ITRI', 'ITRM', 'ITRN', 'ITT', 'ITUB',
    'ITW', 'IVA', 'IVBXF', 'IVCAF', 'IVDA', 'IVDAW', 'IVDN', 'IVEVF', 'IVF', 'IVFH',
    'IVHI', 'IVNHW', 'IVP', 'IVR', 'IVR-PC', 'IVT', 'IVVD', 'IVZ', 'IWAL', 'IWDL',
    'IWFL', 'IWML', 'IWSH', 'IX', 'IXHL', 'IZEA', 'IZM', 'J', 'JACK', 'JACS',
    'JACS-RI', 'JACS-UN', 'JAGX', 'JAKK', 'JAMF', 'JANL', 'JANX', 'JAZZ', 'JBARF', 'JBDI',
    'JBGS', 'JBHIF', 'JBHT', 'JBI', 'JBIO', 'JBK', 'JBL', 'JBLU', 'JBS', 'JBSS',
    'JBTM', 'JCAP', 'JCE', 'JCI', 'JCSE', 'JCTC', 'JCYCF', 'JCYGY', 'JD', 'JDCMF',
    'JDZG', 'JEF', 'JELD', 'JELLF', 'JEM', 'JENA', 'JENA-RI', 'JENA-UN', 'JEQ', 'JETBF',
    'JETD', 'JETMF', 'JETR', 'JETU', 'JFB', 'JFBR', 'JFBRW', 'JFIL', 'JFIN', 'JFR',
    'JFU', 'JG', 'JGH', 'JGLDF', 'JGSHF', 'JHG', 'JHI', 'JHIUF', 'JHPCY', 'JHS',
    'JHX', 'JILL', 'JJETF', 'JJHR', 'JJSF', 'JKHY', 'JKS', 'JL', 'JLHL', 'JLL',
    'JLS', 'JMIA', 'JMM', 'JMSB', 'JNGHF', 'JNJ', 'JOB', 'JOBY', 'JOBY-WT', 'JOCM',
    'JOE', 'JOF', 'JOSS', 'JOUT', 'JOYY', 'JPC', 'JPM', 'JPM-PC', 'JPM-PD', 'JPM-PJ',
    'JPM-PK', 'JPM-PL', 'JPM-PM', 'JPOTF', 'JQC', 'JRI', 'JRS', 'JRSH', 'JRSS', 'JRVR',
    'JSDA', 'JSKJ', 'JSM', 'JSPR', 'JSPRW', 'JSTT', 'JTAI', 'JTGEY', 'JTGLF', 'JTKWY',
    'JUNS', 'JUPGF', 'JUSHF', 'JUVF', 'JVA', 'JWEL', 'JWSMF', 'JWSUF', 'JWSWF', 'JXAMY',
    'JXG', 'JXN', 'JXN-PA', 'JYD', 'JYNT', 'JZ', 'JZXN', 'K', 'KACLF', 'KAI',
    'KAIKY', 'KAKKF', 'KALA', 'KALU', 'KALV', 'KANP', 'KAPA', 'KAR', 'KARO', 'KARX',
    'KAVL', 'KAYS', 'KB', 'KBAT', 'KBDC', 'KBGGY', 'KBH', 'KBLB', 'KBR', 'KBSR',
    'KBSX', 'KC', 'KCCFF', 'KCHV', 'KCHVR', 'KCHVU', 'KCLHF', 'KCRD', 'KD', 'KDDIF',
    'KDK', 'KDKGF', 'KDKRW', 'KDOZF', 'KDP', 'KE', 'KELYA', 'KELYB', 'KEN', 'KENS',
    'KEP', 'KEQU', 'KEX', 'KEY', 'KEY-PI', 'KEY-PJ', 'KEY-PK', 'KEY-PL', 'KEYS', 'KF',
    'KFFB', 'KFII', 'KFIIR', 'KFIIU', 'KFRC', 'KFS', 'KFY', 'KG', 'KGC', 'KGCRF',
    'KGEI', 'KGKG', 'KGS', 'KHC', 'KHOB', 'KIDS', 'KIDZ', 'KIDZW', 'KII', 'KIKOF',
    'KIM', 'KIM-PL', 'KIM-PM', 'KIM-PN', 'KINS', 'KIO', 'KIQSF', 'KISB', 'KITL', 'KITT',
    'KITTW', 'KKR', 'KKR-PD', 'KKRS', 'KKRT', 'KKSIY', 'KLAC', 'KLAR', 'KLC', 'KLIC',
    'KLK', 'KLNG', 'KLRS', 'KLTO', 'KLTOW', 'KLTR', 'KLXE', 'KLYG', 'KMB', 'KMDA',
    'KMFG', 'KMI', 'KMPB', 'KMPR', 'KMRK', 'KMT', 'KMTS', 'KMX', 'KMYGY', 'KN',
    'KNDI', 'KNDYF', 'KNF', 'KNGRF', 'KNOP', 'KNOS', 'KNRX', 'KNSA', 'KNSL', 'KNTK',
    'KNX', 'KO', 'KOAN', 'KOD', 'KODK', 'KOF', 'KOKO', 'KOKSF', 'KOLD', 'KOOYF',
    'KOP', 'KOPN', 'KORE', 'KORGW', 'KOS', 'KOSS', 'KOYN', 'KOYNU', 'KOYNW', 'KPEA',
    'KPHMW', 'KPL', 'KPLT', 'KPLTW', 'KPRX', 'KPTI', 'KR', 'KRC', 'KREF', 'KREF-PA',
    'KRFG', 'KRG', 'KRKR', 'KRMD', 'KRMN', 'KRNGF', 'KRNGY', 'KRNT', 'KRNY', 'KRO',
    'KROS', 'KRP', 'KRRO', 'KRSP', 'KRSP-UN', 'KRSP-WT', 'KRT', 'KRTL', 'KRUS', 'KRYS',
    'KRYXF', 'KSCP', 'KSEZ', 'KSPI', 'KSS', 'KT', 'KTB', 'KTCC', 'KTEL', 'KTF',
    'KTH', 'KTN', 'KTOS', 'KTTA', 'KTTAW', 'KUBR', 'KUKE', 'KUKEY', 'KULR', 'KURA',
    'KVAC', 'KVACU', 'KVACW', 'KVHI', 'KVUE', 'KVYO', 'KW', 'KWIK', 'KWM', 'KWMWW',
    'KWR', 'KXHCF', 'KXIAY', 'KXIN', 'KYFGF', 'KYIV', 'KYIVW', 'KYMR', 'KYN', 'KYTFY',
    'KYTX', 'KZIA', 'KZR', 'L', 'LAAI', 'LAAOF', 'LAB', 'LABFF', 'LAC', 'LACHF',
    'LAD', 'LADR', 'LADX', 'LAES', 'LAFA', 'LAFAR', 'LAFAU', 'LAKE', 'LAMR', 'LAND',
    'LANDM', 'LANDO', 'LANDP', 'LANV', 'LANV-WT', 'LAR', 'LARAX', 'LARK', 'LASE', 'LASR',
    'LATA', 'LATAU', 'LATAW', 'LAUR', 'LAW', 'LAWIL', 'LAWR', 'LAZ', 'LAZR', 'LB',
    'LBCMF', 'LBGJ', 'LBIO', 'LBKX', 'LBRA', 'LBRDA', 'LBRDB', 'LBRDK', 'LBRDP', 'LBRJ',
    'LBRT', 'LBRX', 'LBSR', 'LBTYA', 'LBTYB', 'LBTYK', 'LBUY', 'LBZZ', 'LC', 'LCCC',
    'LCCCR', 'LCCCU', 'LCDC', 'LCFY', 'LCFYW', 'LCGMF', 'LCHD', 'LCID', 'LCII', 'LCNB',
    'LCTC', 'LCTX', 'LCUT', 'LDDD', 'LDDFF', 'LDI', 'LDOS', 'LDP', 'LDTCF', 'LDTDF',
    'LDWY', 'LDXC', 'LE', 'LEA', 'LEAT', 'LEBGF', 'LECO', 'LEDS', 'LEE', 'LEEEF',
    'LEEN', 'LEG', 'LEGH', 'LEGN', 'LEGT', 'LEGT-UN', 'LEGT-WT', 'LEN', 'LEN-B', 'LENDX',
    'LENZ', 'LEO', 'LESL', 'LEU', 'LEVI', 'LEXX', 'LEXXW', 'LFCR', 'LFMD', 'LFMDP',
    'LFS', 'LFST', 'LFT', 'LFT-PA', 'LFUS', 'LFVN', 'LFWD', 'LGCB', 'LGCL', 'LGCXF',
    'LGCY', 'LGDTF', 'LGHL', 'LGI', 'LGIH', 'LGL', 'LGL-WT', 'LGMK', 'LGN', 'LGND',
    'LGNDZ', 'LGNXZ', 'LGNYZ', 'LGNZZ', 'LGO', 'LGPS', 'LGVN', 'LH', 'LHAI', 'LHI',
    'LHSW', 'LHX', 'LI', 'LIANY', 'LICN', 'LIDR', 'LIDRW', 'LIEN', 'LIF', 'LIFD',
    'LIFE', 'LIFX', 'LII', 'LILA', 'LILAB', 'LILAK', 'LILIF', 'LILMF', 'LIMN', 'LIMNW',
    'LIMX', 'LIN', 'LINC', 'LIND', 'LINE', 'LINK', 'LINMF', 'LION', 'LIPO', 'LIQT',
    'LISMF', 'LITB', 'LITE', 'LITM', 'LITS', 'LITSF', 'LIVE', 'LIVN', 'LIXT', 'LIXTW',
    'LKFN', 'LKNCY', 'LKQ', 'LKSP', 'LKSPR', 'LKSPU', 'LLDTF', 'LLOBF', 'LLY', 'LLYVA',
    'LLYVB', 'LLYVK', 'LMAT', 'LMB', 'LMFA', 'LMMY', 'LMND', 'LMND-WT', 'LMNR', 'LMRI',
    'LMRMF', 'LMRXF', 'LMT', 'LNAI', 'LNBY', 'LNC', 'LNC-PD', 'LND', 'LNG', 'LNKB',
    'LNKS', 'LNN', 'LNSR', 'LNT', 'LNTH', 'LNTO', 'LNWO', 'LNXSF', 'LNXSY', 'LNZA',
    'LNZAW', 'LOAN', 'LOAR', 'LOB', 'LOB-PA', 'LOBO', 'LOCL', 'LOCLW', 'LOCO', 'LODE',
    'LOGC', 'LOGI', 'LOKV', 'LOKVU', 'LOKVW', 'LOMA', 'LOMLF', 'LOMWF', 'LONCF', 'LOOP',
    'LOPE', 'LOT', 'LOTMY', 'LOTWW', 'LOVE', 'LOW', 'LPA', 'LPAA', 'LPAAU', 'LPAAW',
    'LPBB', 'LPBBU', 'LPBBW', 'LPCHY', 'LPCN', 'LPG', 'LPGCY', 'LPL', 'LPLA', 'LPRO',
    'LPSIF', 'LPSN', 'LPTH', 'LPTVQ', 'LPUGF', 'LPX', 'LQDA', 'LQDT', 'LQMT', 'LQWC',
    'LRCX', 'LRDC', 'LRE', 'LRHC', 'LRMR', 'LRN', 'LRRIF', 'LRVIY', 'LSAK', 'LSBCF',
    'LSBK', 'LSBWF', 'LSCC', 'LSE', 'LSEB', 'LSEGY', 'LSF', 'LSH', 'LSHGF', 'LSHGY',
    'LSPD', 'LSRCF', 'LSTA', 'LSTR', 'LTAFX', 'LTBR', 'LTC', 'LTCFX', 'LTCN', 'LTH',
    'LTM', 'LTMGF', 'LTRN', 'LTRX', 'LTRYW', 'LTSV', 'LTUM', 'LU', 'LUCD', 'LUCK',
    'LUCN', 'LUCY', 'LUCYW', 'LUD', 'LUDG', 'LULU', 'LUMN', 'LUNG', 'LUNR', 'LUV',
    'LUVU', 'LUXE', 'LVCE', 'LVLU', 'LVO', 'LVPA', 'LVRLF', 'LVRO', 'LVROW', 'LVS',
    'LVTX', 'LVWR', 'LVWR-WT', 'LW', 'LWAC', 'LWACU', 'LWACW', 'LWAY', 'LWLG', 'LWSCF',
    'LX', 'LXEH', 'LXEO', 'LXFR', 'LXP', 'LXP-PC', 'LXRX', 'LXU', 'LYB', 'LYEL',
    'LYFT', 'LYG', 'LYRA', 'LYTHF', 'LYTS', 'LYV', 'LZ', 'LZB', 'LZM', 'LZM-WT',
    'LZMH', 'M', 'MA', 'MAA', 'MAA-PI', 'MAAS', 'MAC', 'MACI', 'MACIU', 'MACIW',
    'MACT', 'MADL', 'MAGE', 'MAGH', 'MAGN', 'MAIA', 'MAIN', 'MAJI', 'MALG', 'MAMA',
    'MAMK', 'MAMO', 'MAN', 'MANA', 'MANDF', 'MANH', 'MANU', 'MAPPF', 'MAPS', 'MAPSW',
    'MAR', 'MARA', 'MARPS', 'MAS', 'MASI', 'MASK', 'MASN', 'MASS', 'MAT', 'MATH',
    'MATV', 'MATW', 'MATX', 'MAUTF', 'MAWAF', 'MAX', 'MAXN', 'MAYA', 'MAYAR', 'MAYAU',
    'MAYS', 'MAZE', 'MB', 'MBAV', 'MBAVU', 'MBAVW', 'MBBC', 'MBC', 'MBCN', 'MBFJF',
    'MBGAF', 'MBGCF', 'MBI', 'MBIN', 'MBINL', 'MBINM', 'MBINN', 'MBIO', 'MBLY', 'MBNKO',
    'MBOT', 'MBRFY', 'MBRX', 'MBSHY', 'MBUMF', 'MBUU', 'MBVI', 'MBVIU', 'MBVIW', 'MBWM',
    'MBX', 'MC', 'MCAG', 'MCAGR', 'MCAGU', 'MCB', 'MCBS', 'MCD', 'MCFT', 'MCGA',
    'MCGAU', 'MCGAW', 'MCHB', 'MCHP', 'MCHPP', 'MCHX', 'MCI', 'MCK', 'MCLE', 'MCN',
    'MCO', 'MCOM', 'MCOMW', 'MCR', 'MCRB', 'MCRI', 'MCRP', 'MCS', 'MCTA', 'MCTR',
    'MCUJF', 'MCW', 'MCY', 'MD', 'MDAI', 'MDAIW', 'MDB', 'MDBH', 'MDCE', 'MDCOY',
    'MDCX', 'MDCXW', 'MDDNF', 'MDDTY', 'MDGL', 'MDIA', 'MDLN', 'MDLZ', 'MDNAF', 'MDNC',
    'MDRR', 'MDT', 'MDU', 'MDV', 'MDV-PA', 'MDWD', 'MDWK', 'MDXG', 'MDXH', 'MDY',
    'MEC', 'MECPF', 'MED', 'MEDG', 'MEDP', 'MEG', 'MEGI', 'MEGL', 'MEHA', 'MEHCQ',
    'MEI', 'MEIUF', 'MELI', 'MENS', 'MEOBF', 'MEOH', 'MER-PK', 'MERC', 'MESA', 'MESO',
    'MET', 'MET-PA', 'MET-PE', 'MET-PF', 'META', 'METC', 'METCB', 'METCI', 'METCZ', 'METRY',
    'MFA', 'MFA-PB', 'MFA-PC', 'MFAN', 'MFAO', 'MFBI', 'MFC', 'MFG', 'MFGCF', 'MFHK',
    'MFI', 'MFIC', 'MFICL', 'MFIN', 'MFM', 'MFON', 'MG', 'MGA', 'MGAM', 'MGCLY',
    'MGCOF', 'MGEE', 'MGF', 'MGHL', 'MGHTF', 'MGIC', 'MGIH', 'MGLD', 'MGM', 'MGMNF',
    'MGN', 'MGNC', 'MGNI', 'MGNO', 'MGNX', 'MGPI', 'MGR', 'MGRB', 'MGRC', 'MGRD',
    'MGRE', 'MGRT', 'MGRX', 'MGSD', 'MGTE', 'MGTEW', 'MGTX', 'MGX', 'MGY', 'MGYR',
    'MH', 'MHD', 'MHF', 'MHH', 'MHK', 'MHLA', 'MHN', 'MHNC', 'MHO', 'MHUA',
    'MHUBF', 'MI', 'MIAX', 'MIBE', 'MICC', 'MICLF', 'MIDD', 'MIGI', 'MILIF', 'MIMDF',
    'MIMI', 'MIMTF', 'MIN', 'MINBY', 'MIND', 'MINR', 'MIR', 'MIRA', 'MIRM', 'MIST',
    'MITI', 'MITK', 'MITN', 'MITP', 'MITQ', 'MITT', 'MITT-PA', 'MITT-PB', 'MITT-PC', 'MIY',
    'MKC', 'MKC-V', 'MKDW', 'MKDWW', 'MKL', 'MKLY', 'MKLYR', 'MKLYU', 'MKSI', 'MKTR',
    'MKTW', 'MKTX', 'MKZR', 'MLAB', 'MLAC', 'MLACR', 'MLACU', 'MLCI', 'MLCMF', 'MLCO',
    'MLEC', 'MLECW', 'MLGO', 'MLI', 'MLIZY', 'MLKN', 'MLM', 'MLMC', 'MLP', 'MLPB',
    'MLPR', 'MLR', 'MLRT', 'MLSPF', 'MLSS', 'MLTX', 'MLYS', 'MMA', 'MMC', 'MMCP',
    'MMD', 'MMEX', 'MMI', 'MMLP', 'MMM', 'MMMW', 'MMS', 'MMSI', 'MMT', 'MMTIF',
    'MMTRS', 'MMTX', 'MMTXU', 'MMU', 'MMYT', 'MNBEF', 'MNDO', 'MNDR', 'MNDY', 'MNESP',
    'MNFYY', 'MNKD', 'MNLCF', 'MNMD', 'MNOV', 'MNPR', 'MNQFF', 'MNR', 'MNRO', 'MNSB',
    'MNSBP', 'MNSKY', 'MNSO', 'MNST', 'MNTK', 'MNTN', 'MNTR', 'MNTS', 'MNTSW', 'MNUFF',
    'MNY', 'MNYFF', 'MNYWW', 'MNZLY', 'MO', 'MOB', 'MOBBW', 'MOBQ', 'MOBQW', 'MOBX',
    'MOBXW', 'MOD', 'MODD', 'MODG', 'MODVQ', 'MOFG', 'MOG-A', 'MOG-B', 'MOGMF', 'MOGO',
    'MOGU', 'MOH', 'MOJO', 'MOLN', 'MOMO', 'MONRF', 'MORN', 'MOS', 'MOV', 'MOVAA',
    'MOVE', 'MP', 'MPA', 'MPAA', 'MPB', 'MPC', 'MPJS', 'MPLT', 'MPLX', 'MPLXP',
    'MPTI', 'MPTI-WT', 'MPU', 'MPV', 'MPVDF', 'MPW', 'MPWR', 'MPX', 'MQ', 'MQT',
    'MQY', 'MRAI', 'MRAM', 'MRBK', 'MRCA', 'MRCC', 'MRCIF', 'MRCY', 'MREGY', 'MREO',
    'MRK', 'MRKR', 'MRM', 'MRMD', 'MRNA', 'MRNO', 'MRNOW', 'MROSY', 'MRP', 'MRP-WI',
    'MRPT', 'MRRDF', 'MRSN', 'MRT', 'MRTN', 'MRUS', 'MRUWY', 'MRVI', 'MRVL', 'MRX',
    'MS', 'MS-PA', 'MS-PE', 'MS-PF', 'MS-PI', 'MS-PK', 'MS-PL', 'MS-PO', 'MS-PP', 'MS-PQ',
    'MSA', 'MSAI', 'MSAIW', 'MSB', 'MSBB', 'MSBI', 'MSBIP', 'MSC', 'MSCI', 'MSCLF',
    'MSD', 'MSDL', 'MSEX', 'MSEXP', 'MSFT', 'MSGE', 'MSGM', 'MSGS', 'MSGY', 'MSI',
    'MSIF', 'MSM', 'MSMU', 'MSN', 'MSOGF', 'MSOKF', 'MSPR', 'MSPRW', 'MSPRZ', 'MSS',
    'MSSAF', 'MSSHY', 'MSSRF', 'MSSUF', 'MSSWF', 'MSTH', 'MSTKY', 'MSTLW', 'MSTR', 'MSUXF',
    'MSW', 'MT', 'MTA', 'MTB', 'MTB-PH', 'MTB-PJ', 'MTB-PK', 'MTBLY', 'MTC', 'MTCH',
    'MTD', 'MTDR', 'MTEK', 'MTEKW', 'MTEN', 'MTEX', 'MTG', 'MTH', 'MTLK', 'MTLPF',
    'MTLS', 'MTMTY', 'MTMV', 'MTN', 'MTNB', 'MTR', 'MTRN', 'MTRX', 'MTSI', 'MTTCF',
    'MTUL', 'MTUS', 'MTVA', 'MTW', 'MTWO', 'MTX', 'MTZ', 'MU', 'MUA', 'MUC',
    'MUE', 'MUFG', 'MUJ', 'MUR', 'MURA', 'MUSA', 'MUX', 'MVBF', 'MVCO', 'MVF',
    'MVIS', 'MVNC', 'MVO', 'MVRL', 'MVST', 'MVSTW', 'MVT', 'MWA', 'MWAI', 'MWG',
    'MWYN', 'MX', 'MXC', 'MXCT', 'MXE', 'MXF', 'MXL', 'MXUBY', 'MXUGF', 'MYCB',
    'MYCRY', 'MYD', 'MYE', 'MYFW', 'MYGN', 'MYI', 'MYN', 'MYND', 'MYNZ', 'MYO',
    'MYPS', 'MYPSW', 'MYRG', 'MYSE', 'MYSEW', 'MYSZ', 'MZHOF', 'MZTI', 'NA', 'NAAS',
    'NABL', 'NAC', 'NAD', 'NAGE', 'NAII', 'NAK', 'NAKA', 'NAKAW', 'NAMI', 'NAMM',
    'NAMMW', 'NAMS', 'NAMSW', 'NAN', 'NAOV', 'NASC', 'NAT', 'NATH', 'NATL', 'NATR',
    'NAUT', 'NAVI', 'NAVN', 'NAZ', 'NB', 'NBB', 'NBBI', 'NBBK', 'NBH', 'NBHC',
    'NBIS', 'NBIX', 'NBKBY', 'NBLWF', 'NBND', 'NBP', 'NBR', 'NBRG', 'NBRWF', 'NBTB',
    'NBTX', 'NBXG', 'NBY', 'NC', 'NCA', 'NCAUF', 'NCDL', 'NCEL', 'NCEW', 'NCHEF',
    'NCI', 'NCIQ', 'NCL', 'NCLH', 'NCLTF', 'NCLTY', 'NCMI', 'NCNA', 'NCNCF', 'NCNO',
    'NCPL', 'NCPLW', 'NCRA', 'NCRRP', 'NCSM', 'NCSYF', 'NCT', 'NCTY', 'NCV', 'NCV-PA',
    'NCZ', 'NCZ-PA', 'NDAQ', 'NDEKF', 'NDLS', 'NDMO', 'NDRA', 'NDSN', 'NE', 'NE-WT',
    'NE-WTA', 'NEA', 'NECB', 'NEE', 'NEE-PN', 'NEE-PS', 'NEE-PT', 'NEE-PU', 'NEGG', 'NELR',
    'NEM', 'NEMCL', 'NEN', 'NEO', 'NEOG', 'NEON', 'NEOV', 'NEOVW', 'NEPH', 'NERV',
    'NESR', 'NET', 'NETD', 'NETDU', 'NETDW', 'NETTF', 'NEU', 'NEUE', 'NEUP', 'NEWEN',
    'NEWH', 'NEWP', 'NEWT', 'NEWTG', 'NEWTH', 'NEWTI', 'NEWTP', 'NEWTZ', 'NEXA', 'NEXCF',
    'NEXHY', 'NEXM', 'NEXN', 'NEXT', 'NFBK', 'NFE', 'NFG', 'NFGC', 'NFJ', 'NFLX',
    'NFSN', 'NFTM', 'NFTN', 'NG', 'NGCG', 'NGCOY', 'NGD', 'NGG', 'NGGTF', 'NGHI',
    'NGHLF', 'NGL', 'NGL-PB', 'NGL-PC', 'NGLD', 'NGNE', 'NGS', 'NGSCF', 'NGTF', 'NGVC',
    'NGVT', 'NGXXF', 'NHC', 'NHFLF', 'NHI', 'NHIC', 'NHICU', 'NHICW', 'NHKGF', 'NHKSY',
    'NHLTY', 'NHNKF', 'NHPAP', 'NHPBP', 'NHS', 'NHTC', 'NI', 'NIC', 'NICE', 'NICHX',
    'NIE', 'NIHK', 'NIKA', 'NIM', 'NIMU', 'NINE', 'NIO', 'NIOBW', 'NIOIF', 'NIOMF',
    'NIPG', 'NIQ', 'NISN', 'NITO', 'NIU', 'NIVF', 'NIVFW', 'NIXX', 'NIXXW', 'NJDCY',
    'NJR', 'NKE', 'NKGFF', 'NKLAQ', 'NKLR', 'NKSH', 'NKTR', 'NKTX', 'NKX', 'NL',
    'NLCP', 'NLOP', 'NLST', 'NLY', 'NLY-PF', 'NLY-PG', 'NLY-PI', 'NLY-PJ', 'NMAI', 'NMAX',
    'NMCO', 'NMEX', 'NMFC', 'NMFCZ', 'NMG', 'NMGX', 'NMHI', 'NMHIW', 'NMI', 'NMIH',
    'NMKBP', 'NMKCP', 'NML', 'NMM', 'NMP', 'NMPAR', 'NMPAU', 'NMPGY', 'NMPRY', 'NMPWP',
    'NMR', 'NMRA', 'NMREF', 'NMRK', 'NMS', 'NMT', 'NMTC', 'NMZ', 'NN', 'NNAVW',
    'NNAX', 'NNBR', 'NNDM', 'NNE', 'NNGPF', 'NNI', 'NNN', 'NNNN', 'NNOX', 'NNUP',
    'NNVC', 'NNWWF', 'NNY', 'NOA', 'NOAH', 'NOBH', 'NOC', 'NODK', 'NOEM', 'NOEMR',
    'NOEMU', 'NOEMW', 'NOG', 'NOK', 'NOKBF', 'NOM', 'NOMA', 'NOMD', 'NONOF', 'NORD',
    'NOTE', 'NOTE-WT', 'NOTV', 'NOV', 'NOVAQ', 'NOVT', 'NOVTU', 'NOW', 'NOWG', 'NP',
    'NPAC', 'NPACU', 'NPACW', 'NPB', 'NPCE', 'NPCT', 'NPEHF', 'NPFD', 'NPICF', 'NPIFF',
    'NPIXY', 'NPK', 'NPKI', 'NPO', 'NPPXF', 'NPRFF', 'NPT', 'NPV', 'NPWR', 'NPWR-WT',
    'NPXYY', 'NQP', 'NRC', 'NRDE', 'NRDS', 'NRDY', 'NREF', 'NREF-PA', 'NRG', 'NRGD',
    'NRGU', 'NRGV', 'NRHI', 'NRIM', 'NRIS', 'NRIX', 'NRK', 'NRO', 'NROM', 'NRP',
    'NRRSF', 'NRRWF', 'NRSAX', 'NRSCF', 'NRSN', 'NRSNW', 'NRT', 'NRUC', 'NRXP', 'NRXPW',
    'NRXS', 'NRYCF', 'NSA', 'NSA-PA', 'NSA-PB', 'NSARO', 'NSARP', 'NSC', 'NSFDF', 'NSIT',
    'NSKFF', 'NSNFY', 'NSP', 'NSPR', 'NSRCF', 'NSRX', 'NSSC', 'NSTM', 'NSTS', 'NSYS',
    'NTAP', 'NTB', 'NTCL', 'NTCS', 'NTCT', 'NTES', 'NTGR', 'NTHI', 'NTIC', 'NTIP',
    'NTLA', 'NTNX', 'NTPIF', 'NTR', 'NTRA', 'NTRB', 'NTRBW', 'NTRP', 'NTRR', 'NTRS',
    'NTRSO', 'NTRX', 'NTSK', 'NTST', 'NTTYY', 'NTWK', 'NTWO', 'NTWOU', 'NTWOW', 'NTZ',
    'NU', 'NUAI', 'NUAIW', 'NUE', 'NUGN', 'NUKK', 'NUKKW', 'NULGF', 'NUMD', 'NUS',
    'NUTR', 'NUTX', 'NUV', 'NUVB', 'NUVB-WT', 'NUVI', 'NUVL', 'NUVR', 'NUW', 'NUWE',
    'NVA', 'NVAAF', 'NVACW', 'NVAWW', 'NVAX', 'NVCR', 'NVCT', 'NVDA', 'NVEC', 'NVG',
    'NVGLF', 'NVGS', 'NVMI', 'NVNBW', 'NVNI', 'NVNIW', 'NVNO', 'NVNXF', 'NVO', 'NVR',
    'NVRI', 'NVS', 'NVSEF', 'NVSGF', 'NVST', 'NVT', 'NVTS', 'NVVE', 'NVVEW', 'NVX',
    'NVZMF', 'NWBI', 'NWBO', 'NWCYY', 'NWE', 'NWFL', 'NWG', 'NWGL', 'NWL', 'NWN',
    'NWOEF', 'NWPP', 'NWPX', 'NWS', 'NWSA', 'NWSGY', 'NWSZF', 'NWTG', 'NX', 'NXC',
    'NXDR', 'NXDT', 'NXDT-PA', 'NXE', 'NXFNF', 'NXFTY', 'NXG', 'NXGL', 'NXGLW', 'NXJ',
    'NXL', 'NXMH', 'NXN', 'NXNT', 'NXNVW', 'NXP', 'NXPGF', 'NXPGY', 'NXPI', 'NXPL',
    'NXPLW', 'NXPRF', 'NXRT', 'NXST', 'NXT', 'NXTC', 'NXTT', 'NXUR', 'NXXT', 'NYAE',
    'NYAX', 'NYC', 'NYMXF', 'NYT', 'NYXH', 'NZAUF', 'NZEOF', 'NZEOY', 'NZERF', 'NZF',
    'O', 'OABI', 'OABIW', 'OACC', 'OACCU', 'OACCW', 'OAK-PA', 'OAK-PB', 'OAKU', 'OAKUR',
    'OAKUU', 'OAKUW', 'OAKV', 'OBA', 'OBAWU', 'OBAWW', 'OBBCY', 'OBCKF', 'OBDC', 'OBE',
    'OBIBF', 'OBICY', 'OBIIF', 'OBIO', 'OBK', 'OBLG', 'OBNB', 'OBNK', 'OBOCY', 'OBT',
    'OBTC', 'OC', 'OCC', 'OCCI', 'OCCIM', 'OCCIN', 'OCCIO', 'OCEA', 'OCFC', 'OCG',
    'OCGN', 'OCGSF', 'OCLN', 'OCS', 'OCSAW', 'OCSL', 'OCUL', 'ODC', 'ODD', 'ODDAF',
    'ODFL', 'ODOT', 'ODP', 'ODRS', 'ODV', 'ODVWZ', 'ODYS', 'ODYY', 'OEC', 'OESX',
    'OFAL', 'OFED', 'OFG', 'OFIX', 'OFLX', 'OFRM', 'OFS', 'OFSSH', 'OFSSO', 'OGCP',
    'OGE', 'OGEN', 'OGI', 'OGN', 'OGS', 'OHCFF', 'OHI', 'OI', 'OIA', 'OII',
    'OILCF', 'OILD', 'OILSF', 'OILU', 'OIS', 'OKE', 'OKLO', 'OKMN', 'OKTA', 'OKUR',
    'OKYO', 'OLB', 'OLED', 'OLLI', 'OLMA', 'OLN', 'OLOXF', 'OLP', 'OLPX', 'OM',
    'OMAB', 'OMC', 'OMCC', 'OMCL', 'OMDA', 'OMER', 'OMEX', 'OMF', 'OMH', 'OMI',
    'OMNI', 'OMQS', 'OMSE', 'OMTK', 'OMVJF', 'OMVKY', 'ON', 'ONAR', 'ONB', 'ONBPO',
    'ONBPP', 'ONC', 'ONCH', 'ONCHU', 'ONCHW', 'ONCO', 'ONCY', 'ONDS', 'ONEG', 'ONEI',
    'ONEW', 'ONFO', 'ONFOP', 'ONFOW', 'ONIT', 'ONL', 'ONMD', 'ONMDW', 'ONON', 'ONSS',
    'ONTF', 'ONTO', 'ONWRF', 'ONWRY', 'OOMA', 'OPAD', 'OPADW', 'OPAL', 'OPBK', 'OPCH',
    'OPEN', 'OPENL', 'OPENW', 'OPENZ', 'OPFI', 'OPFI-WT', 'OPGN', 'OPHC', 'OPIRQ', 'OPITQ',
    'OPK', 'OPP', 'OPP-PA', 'OPP-PB', 'OPP-PC', 'OPRA', 'OPRT', 'OPRX', 'OPTEY', 'OPTH',
    'OPTHF', 'OPTT', 'OPTU', 'OPTX', 'OPTXW', 'OPXS', 'OPY', 'OR', 'ORA', 'ORANY',
    'ORBS', 'ORC', 'ORCL', 'ORESF', 'ORGN', 'ORGNW', 'ORGO', 'ORGS', 'ORI', 'ORIB',
    'ORIC', 'ORIQ', 'ORIQU', 'ORIQW', 'ORIS', 'ORKA', 'ORKT', 'ORLA', 'ORLY', 'ORMNF',
    'ORMP', 'ORN', 'ORRCF', 'ORRF', 'ORXCF', 'OS', 'OSBC', 'OSBK', 'OSCR', 'OSCUF',
    'OSG', 'OSIS', 'OSK', 'OSOL', 'OSPN', 'OSRH', 'OSRHW', 'OSS', 'OSSFF', 'OSSUY',
    'OST', 'OSTX', 'OSUR', 'OSW', 'OTEX', 'OTF', 'OTGA', 'OTGAU', 'OTGAW', 'OTH',
    'OTIS', 'OTLC', 'OTLK', 'OTLY', 'OTRKQ', 'OTSA', 'OTTR', 'OUNZ', 'OUST', 'OUSTZ',
    'OUT', 'OUTFF', 'OUTKY', 'OVATF', 'OVBC', 'OVID', 'OVLY', 'OVTZ', 'OVV', 'OWL',
    'OWLS', 'OWLT', 'OWLTW', 'OWPC', 'OWSCX', 'OXBR', 'OXBRW', 'OXLC', 'OXLCG', 'OXLCI',
    'OXLCL', 'OXLCN', 'OXLCO', 'OXLCP', 'OXLCZ', 'OXM', 'OXSQ', 'OXSQG', 'OXSQH', 'OXY',
    'OXY-WT', 'OYCG', 'OYSE', 'OYSER', 'OYSEU', 'OZ', 'OZK', 'OZKAP', 'OZSC', 'PAA',
    'PAANF', 'PAAPU', 'PAAS', 'PAASF', 'PAC', 'PACB', 'PACH', 'PACHU', 'PACHW', 'PACK',
    'PACS', 'PADEF', 'PAEXY', 'PAG', 'PAGP', 'PAGS', 'PAHC', 'PAI', 'PAII', 'PAII-UN',
    'PAII-WT', 'PAIYY', 'PAL', 'PALI', 'PALL', 'PAM', 'PAMT', 'PANL', 'PANW', 'PAPA',
    'PAPL', 'PAR', 'PARK', 'PARR', 'PARXF', 'PASG', 'PASTF', 'PASTY', 'PASW', 'PATH',
    'PATK', 'PAVM', 'PAVS', 'PAX', 'PAXH', 'PAXS', 'PAY', 'PAYC', 'PAYD', 'PAYO',
    'PAYS', 'PAYX', 'PB', 'PBA', 'PBBK', 'PBF', 'PBFS', 'PBH', 'PBHC', 'PBI',
    'PBI-PB', 'PBM', 'PBMLF', 'PBMWW', 'PBNAF', 'PBNNF', 'PBPRF', 'PBR', 'PBR-A', 'PBRRY',
    'PBSV', 'PBT', 'PBYI', 'PC', 'PCAP', 'PCAPU', 'PCAPW', 'PCAR', 'PCB', 'PCCOF',
    'PCCYF', 'PCF', 'PCG', 'PCG-PA', 'PCG-PB', 'PCG-PC', 'PCG-PD', 'PCG-PE', 'PCG-PG', 'PCG-PH',
    'PCG-PI', 'PCG-PX', 'PCH', 'PCLA', 'PCM', 'PCMC', 'PCN', 'PCOK', 'PCOR', 'PCPGY',
    'PCPPF', 'PCQ', 'PCRX', 'PCSA', 'PCSC', 'PCSV', 'PCT', 'PCTBP', 'PCTTU', 'PCTTW',
    'PCTY', 'PCVX', 'PCYO', 'PD', 'PDCC', 'PDD', 'PDEX', 'PDFS', 'PDI', 'PDLB',
    'PDM', 'PDO', 'PDPA', 'PDS', 'PDSB', 'PDSKX', 'PDSRX', 'PDT', 'PDX', 'PDYN',
    'PDYNW', 'PDYTY', 'PEB', 'PEB-PE', 'PEB-PF', 'PEB-PG', 'PEB-PH', 'PEBK', 'PEBO', 'PECO',
    'PED', 'PEG', 'PEGA', 'PEGRF', 'PELI', 'PELIR', 'PELIU', 'PEN', 'PENG', 'PENN',
    'PEO', 'PEP', 'PEPG', 'PERF', 'PERF-WT', 'PERI', 'PESI', 'PETS', 'PETV', 'PETVW',
    'PETZ', 'PEVM', 'PEW', 'PEW-WT', 'PEXZF', 'PFAI', 'PFBC', 'PFBX', 'PFD', 'PFE',
    'PFFL', 'PFFLX', 'PFG', 'PFGC', 'PFH', 'PFHO', 'PFIS', 'PFL', 'PFLT', 'PFN',
    'PFO', 'PFS', 'PFSA', 'PFSB', 'PFSI', 'PFX', 'PFXNZ', 'PG', 'PGAC', 'PGACR',
    'PGACU', 'PGC', 'PGEN', 'PGIM', 'PGNY', 'PGOL', 'PGP', 'PGR', 'PGRE', 'PGY',
    'PGYWW', 'PGZ', 'PGZFF', 'PH', 'PHAR', 'PHAT', 'PHBI', 'PHCI', 'PHD', 'PHDWY',
    'PHG', 'PHGE', 'PHI', 'PHIL', 'PHIN', 'PHIO', 'PHK', 'PHM', 'PHNMF', 'PHOE',
    'PHR', 'PHTCF', 'PHUN', 'PHVS', 'PHXE-P', 'PHYS', 'PI', 'PIAC', 'PII', 'PIII',
    'PIIIW', 'PIM', 'PINC', 'PINE', 'PINE-PA', 'PINS', 'PINWF', 'PIPR', 'PITEF', 'PJT',
    'PK', 'PKBK', 'PKE', 'PKG', 'PKIUY', 'PKKFF', 'PKOH', 'PKST', 'PKTX', 'PKX',
    'PL', 'PL-WT', 'PLAB', 'PLAG', 'PLAI', 'PLAY', 'PLBC', 'PLBL', 'PLBY', 'PLCE',
    'PLCKF', 'PLD', 'PLDGP', 'PLG', 'PLMJF', 'PLMK', 'PLMKU', 'PLMKW', 'PLMR', 'PLMUF',
    'PLMWF', 'PLNH', 'PLNT', 'PLOW', 'PLPC', 'PLPL', 'PLRX', 'PLRZ', 'PLSAY', 'PLSE',
    'PLSH', 'PLTK', 'PLTM', 'PLTR', 'PLTS', 'PLTYF', 'PLUG', 'PLUR', 'PLUS', 'PLUT',
    'PLX', 'PLXS', 'PLYM', 'PLYX', 'PM', 'PMAX', 'PMBPF', 'PMCB', 'PMCUF', 'PMDI',
    'PMDIY', 'PMEC', 'PMEDF', 'PMFAX', 'PMHG', 'PMHMY', 'PMHS', 'PMI', 'PML', 'PMM',
    'PMMBF', 'PMMCF', 'PMN', 'PMNT', 'PMO', 'PMT', 'PMT-PA', 'PMT-PB', 'PMT-PC', 'PMTR',
    'PMTRU', 'PMTRW', 'PMTS', 'PMTU', 'PMTV', 'PMTW', 'PMVC', 'PMVCW', 'PMVP', 'PN',
    'PNBK', 'PNC', 'PNDRY', 'PNDZF', 'PNFP', 'PNFPP', 'PNI', 'PNMXO', 'PNNT', 'PNPL',
    'PNPNF', 'PNR', 'PNRG', 'PNTG', 'PNW', 'PNXP', 'PNYG', 'POAHF', 'POAI', 'POAS',
    'POCI', 'PODC', 'PODD', 'POET', 'POLA', 'POLE', 'POLEU', 'POLEW', 'POM', 'PONY',
    'POOL', 'POR', 'POSC', 'POST', 'POWI', 'POWL', 'POWMF', 'POWW', 'POWWP', 'PPBT',
    'PPC', 'PPCB', 'PPENF', 'PPG', 'PPIH', 'PPL', 'PPLAF', 'PPLOF', 'PPLT', 'PPRUF',
    'PPSI', 'PPT', 'PPTA', 'PR', 'PRA', 'PRAA', 'PRAX', 'PRCH', 'PRCT', 'PRDIY',
    'PRDO', 'PRE', 'PREJF', 'PREKF', 'PREM', 'PRENW', 'PRFX', 'PRG', 'PRGO', 'PRGS',
    'PRGY', 'PRH', 'PRHI', 'PRHIZ', 'PRI', 'PRIAF', 'PRIF-PD', 'PRIF-PJ', 'PRIF-PK', 'PRIF-PL',
    'PRIM', 'PRK', 'PRKA', 'PRKR', 'PRKS', 'PRLB', 'PRLD', 'PRM', 'PRMB', 'PRME',
    'PRMLF', 'PRNAF', 'PRNCF', 'PRO', 'PROF', 'PROK', 'PROP', 'PROV', 'PRPH', 'PRPL',
    'PRPO', 'PRQR', 'PRRUF', 'PRS', 'PRSI', 'PRSKY', 'PRSO', 'PRSU', 'PRT', 'PRTA',
    'PRTC', 'PRTH', 'PRTHU', 'PRTS', 'PRTX', 'PRU', 'PRVA', 'PRXA', 'PRXK', 'PRXXF',
    'PRZO', 'PSA', 'PSA-PF', 'PSA-PG', 'PSA-PH', 'PSA-PI', 'PSA-PJ', 'PSA-PK', 'PSA-PL', 'PSA-PM',
    'PSA-PN', 'PSA-PO', 'PSA-PP', 'PSA-PQ', 'PSA-PR', 'PSA-PS', 'PSBAF', 'PSBD', 'PSBTY', 'PSEC',
    'PSEC-PA', 'PSEWF', 'PSF', 'PSFE', 'PSHG', 'PSIG', 'PSIX', 'PSKRF', 'PSKY', 'PSLV',
    'PSMT', 'PSN', 'PSNL', 'PSNY', 'PSNYW', 'PSO', 'PSORF', 'PSPX', 'PSQH', 'PSQH-WT',
    'PSRHF', 'PSTG', 'PSTL', 'PSTV', 'PSX', 'PSYCF', 'PT', 'PTA', 'PTAXY', 'PTC',
    'PTCAY', 'PTCCY', 'PTCHF', 'PTCO', 'PTCT', 'PTEN', 'PTGX', 'PTHL', 'PTHRF', 'PTHS',
    'PTIX', 'PTIXW', 'PTLE', 'PTLO', 'PTN', 'PTNM', 'PTNRF', 'PTNT', 'PTON', 'PTOP',
    'PTOR', 'PTOS', 'PTPI', 'PTPIF', 'PTRN', 'PTXAF', 'PTXKY', 'PTY', 'PUBC', 'PUBM',
    'PUCCF', 'PUGBY', 'PUIGF', 'PUK', 'PUKPF', 'PULM', 'PUMP', 'PURE', 'PVCT', 'PVH',
    'PVL', 'PVLA', 'PVOZ', 'PW', 'PW-PA', 'PWDY', 'PWP', 'PWR', 'PWRL', 'PX',
    'PXED', 'PXLW', 'PXS', 'PYPD', 'PYPL', 'PYRGF', 'PYT', 'PYXS', 'PYYX', 'PZG',
    'PZZA', 'Q', 'QADR', 'QBTS', 'QCLS', 'QCOM', 'QCRH', 'QD', 'QDEL', 'QDMI',
    'QETA', 'QETAR', 'QETAU', 'QETH', 'QF', 'QFIN', 'QFNHF', 'QGEN', 'QH', 'QIND',
    'QIPT', 'QLUNF', 'QLYS', 'QMCI', 'QMCO', 'QMMM', 'QNBC', 'QNCX', 'QNRX', 'QNST',
    'QNTM', 'QNTO', 'QPRC', 'QQQ', 'QQQX', 'QRED', 'QRHC', 'QRVO', 'QS', 'QSEA',
    'QSEAR', 'QSEAU', 'QSEP', 'QSI', 'QSIAW', 'QSJC', 'QSOL', 'QSR', 'QTIH', 'QTIWW',
    'QTRX', 'QTTB', 'QTTOY', 'QTWO', 'QTZM', 'QUAD', 'QUBT', 'QUIK', 'QULL', 'QUMS',
    'QUMSR', 'QUMSU', 'QURE', 'QURT', 'QVCC', 'QVCD', 'QVCGA', 'QVCGB', 'QVCGP', 'QWTR',
    'QXO', 'QXO-PB', 'QYOUF', 'QZMRF', 'R', 'RA', 'RAAQ', 'RAAQU', 'RAAQW', 'RAASY',
    'RAC', 'RAC-UN', 'RAC-WT', 'RACE', 'RADX', 'RAIL', 'RAIN', 'RAINW', 'RAJAF', 'RAKR',
    'RAL', 'RAMP', 'RAND', 'RANG', 'RANGR', 'RANGU', 'RANI', 'RAPH', 'RAPP', 'RAPT',
    'RARE', 'RAVE', 'RAY', 'RAYA', 'RBA', 'RBB', 'RBBN', 'RBC', 'RBCAA', 'RBCN',
    'RBKB', 'RBLX', 'RBNE', 'RBOT', 'RBOT-WT', 'RBRK', 'RBSPF', 'RBTK', 'RC', 'RC-PC',
    'RC-PE', 'RCAT', 'RCB', 'RCC', 'RCD', 'RCEL', 'RCG', 'RCI', 'RCIAF', 'RCKT',
    'RCKTW', 'RCKY', 'RCL', 'RCMT', 'RCON', 'RCS', 'RCT', 'RCUS', 'RCWBY', 'RCWLY',
    'RDAC', 'RDACR', 'RDACU', 'RDAG', 'RDAGU', 'RDAGW', 'RDAR', 'RDCM', 'RDDT', 'RDGA',
    'RDGL', 'RDGMF', 'RDGT', 'RDHL', 'RDI', 'RDIB', 'RDN', 'RDNT', 'RDNW', 'RDPTF',
    'RDVT', 'RDW', 'RDWQS', 'RDWR', 'RDY', 'RDZN', 'RDZNW', 'REAL', 'REAX', 'REBN',
    'RECCY', 'RECT', 'REDRF', 'REE', 'REECF', 'REED', 'REEDD', 'REEMF', 'REEUF', 'REEWF',
    'REFI', 'REFR', 'REG', 'REGCO', 'REGCP', 'REGN', 'REI', 'REKR', 'RELI', 'RELIW',
    'RELL', 'RELX', 'RELY', 'RENEF', 'RENGY', 'RENT', 'RENXF', 'REOS', 'REPL', 'REPX',
    'RERE', 'RES', 'RETO', 'REVB', 'REVBW', 'REVFF', 'REVG', 'REX', 'REXR', 'REXR-PB',
    'REXR-PC', 'REYN', 'REZI', 'RF', 'RF-PC', 'RF-PE', 'RF-PF', 'RFAI', 'RFAIR', 'RFAIU',
    'RFI', 'RFIL', 'RFL', 'RFL-WT', 'RFM', 'RFMZ', 'RGA', 'RGBP', 'RGBPP', 'RGC',
    'RGCCF', 'RGCO', 'RGEN', 'RGGG', 'RGLD', 'RGNT', 'RGNX', 'RGP', 'RGPX', 'RGR',
    'RGS', 'RGT', 'RGTI', 'RGTIW', 'RH', 'RHEP', 'RHEPA', 'RHEPB', 'RHEPZ', 'RHI',
    'RHLD', 'RHNO', 'RHP', 'RIBB', 'RIBBR', 'RIBBU', 'RICK', 'RIG', 'RIGL', 'RIKU',
    'RILY', 'RILYG', 'RILYK', 'RILYL', 'RILYN', 'RILYP', 'RILYT', 'RILYZ', 'RIME', 'RIO',
    'RIOFF', 'RIOT', 'RITE', 'RITM', 'RITM-PA', 'RITM-PB', 'RITM-PC', 'RITM-PD', 'RITM-PE', 'RITR',
    'RIV', 'RIV-PA', 'RIVF', 'RIVN', 'RJET', 'RJF', 'RJF-PB', 'RKDA', 'RKGRY', 'RKHNF',
    'RKLB', 'RKLIF', 'RKT', 'RKUNF', 'RKUNY', 'RKWAD', 'RKWBF', 'RL', 'RLAY', 'RLBY',
    'RLEA', 'RLFTF', 'RLFTY', 'RLGT', 'RLI', 'RLJ', 'RLJ-PA', 'RLMD', 'RLNDF', 'RLTY',
    'RLX', 'RLXXF', 'RLYB', 'RLYGF', 'RM', 'RMAX', 'RMBI', 'RMBS', 'RMCF', 'RMCO',
    'RMCOW', 'RMD', 'RMESF', 'RMHI', 'RMI', 'RMIX', 'RMM', 'RMMZ', 'RMNI', 'RMR',
    'RMSG', 'RMSGW', 'RMSL', 'RMT', 'RMTG', 'RMTI', 'RMXI', 'RNA', 'RNAC', 'RNAZ',
    'RNBW', 'RNG', 'RNGC', 'RNGE', 'RNGOF', 'RNGR', 'RNGT', 'RNGTU', 'RNGTW', 'RNKGF',
    'RNLXY', 'RNP', 'RNR', 'RNR-PF', 'RNR-PG', 'RNST', 'RNTX', 'RNW', 'RNWWW', 'RNXT',
    'ROAD', 'ROCK', 'ROG', 'ROIV', 'ROK', 'ROKU', 'ROL', 'ROLR', 'ROMA', 'ROMJF',
    'ROOT', 'ROP', 'ROST', 'ROYL', 'ROYMY', 'RPAY', 'RPD', 'RPDL', 'RPGL', 'RPID',
    'RPM', 'RPMT', 'RPRX', 'RPT', 'RPT-PC', 'RPTX', 'RQI', 'RR', 'RRACF', 'RRBI',
    'RRC', 'RRGB', 'RRR', 'RRX', 'RS', 'RSCF', 'RSF', 'RSG', 'RSHGY', 'RSHL',
    'RSI', 'RSKD', 'RSKIA', 'RSMDF', 'RSMXD', 'RSMXF', 'RSNHF', 'RSRBF', 'RSRV', 'RSSS',
    'RSTRF', 'RSVR', 'RSVRW', 'RTAC', 'RTACU', 'RTACW', 'RTCJF', 'RTGN', 'RTNTF', 'RTO',
    'RTON', 'RTPPF', 'RTSL', 'RTX', 'RUBI', 'RUM', 'RUMBW', 'RUN', 'RUPRF', 'RUSHA',
    'RUSHB', 'RVI', 'RVLGF', 'RVLV', 'RVMD', 'RVMDW', 'RVP', 'RVPH', 'RVPHW', 'RVRC',
    'RVRF', 'RVSB', 'RVSN', 'RVSNW', 'RVT', 'RVTY', 'RVYL', 'RWAY', 'RWAYL', 'RWAYZ',
    'RWT', 'RWT-PA', 'RWTN', 'RWTO', 'RWTP', 'RWTQ', 'RXO', 'RXRX', 'RXST', 'RXT',
    'RY', 'RYAAY', 'RYAM', 'RYAN', 'RYAOF', 'RYDAF', 'RYDE', 'RYES', 'RYET', 'RYI',
    'RYKKF', 'RYLBF', 'RYLPF', 'RYM', 'RYN', 'RYOJ', 'RYPBF', 'RYTM', 'RZB', 'RZC',
    'RZLT', 'RZLV', 'RZLVW', 'S', 'SA', 'SAABF', 'SABA', 'SABOF', 'SABR', 'SABS',
    'SABSW', 'SACH', 'SACH-PA', 'SADMF', 'SAFE', 'SAFT', 'SAFX', 'SAGGF', 'SAGT', 'SAH',
    'SAIA', 'SAIC', 'SAIH', 'SAIHW', 'SAIL', 'SAJ', 'SALM', 'SAM', 'SAMG', 'SAN',
    'SANA', 'SAND', 'SANG', 'SANM', 'SANW', 'SAP', 'SAPGF', 'SAPIF', 'SAPUY', 'SAR',
    'SARO', 'SASOF', 'SAT', 'SATA', 'SATL', 'SATLF', 'SATLW', 'SATS', 'SATT', 'SAVA',
    'SAXPF', 'SAY', 'SAZ', 'SB', 'SB-PC', 'SB-PD', 'SBAC', 'SBAQ', 'SBBCF', 'SBBTF',
    'SBC', 'SBCF', 'SBCLY', 'SBCWW', 'SBDS', 'SBET', 'SBEV', 'SBEVW', 'SBFG', 'SBFM',
    'SBFMW', 'SBGI', 'SBH', 'SBI', 'SBIG', 'SBIGW', 'SBKLF', 'SBLK', 'SBLX', 'SBMW',
    'SBOEF', 'SBR', 'SBRA', 'SBS', 'SBSI', 'SBSW', 'SBUX', 'SBXD', 'SBXD-UN', 'SBXD-WT',
    'SBXE', 'SBYSF', 'SCAG', 'SCAGW', 'SCBXY', 'SCCD', 'SCCE', 'SCCF', 'SCCG', 'SCCO',
    'SCD', 'SCDL', 'SCE-PG', 'SCE-PJ', 'SCE-PK', 'SCE-PL', 'SCE-PM', 'SCE-PN', 'SCGX', 'SCGY',
    'SCHL', 'SCHW', 'SCHW-PD', 'SCHW-PJ', 'SCI', 'SCIA', 'SCII', 'SCIIU', 'SCKT', 'SCL',
    'SCLX', 'SCLXW', 'SCM', 'SCND', 'SCNI', 'SCNX', 'SCO', 'SCOR', 'SCPQ', 'SCRNY',
    'SCS', 'SCSC', 'SCSKF', 'SCSXY', 'SCTAY', 'SCTH', 'SCTSF', 'SCVL', 'SCWO', 'SCXBF',
    'SCYX', 'SCYYF', 'SD', 'SDA', 'SDAWW', 'SDCH', 'SDGR', 'SDHC', 'SDHI', 'SDHIR',
    'SDHIU', 'SDHY', 'SDM', 'SDOT', 'SDRL', 'SDST', 'SDSTW', 'SDSYA', 'SE', 'SEAH',
    'SEAL-PA', 'SEAL-PB', 'SEAT', 'SEATW', 'SEAV', 'SEB', 'SEDG', 'SEE', 'SEED', 'SEER',
    'SEG', 'SEGG', 'SEI', 'SEIC', 'SELF', 'SELX', 'SEM', 'SEMR', 'SENEA', 'SENEB',
    'SENEL', 'SENEM', 'SENR', 'SENS', 'SEOVF', 'SEPN', 'SEPSF', 'SER', 'SERA', 'SERPY',
    'SERV', 'SES', 'SES-WT', 'SETO', 'SEV', 'SEVN', 'SEVNR', 'SEZL', 'SF', 'SF-PB',
    'SF-PC', 'SF-PD', 'SFB', 'SFBC', 'SFBS', 'SFCO', 'SFD', 'SFDL', 'SFDMF', 'SFDMY',
    'SFES', 'SFGRF', 'SFGYY', 'SFHG', 'SFIX', 'SFL', 'SFM', 'SFNC', 'SFRX', 'SFST',
    'SFUNY', 'SFWJ', 'SFWL', 'SG', 'SGA', 'SGBAF', 'SGBX', 'SGC', 'SGD', 'SGHC',
    'SGHT', 'SGI', 'SGIHY', 'SGIOF', 'SGIPF', 'SGLA', 'SGLY', 'SGML', 'SGMO', 'SGMT',
    'SGN', 'SGOL', 'SGRP', 'SGRY', 'SGTM', 'SGU', 'SHAK', 'SHBI', 'SHC', 'SHCO',
    'SHEL', 'SHEN', 'SHFS', 'SHFSW', 'SHG', 'SHGI', 'SHIM', 'SHIP', 'SHLAF', 'SHLRF',
    'SHLS', 'SHMD', 'SHMDW', 'SHMLF', 'SHMXY', 'SHMZF', 'SHNY', 'SHO', 'SHO-PH', 'SHO-PI',
    'SHOO', 'SHOP', 'SHPH', 'SHTPY', 'SHVLF', 'SHW', 'SI', 'SIBN', 'SIBO', 'SID',
    'SIDU', 'SIEB', 'SIF', 'SIFY', 'SIG', 'SIGA', 'SIGI', 'SIGIP', 'SIGY', 'SII',
    'SILA', 'SILC', 'SILEF', 'SILO', 'SIM', 'SIMA', 'SIMAU', 'SIMAW', 'SIMO', 'SINC',
    'SINT', 'SION', 'SIPN', 'SIREF', 'SIRI', 'SISI', 'SITC', 'SITE', 'SITKF', 'SITM',
    'SITS', 'SIVR', 'SJ', 'SJM', 'SJT', 'SKAS', 'SKBL', 'SKE', 'SKFG', 'SKIL',
    'SKILW', 'SKIN', 'SKK', 'SKKY', 'SKLTF', 'SKLZ', 'SKM', 'SKNGY', 'SKT', 'SKVI',
    'SKWD', 'SKY', 'SKYE', 'SKYH', 'SKYH-WT', 'SKYI', 'SKYQ', 'SKYT', 'SKYW', 'SKYX',
    'SLAB', 'SLAI', 'SLAMF', 'SLB', 'SLBK', 'SLDB', 'SLDC', 'SLDE', 'SLDP', 'SLDPW',
    'SLE', 'SLF', 'SLFIF', 'SLFPF', 'SLFPY', 'SLFQF', 'SLG', 'SLG-PI', 'SLGB', 'SLGL',
    'SLGN', 'SLI', 'SLM', 'SLMBP', 'SLMT', 'SLN', 'SLNCF', 'SLND', 'SLND-WT', 'SLNG',
    'SLNH', 'SLNHP', 'SLNO', 'SLP', 'SLQT', 'SLRC', 'SLRX', 'SLS', 'SLSN', 'SLSR',
    'SLTN', 'SLV', 'SLVM', 'SLVO', 'SLXN', 'SLXNW', 'SM', 'SMA', 'SMBC', 'SMBK',
    'SMC', 'SMCI', 'SMDRF', 'SMFG', 'SMFNF', 'SMFRF', 'SMFSY', 'SMG', 'SMHB', 'SMHI',
    'SMID', 'SMIP', 'SMKG', 'SMLR', 'SMMT', 'SMNR', 'SMNRW', 'SMOFF', 'SMP', 'SMPHY',
    'SMPL', 'SMR', 'SMRT', 'SMSEY', 'SMSI', 'SMSOF', 'SMTC', 'SMTGF', 'SMTI', 'SMTK',
    'SMWB', 'SMX', 'SMXT', 'SMXWW', 'SN', 'SNA', 'SNAL', 'SNAP', 'SNBH', 'SNBR',
    'SNCR', 'SNCY', 'SND', 'SNDA', 'SNDK', 'SNDL', 'SNDR', 'SNDX', 'SNEJF', 'SNES',
    'SNEVY', 'SNEX', 'SNFCA', 'SNGX', 'SNIRF', 'SNN', 'SNNC', 'SNNF', 'SNNGF', 'SNNRF',
    'SNNUF', 'SNOA', 'SNOW', 'SNPMF', 'SNPS', 'SNPTF', 'SNPW', 'SNRBY', 'SNRG', 'SNROF',
    'SNROY', 'SNSE', 'SNT', 'SNTG', 'SNTI', 'SNTL', 'SNTUF', 'SNTW', 'SNV', 'SNV-PD',
    'SNV-PE', 'SNWV', 'SNX', 'SNY', 'SNYNF', 'SNYR', 'SO', 'SOAGY', 'SOAR', 'SOARW',
    'SOBO', 'SOBR', 'SOC', 'SOCA', 'SOCAU', 'SOCAW', 'SOCGM', 'SOCGP', 'SODI', 'SOEZ',
    'SOFI', 'SOGP', 'SOHGF', 'SOHGY', 'SOHO', 'SOHOB', 'SOHON', 'SOHOO', 'SOHU', 'SOJC',
    'SOJD', 'SOJE', 'SOJF', 'SOL', 'SOLC', 'SOLS', 'SOLV', 'SOMN', 'SON', 'SOND',
    'SONDQ', 'SONDW', 'SONG', 'SONM', 'SONN', 'SONO', 'SONWQ', 'SONX', 'SONY', 'SOPA',
    'SOPH', 'SOR', 'SORA', 'SOS', 'SOTGY', 'SOTK', 'SOUL', 'SOUL-RI', 'SOUL-UN', 'SOUN',
    'SOUNW', 'SOWG', 'SOYB', 'SPAI', 'SPAUF', 'SPB', 'SPCB', 'SPCE', 'SPE', 'SPE-PC',
    'SPED', 'SPEG', 'SPEGR', 'SPEGU', 'SPEV', 'SPFI', 'SPFX', 'SPG', 'SPG-PJ', 'SPGDF',
    'SPGI', 'SPGNY', 'SPH', 'SPHIF', 'SPHL', 'SPHR', 'SPHXF', 'SPIIY', 'SPIR', 'SPIWF',
    'SPKL', 'SPKLU', 'SPKLW', 'SPLP', 'SPMA', 'SPMC', 'SPME', 'SPND', 'SPNS', 'SPNT',
    'SPNT-PB', 'SPOK', 'SPOT', 'SPOWF', 'SPPL', 'SPPP', 'SPQS', 'SPR', 'SPRB', 'SPRC',
    'SPRO', 'SPRS', 'SPRU', 'SPRY', 'SPSC', 'SPST', 'SPT', 'SPTJF', 'SPTY', 'SPWH',
    'SPWR', 'SPWRW', 'SPXC', 'SPXSF', 'SPXX', 'SPY', 'SQEI', 'SQFT', 'SQFTP', 'SQFTW',
    'SQLLW', 'SQM', 'SQNS', 'SR', 'SR-PA', 'SRAD', 'SRBEF', 'SRBK', 'SRCE', 'SRCO',
    'SRE', 'SREA', 'SRFM', 'SRG', 'SRG-PA', 'SRGXF', 'SRGZ', 'SRI', 'SRKZF', 'SRL',
    'SRLZF', 'SRMX', 'SRPT', 'SRRE', 'SRRK', 'SRTA', 'SRTAW', 'SRTS', 'SRV', 'SRV-RI',
    'SRXH', 'SRZN', 'SRZNW', 'SSAC', 'SSB', 'SSD', 'SSEA', 'SSEAR', 'SSEAU', 'SSGC',
    'SSHT', 'SSII', 'SSKN', 'SSL', 'SSM', 'SSMXF', 'SSNC', 'SSP', 'SSPFF', 'SSRGF',
    'SSRLY', 'SSRM', 'SSSS', 'SSSSL', 'SSST', 'SST', 'SSTI', 'SSTK', 'SSTPW', 'SSTR',
    'SSUP', 'SSVFF', 'SSYS', 'ST', 'STAA', 'STAG', 'STAI', 'STAK', 'STBA', 'STBXF',
    'STC', 'STCB', 'STCK', 'STE', 'STEC', 'STEK', 'STEL', 'STEM', 'STEP', 'STEW',
    'STEX', 'STFS', 'STG', 'STGW', 'STHO', 'STI', 'STIM', 'STJNY', 'STK', 'STKE',
    'STKH', 'STKL', 'STKS', 'STKTF', 'STLA', 'STLD', 'STLE', 'STLJF', 'STLNF', 'STLRF',
    'STLY', 'STM', 'STME', 'STMEF', 'STN', 'STNDF', 'STNE', 'STNG', 'STOHF', 'STOK',
    'STQN', 'STRA', 'STRC', 'STRD', 'STRF', 'STRG', 'STRK', 'STRL', 'STRO', 'STRR',
    'STRRP', 'STRS', 'STRT', 'STRW', 'STRZ', 'STSBF', 'STSFF', 'STSR', 'STSS', 'STSSW',
    'STT', 'STT-PG', 'STTDF', 'STTK', 'STTPF', 'STUB', 'STVN', 'STWD', 'STX', 'STXS',
    'STZ', 'SU', 'SUGP', 'SUGRF', 'SUI', 'SUIC', 'SUIG', 'SUN', 'SUNC', 'SUND',
    'SUNE', 'SUNFF', 'SUNS', 'SUNXF', 'SUPN', 'SUPV', 'SUPX', 'SURDF', 'SURG', 'SUUN',
    'SUWA', 'SUZ', 'SVA', 'SVAC', 'SVACU', 'SVACW', 'SVAQ', 'SVAUF', 'SVBL', 'SVC',
    'SVCC', 'SVCCU', 'SVCCW', 'SVCO', 'SVICY', 'SVIIF', 'SVIRF', 'SVIUF', 'SVIWF', 'SVIX',
    'SVM', 'SVMB', 'SVNDF', 'SVNHF', 'SVRA', 'SVRE', 'SVREW', 'SVRN', 'SVRSF', 'SVUHF',
    'SVUWF', 'SVV', 'SVVC', 'SVXY', 'SVYSF', 'SW', 'SWAG', 'SWAGW', 'SWBI', 'SWIM',
    'SWISF', 'SWK', 'SWKH', 'SWKHL', 'SWKS', 'SWRD', 'SWVL', 'SWVLW', 'SWX', 'SWZ',
    'SXC', 'SXGCF', 'SXI', 'SXT', 'SXTC', 'SXTP', 'SXTPW', 'SY', 'SYAXF', 'SYBT',
    'SYBX', 'SYF', 'SYF-PA', 'SYF-PB', 'SYHBF', 'SYIN', 'SYK', 'SYM', 'SYNA', 'SYNSY',
    'SYNX', 'SYPR', 'SYRA', 'SYRE', 'SYY', 'SZZL', 'SZZLR', 'SZZLU', 'T', 'T-PA',
    'T-PC', 'TAAG', 'TAC', 'TACH', 'TACHU', 'TACHW', 'TACO', 'TACOU', 'TACOW', 'TACPF',
    'TACT', 'TAGS', 'TAIT', 'TAK', 'TAL', 'TALK', 'TALKW', 'TALO', 'TANAF', 'TANH',
    'TAOP', 'TAOX', 'TAP', 'TAP-A', 'TAPR', 'TARA', 'TARS', 'TARSF', 'TASK', 'TATT',
    'TAVI', 'TAVIR', 'TAVIU', 'TAYD', 'TBB', 'TBBB', 'TBBK', 'TBCH', 'TBH', 'TBHC',
    'TBI', 'TBLA', 'TBLAW', 'TBLD', 'TBLLF', 'TBMC', 'TBMCR', 'TBN', 'TBNRL', 'TBPH',
    'TBRG', 'TBTC', 'TC', 'TCANF', 'TCBC', 'TCBI', 'TCBIO', 'TCBK', 'TCBPY', 'TCBS',
    'TCBWF', 'TCBX', 'TCENF', 'TCEYF', 'TCGL', 'TCI', 'TCKRF', 'TCLHF', 'TCLXY', 'TCMD',
    'TCMEF', 'TCMFF', 'TCNCF', 'TCNNF', 'TCOM', 'TCPA', 'TCPC', 'TCRG', 'TCRI', 'TCRT',
    'TCRX', 'TCX', 'TD', 'TDAC', 'TDACU', 'TDACW', 'TDAY', 'TDBCP', 'TDC', 'TDDWW',
    'TDF', 'TDG', 'TDGGF', 'TDGMW', 'TDHG', 'TDIC', 'TDOC', 'TDRRF', 'TDS', 'TDS-PU',
    'TDS-PV', 'TDST', 'TDTH', 'TDUP', 'TDW', 'TDWD', 'TDWDU', 'TDY', 'TE', 'TE-WT',
    'TEAD', 'TEAM', 'TECH', 'TECK', 'TECTP', 'TECX', 'TEF', 'TEFOF', 'TEI', 'TEICY',
    'TEL', 'TELA', 'TELNF', 'TELNY', 'TELO', 'TEM', 'TEN', 'TEN-PE', 'TEN-PF', 'TENB',
    'TENX', 'TEO', 'TER', 'TERN', 'TESI', 'TETEF', 'TETH', 'TETOF', 'TETUF', 'TETWF',
    'TEVA', 'TEVJF', 'TEX', 'TFC', 'TFC-PI', 'TFC-PO', 'TFC-PR', 'TFII', 'TFIN', 'TFIN-P',
    'TFLM', 'TFPM', 'TFSA', 'TFSL', 'TFX', 'TG', 'TGB', 'TGCB', 'TGE', 'TGE-WT',
    'TGEN', 'TGHL', 'TGL', 'TGLO', 'TGLS', 'TGMPF', 'TGNA', 'TGNT', 'TGOPF', 'TGS',
    'TGT', 'TGTX', 'TH', 'THAR', 'THC', 'THCH', 'THCLY', 'THFF', 'THG', 'THH',
    'THM', 'THMG', 'THO', 'THQ', 'THQQF', 'THR', 'THRM', 'THRY', 'THS', 'THSGF',
    'THURF', 'THW', 'TIC', 'TICAW', 'TIGCF', 'TIGO', 'TIGR', 'TII', 'TIKK', 'TIL',
    'TILE', 'TIMB', 'TIMCD', 'TINFF', 'TIPT', 'TIRX', 'TISI', 'TITN', 'TIVC', 'TIXT',
    'TJX', 'TK', 'TKC', 'TKCM', 'TKCYY', 'TKLF', 'TKLS', 'TKMEF', 'TKMO', 'TKMTY',
    'TKNO', 'TKO', 'TKOI', 'TKPHF', 'TKR', 'TKRFD', 'TLF', 'TLGUF', 'TLGWF', 'TLGYF',
    'TLIH', 'TLK', 'TLLTF', 'TLN', 'TLNC', 'TLNCU', 'TLNCW', 'TLPH', 'TLPPF', 'TLRY',
    'TLS', 'TLSA', 'TLSI', 'TLSIW', 'TLSS', 'TLX', 'TLYS', 'TM', 'TMC', 'TMCI',
    'TMCWW', 'TMDE', 'TMDX', 'TME', 'TMGI', 'TMGX', 'TMHC', 'TMKVY', 'TMO', 'TMP',
    'TMQ', 'TMRC', 'TMRD', 'TMSOF', 'TMTNF', 'TMTNY', 'TMUS', 'TMUSI', 'TMUSL', 'TMUSZ',
    'TNBI', 'TNC', 'TNCAF', 'TNDEF', 'TNDM', 'TNET', 'TNEYF', 'TNGX', 'TNK', 'TNL',
    'TNMG', 'TNMWF', 'TNON', 'TNONW', 'TNRSF', 'TNXP', 'TNYA', 'TNYZD', 'TNYZF', 'TOBAF',
    'TOETF', 'TOFB', 'TOGI', 'TOGIW', 'TOI', 'TOIIW', 'TOKCF', 'TOL', 'TOMYF', 'TOMZ',
    'TONX', 'TOON', 'TOP', 'TOPP', 'TOPS', 'TORO', 'TOST', 'TOUR', 'TOVX', 'TOXR',
    'TOYO', 'TOYOF', 'TOYWF', 'TPB', 'TPC', 'TPCS', 'TPET', 'TPG', 'TPGXL', 'TPH',
    'TPHS', 'TPICQ', 'TPL', 'TPR', 'TPST', 'TPTA', 'TPVG', 'TPZEF', 'TPZEY', 'TR',
    'TRAK', 'TRAW', 'TRBMF', 'TRBRF', 'TRC', 'TRCK', 'TRDA', 'TREE', 'TREO', 'TREX',
    'TRGP', 'TRI', 'TRIB', 'TRIN', 'TRINI', 'TRINZ', 'TRIP', 'TRLC', 'TRLEF', 'TRMB',
    'TRMD', 'TRMK', 'TRMLF', 'TRMOY', 'TRN', 'TRNO', 'TRNR', 'TRNS', 'TROLB', 'TRON',
    'TROO', 'TROW', 'TROX', 'TRP', 'TRPCF', 'TRPEF', 'TRPPF', 'TRPRF', 'TRRFF', 'TRS',
    'TRSG', 'TRSO', 'TRST', 'TRT', 'TRTN-PA', 'TRTN-PB', 'TRTN-PC', 'TRTN-PD', 'TRTN-PE', 'TRTN-PF',
    'TRTX', 'TRTX-PC', 'TRU', 'TRUE', 'TRUG', 'TRUP', 'TRV', 'TRVG', 'TRVI', 'TRWD',
    'TRX', 'TRXA', 'TS', 'TSAT', 'TSBK', 'TSCFY', 'TSCO', 'TSE', 'TSEM', 'TSHA',
    'TSI', 'TSKFF', 'TSLA', 'TSLTF', 'TSLVF', 'TSLX', 'TSM', 'TSMWF', 'TSN', 'TSNDF',
    'TSOL', 'TSPH', 'TSQ', 'TSSI', 'TT', 'TTAM', 'TTAN', 'TTC', 'TTD', 'TTE',
    'TTEC', 'TTEI', 'TTEK', 'TTFNF', 'TTGT', 'TTI', 'TTIPF', 'TTMI', 'TTRX', 'TTSH',
    'TTWO', 'TU', 'TUNGF', 'TURB', 'TUSK', 'TUTH', 'TUYA', 'TV', 'TVA', 'TVACU',
    'TVACW', 'TVAI', 'TVAIR', 'TVAIU', 'TVC', 'TVCN', 'TVE', 'TVGN', 'TVGNW', 'TVRD',
    'TVTX', 'TW', 'TWFG', 'TWG', 'TWI', 'TWIN', 'TWLO', 'TWN', 'TWNP', 'TWO',
    'TWO-PA', 'TWO-PB', 'TWO-PC', 'TWOD', 'TWOH', 'TWST', 'TX', 'TXEMF', 'TXG', 'TXMD',
    'TXN', 'TXNM', 'TXO', 'TXRH', 'TXT', 'TY', 'TY-P', 'TYBB', 'TYFG', 'TYG',
    'TYGO', 'TYHOF', 'TYHOY', 'TYHT', 'TYL', 'TYNPF', 'TYRA', 'TYTMF', 'TZOO', 'TZUP',
    'U', 'UA', 'UAA', 'UAL', 'UAMY', 'UAN', 'UATCY', 'UAVS', 'UBCP', 'UBER',
    'UBFO', 'UBOH', 'UBS', 'UBSI', 'UBXG', 'UCAR', 'UCASU', 'UCB', 'UCFI', 'UCFIW',
    'UCIB', 'UCIX', 'UCL', 'UCO', 'UCTT', 'UDMY', 'UDN', 'UDR', 'UE', 'UEC',
    'UECXF', 'UEEC', 'UEIC', 'UELMO', 'UEPCN', 'UEPCO', 'UEPCP', 'UEPEM', 'UEPEN', 'UEPEO',
    'UEPEP', 'UFCS', 'UFG', 'UFI', 'UFPI', 'UFPT', 'UG', 'UGA', 'UGI', 'UGL',
    'UGP', 'UGRO', 'UHAL', 'UHAL-B', 'UHG', 'UHGI', 'UHGWW', 'UHL', 'UHP', 'UHS',
    'UHT', 'UI', 'UIS', 'UK', 'UL', 'ULBI', 'ULCC', 'ULE', 'ULH', 'ULIX',
    'ULS', 'ULTA', 'ULY', 'UMAC', 'UMBF', 'UMBFO', 'UMC', 'UMEWF', 'UMH', 'UMH-PD',
    'UNB', 'UNCY', 'UNF', 'UNFI', 'UNG', 'UNH', 'UNIB', 'UNIT', 'UNL', 'UNLYF',
    'UNM', 'UNMA', 'UNP', 'UNTC', 'UNTCW', 'UNTY', 'UNXP', 'UOKA', 'UONE', 'UONEK',
    'UP', 'UPB', 'UPBD', 'UPC', 'UPLD', 'UPS', 'UPST', 'UPWK', 'UPX', 'UPXI',
    'UPYY', 'URBN', 'URG', 'URGN', 'URI', 'UROY', 'URZEF', 'USA', 'USAC', 'USAQ',
    'USAR', 'USARW', 'USAS', 'USAU', 'USB', 'USB-PA', 'USB-PH', 'USB-PP', 'USB-PQ', 'USB-PR',
    'USB-PS', 'USBC', 'USCB', 'USCI', 'USCTF', 'USDE', 'USDP', 'USEA', 'USEG', 'USFD',
    'USGDF', 'USGO', 'USGOW', 'USIC', 'USIO', 'USL', 'USLM', 'USML', 'USNA', 'USO',
    'USOI', 'USPCY', 'USPH', 'USTWF', 'UTF', 'UTG', 'UTGN', 'UTHR', 'UTI', 'UTKN',
    'UTL', 'UTMD', 'UTSI', 'UTX', 'UTZ', 'UUGRY', 'UUGWF', 'UUP', 'UURAF', 'UUU',
    'UUUFF', 'UUUU', 'UVE', 'UVIX', 'UVSP', 'UVV', 'UVXY', 'UWHR', 'UWMC', 'UWMC-WT',
    'UXIN', 'UYSC', 'UYSCR', 'UYSCU', 'UZD', 'UZE', 'UZF', 'V', 'VABK', 'VAC',
    'VACH', 'VACHU', 'VACHW', 'VACI', 'VACI-UN', 'VACI-WT', 'VAL', 'VAL-WT', 'VALE', 'VALN',
    'VALU', 'VANI', 'VARRY', 'VASO', 'VATE', 'VAUCF', 'VBF', 'VBIX', 'VBNK', 'VBREY',
    'VC', 'VCEL', 'VCIC', 'VCICU', 'VCICW', 'VCIG', 'VCNX', 'VCRDX', 'VCTR', 'VCUFF',
    'VCV', 'VCYT', 'VECO', 'VEEA', 'VEEAW', 'VEEE', 'VEEV', 'VEL', 'VELO', 'VENAF',
    'VENU', 'VEON', 'VERA', 'VERI', 'VERO', 'VERTF', 'VERU', 'VERX', 'VEST', 'VET',
    'VFC', 'VFF', 'VFL', 'VFS', 'VFSWW', 'VG', 'VGAS', 'VGASW', 'VGES', 'VGI',
    'VGM', 'VGZ', 'VHABW', 'VHAI', 'VHAIW', 'VHC', 'VHCP', 'VHI', 'VIA', 'VIASP',
    'VIAV', 'VICI', 'VICP', 'VICR', 'VIIQ', 'VIK', 'VINC', 'VINP', 'VIOT', 'VIPRF',
    'VIPS', 'VIPZ', 'VIR', 'VIRC', 'VIRT', 'VIRX', 'VISL', 'VISM', 'VIST', 'VITL',
    'VIV', 'VIVC', 'VIVK', 'VIVS', 'VIXM', 'VIXY', 'VIZNF', 'VKI', 'VKQ', 'VKTX',
    'VLDXW', 'VLGEA', 'VLN', 'VLN-WT', 'VLO', 'VLRS', 'VLT', 'VLTLF', 'VLTO', 'VLY',
    'VLYPN', 'VLYPO', 'VLYPP', 'VMAR', 'VMC', 'VMCAF', 'VMCUF', 'VMCWF', 'VMD', 'VMEO',
    'VMI', 'VMNT', 'VMO', 'VNCE', 'VNDA', 'VNET', 'VNJA', 'VNME', 'VNMEU', 'VNMEW',
    'VNO', 'VNO-PL', 'VNO-PM', 'VNO-PN', 'VNO-PO', 'VNOM', 'VNORP', 'VNRX', 'VNT', 'VNTG',
    'VNTH', 'VNUE', 'VOC', 'VOD', 'VODPF', 'VOR', 'VOXR', 'VOYA', 'VOYA-PB', 'VOYG',
    'VPER', 'VPG', 'VPLM', 'VPRB', 'VPV', 'VQSSF', 'VRA', 'VRAR', 'VRAX', 'VRBCF',
    'VRCA', 'VRDN', 'VRDR', 'VRE', 'VREOF', 'VREX', 'VRM', 'VRME', 'VRMWW', 'VRNO',
    'VRNS', 'VRNT', 'VROYF', 'VRRCF', 'VRRM', 'VRSK', 'VRSN', 'VRSSF', 'VRT', 'VRTS',
    'VRTX', 'VRXA', 'VS', 'VSA', 'VSAT', 'VSBGF', 'VSCO', 'VSEC', 'VSEE', 'VSEEW',
    'VSGRY', 'VSH', 'VSME', 'VSNT', 'VSOGF', 'VSSYW', 'VST', 'VSTA', 'VSTD', 'VSTM',
    'VSTS', 'VTAK', 'VTBAS', 'VTEK', 'VTEX', 'VTGN', 'VTLE', 'VTMX', 'VTN', 'VTOL',
    'VTR', 'VTRS', 'VTS', 'VTSI', 'VTTGF', 'VTVT', 'VTYB', 'VTYX', 'VUZI', 'VVOS',
    'VVPR', 'VVR', 'VVV', 'VVX', 'VWAV', 'VWAVW', 'VWFB', 'VXRT', 'VXX', 'VXZ',
    'VYCO', 'VYGR', 'VYLD', 'VYND', 'VYNE', 'VYRE', 'VYST', 'VYX', 'VZ', 'VZLA',
    'W', 'WAB', 'WABC', 'WAFD', 'WAFDP', 'WAFU', 'WAI', 'WAL', 'WAL-PA', 'WALD',
    'WALDW', 'WAMFF', 'WASH', 'WAST', 'WAT', 'WATT', 'WAVE', 'WAY', 'WB', 'WBD',
    'WBHC', 'WBI', 'WBKCY', 'WBQNL', 'WBS', 'WBS-PF', 'WBS-PG', 'WBSR', 'WBTN', 'WBUY',
    'WBX', 'WBXWF', 'WCC', 'WCN', 'WCPRF', 'WCT', 'WD', 'WDAY', 'WDC', 'WDFC',
    'WDH', 'WDI', 'WDLF', 'WDS', 'WDSP', 'WEA', 'WEAT', 'WEAV', 'WEBJF', 'WEBNF',
    'WEC', 'WEIBF', 'WELL', 'WELNF', 'WELPM', 'WELPP', 'WELUF', 'WELWF', 'WEN', 'WENN',
    'WENNU', 'WENNW', 'WERN', 'WES', 'WEST', 'WETH', 'WETO', 'WEWA', 'WEX', 'WEYS',
    'WF', 'WFC', 'WFC-PA', 'WFC-PC', 'WFC-PD', 'WFC-PL', 'WFC-PY', 'WFC-PZ', 'WFCF', 'WFCNP',
    'WFF', 'WFG', 'WFRD', 'WGO', 'WGRX', 'WGS', 'WGSWW', 'WH', 'WHD', 'WHEN',
    'WHF', 'WHFCL', 'WHG', 'WHGOF', 'WHLM', 'WHLR', 'WHLRD', 'WHLRL', 'WHLRP', 'WHLT',
    'WHR', 'WHTCF', 'WHWK', 'WIA', 'WILC', 'WIMI', 'WINA', 'WING', 'WINT', 'WINTW',
    'WIPKF', 'WIT', 'WIW', 'WIX', 'WK', 'WKC', 'WKEY', 'WKHS', 'WKSP', 'WLAC',
    'WLACU', 'WLACW', 'WLDN', 'WLDS', 'WLDSW', 'WLFC', 'WLFFF', 'WLGMF', 'WLGSF', 'WLK',
    'WLKP', 'WLMIF', 'WLSS', 'WLTH', 'WLY', 'WLYB', 'WM', 'WMB', 'WMG', 'WMK',
    'WMS', 'WMT', 'WNC', 'WNDW', 'WNEB', 'WNFT', 'WNHK', 'WNLV', 'WNS', 'WNW',
    'WOK', 'WOLF', 'WOLTF', 'WOLV', 'WONDF', 'WOOF', 'WOPEF', 'WOR', 'WORX', 'WOW',
    'WPC', 'WPGCF', 'WPM', 'WPP', 'WPPGF', 'WPRT', 'WRAP', 'WRB', 'WRB-PE', 'WRB-PF',
    'WRB-PG', 'WRB-PH', 'WRBY', 'WRD', 'WRIV', 'WRLD', 'WRLGF', 'WRLRF', 'WRN', 'WRPT',
    'WS', 'WSBC', 'WSBCO', 'WSBF', 'WSBK', 'WSC', 'WSFS', 'WSHP', 'WSKEF', 'WSM',
    'WSO', 'WSO-B', 'WSR', 'WST', 'WSTN', 'WSTNU', 'WSTRF', 'WSUPW', 'WT', 'WTBA',
    'WTCHF', 'WTER', 'WTF', 'WTFC', 'WTFCN', 'WTG', 'WTGUR', 'WTGUU', 'WTHVF', 'WTI',
    'WTID', 'WTIU', 'WTKWY', 'WTM', 'WTMA', 'WTMAR', 'WTMAU', 'WTO', 'WTRG', 'WTS',
    'WTTR', 'WTW', 'WU', 'WULF', 'WVE', 'WVVI', 'WVVIP', 'WW', 'WWD', 'WWR',
    'WWW', 'WXM', 'WY', 'WYFI', 'WYGC', 'WYHG', 'WYNN', 'WYTC', 'WYY', 'XAGE',
    'XAGEW', 'XAIR', 'XBIO', 'XBIT', 'XBP', 'XBPEW', 'XCBE', 'XCH', 'XCRT', 'XCUR',
    'XEL', 'XELB', 'XELLL', 'XENE', 'XERI', 'XERS', 'XFLH', 'XFLT', 'XFOR', 'XGN',
    'XHG', 'XHLD', 'XHR', 'XIFR', 'XIN', 'XITO', 'XJNGF', 'XLO', 'XMTR', 'XNCR',
    'XNDA', 'XNET', 'XNJJY', 'XOM', 'XOMA', 'XOMAO', 'XOMAP', 'XONI', 'XOS', 'XOSWW',
    'XP', 'XPD', 'XPEL', 'XPER', 'XPEV', 'XPL', 'XPNGF', 'XPO', 'XPOF', 'XPON',
    'XPRO', 'XRAY', 'XRP', 'XRPN', 'XRPNU', 'XRPNW', 'XRPZ', 'XRTX', 'XRX', 'XSIAX',
    'XTGRF', 'XTIA', 'XTKG', 'XTLB', 'XTNT', 'XTRAF', 'XTXXF', 'XWEL', 'XWIN', 'XXAAU',
    'XXC', 'XXI', 'XXII', 'XYF', 'XYJG', 'XYL', 'XYLB', 'XYZ', 'XZJCF', 'XZO',
    'YAAS', 'YALA', 'YAMHF', 'YB', 'YBGJ', 'YCBD', 'YCL', 'YCS', 'YCY', 'YCY-UN',
    'YCY-WT', 'YDDL', 'YDES', 'YDESW', 'YDKG', 'YELP', 'YETI', 'YEXT', 'YGMZ', 'YGSHY',
    'YHC', 'YHGJ', 'YHNA', 'YHNAR', 'YHNAU', 'YI', 'YIBO', 'YJ', 'YJGJ', 'YKLTF',
    'YKLTY', 'YLY', 'YMAT', 'YMHAY', 'YMM', 'YMT', 'YMXK', 'YORW', 'YOTA', 'YOU',
    'YOUL', 'YPF', 'YQ', 'YQAI', 'YRD', 'YSG', 'YSHLF', 'YSS', 'YSXT', 'YTFD',
    'YTRA', 'YUM', 'YUMC', 'YXT', 'YYAI', 'YYGH', 'Z', 'ZAPPF', 'ZAPWF', 'ZBAI',
    'ZBAO', 'ZBH', 'ZBIO', 'ZBRA', 'ZCAR', 'ZCARW', 'ZCMD', 'ZCRMF', 'ZCSH', 'ZD',
    'ZDAI', 'ZDAN', 'ZDCAF', 'ZDGE', 'ZDPY', 'ZEFIF', 'ZENA', 'ZENV', 'ZEO', 'ZEOWW',
    'ZEOX', 'ZEPP', 'ZETA', 'ZEUS', 'ZG', 'ZGJLY', 'ZGM', 'ZGN', 'ZH', 'ZHIHF',
    'ZHJD', 'ZICX', 'ZIM', 'ZION', 'ZIONP', 'ZIP', 'ZIVO', 'ZIVOW', 'ZJGIY', 'ZJK',
    'ZJLMF', 'ZJNGF', 'ZJYL', 'ZK', 'ZKH', 'ZKIN', 'ZKP', 'ZLAB', 'ZLDPF', 'ZLME',
    'ZM', 'ZNB', 'ZNKUF', 'ZNOG', 'ZNOGW', 'ZNTL', 'ZOMDF', 'ZONE', 'ZOOZ', 'ZOOZW',
    'ZPHYF', 'ZRCN', 'ZS', 'ZSHLY', 'ZSHOF', 'ZSICY', 'ZSL', 'ZSPC', 'ZSSK', 'ZTEK',
    'ZTO', 'ZTOEF', 'ZTR', 'ZTS', 'ZTSTF', 'ZUMZ', 'ZURA', 'ZVIA', 'ZVRA', 'ZVSA',
    'ZWS', 'ZY', 'ZYBT', 'ZYME', 'ZYXI', 'TLT', 'USDU', 'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.PA', 
    'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 
    'ETL.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'UG.PA', 
    'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STM.PA', 'TEP.PA', 'HO.PA', 
    'TTE.PA', 'URW.PA', 'FR.PA', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA', 'ABBN.SW', 'ADEN.SW', 'ALSO.SW', 'BAER.SW', 'BALN.SW', 'BANB.SW', 'BBBI.SW', 'CFR.SW', 'CIE.SW', 'CSGN.SW', 
     'DUFN.SW', 'EFGN.SW', 'EMSN.SW', 'GEBN.SW', 'GIVN.SW', 'GPSN.SW', 'HBMN.SW', 'IMPN.SW', 'KOF.SW', 'LHN.SW', 'LONN.SW', 
     'NEON.SW', 'NESN.SW', 'NWRN.SW', 'OCO.SW', 'PGHN.SW', 'RICN.SW', 'ROSE.SW', 'SAN.SW', 'SCMN.SW', 'SGSN.SW', 'SIX.SW', 'SLHN.SW', 
     'SMGR.SW', 'SREN.SW', 'STLN.SW', 'SWCA.SW', 'SWX.SW', 'SYNN.SW', 'TLSN.SW', 'UBSG.SW', 'UBSN.SW', 'UHR.SW', 'VAPO.SW', 'VIFN.SW', 'VTX.SW', 'ZURN.SW',
'ADYEN.AS', 'AKZO.AS', 'ASML.AS', 'BAMNB.AS', 'DSM.AS', 'HEIA.AS', 'INGA.AS', 'KPN.AS', 'NN.AS', 'PHIA.AS', 'RAND.AS', 'RDSB.AS', 'SBMO.AS', 'SHV.AS', 'TKWY.AS', 
 'UNA.AS', 'WKL.AS', '1JAN.DE', 'ADN.DE', 'AIXA.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE', 'BPE.DE', 'CBK.DE', 'CON.DE', 'DTE.DE', 'EON.DE', 'FME.DE', 
 'FPE3.DE', 'FRA.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'IWG.DE', 'KPN.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'NOAJ.DE', 'OSR.DE', 'RIB.DE', 'SAP.DE', 'SDF.DE', 
 'SIE.DE', 'TKA.DE', 'VOW3.DE', 'VZ.DE', 'WCH.DE', '3IN.L', 'AAL.L', 'ABF.L', 'ADM.L', 'AHT.L', 'AMGO.L', 'ANTO.L', 'ARM.L', 'AV.L', 'AVST.L', 'AZN.L', 'BA.L', 
 'BARC.L', 'BATS.L', 'BDEV.L', 'BHP.L', 'BP.L', 'BRBY.L', 'BT.L', 'CRH.L', 'CRST.L', 'DCC.L', 'DGE.L', 'DIAG.L', 'ENT.L', 'EXPN.L', 'FERG.L', 'GKN.L', 'GVC.L', 
 'HAS.L', 'HSBA.L', 'HSX.L', 'IMI.L', 'INF.L', 'INTU.L', 'ITRK.L', 'ITV.L', 'JET.L', 'JMAT.L', 'KGF.L', 'KIE.L', 'LAND.L', 'LGEN.L', 'LLOY.L', 'LSE.L', 'MGAM.L', 
 'MKS.L', 'MRW.L', 'NG.L', 'NMC.L', 'NXT.L', 'OCDO.L', 'PFC.L', 'PSH.L', 'PSN.L', 'REL.L', 'RIO.L', 'RMG.L', 'RMV.L', 'RR.L', 'RTO.L', 'SAB.L', 'SBRY.L', 'SGC.L', 
 'SGRO.L', 'SHB.L', 'SHP.L', 'SMT.L', 'SMWH.L', 'SOPH.L', 'SPX.L', 'SSE.L', 'STAN.L', 'STJ.L', 'STMR.L', 'SVS.L', 'TALK.L', 'TED.L', 'TEP.L', 'TGT.L', 'TPK.L', 'TUI.L',
 'TW.L', 'ULE.L', 'ULVR.L', 'UU.L', 'VOD.L', 'WEIR.L', 'WIZZ.L', 'WPP.L', 'A2A.MI', 'ACE.MI', 'AMP.MI', 'ATL.MI', 'AZIM.MI', 'BAMI.MI', 'BANCA.MI', 'BAMI.MI', 'BAS.MI', 
 'BPER.MI', 'BRE.MI', 'BU.MI', 'CALT.MI', 'CNHI.MI', 'CPR.MI', 'CRT.MI', 'ENEL.MI', 'ENI.MI', 'ERM.MI', 'EXO.MI', 'FER.MI', 'FBK.MI', 'G.MI', 'GEO.MI', 'IGD.MI', 'ILTY.MI', 
 'INT.MI', 'ISP.MI', 'IT.MI', 'LDO.MI', 'LUX.MI', 'MB.MI', 'MED.MI', 'MONC.MI', 'MPS.MI', 'MT.MI', 'NEXI.MI', 'PST.MI', 'PRY.MI', 'RCS.MI', 'REC.MI', 'SFER.MI', 'SIFI.MI',
 'SPE.MI', 'SRG.MI', 'STLA.MI', 'TEN.MI', 'TIT.MI', 'TOD.MI', 'UBI.MI', 'UCG.MI', 'UNI.MI', 'US.MI', 'ABE.MC', 'ACX.MC', 'ACS.MC', 'AENA.MC', 'AGF.MC', 'ALM.MC', 'AMPER.MC',
 'ANA.MC', 'APD.MC', 'ATE.MC', 'BBVA.MC', 'BKT.MC', 'BMN.MC', 'BNC.MC', 'CAI.MC', 'CIE.MC', 'CLNX.MC', 'COL.MC', 'DIA.MC', 'ELE.MC', 'ENG.MC', 'FER.MC', 'GAM.MC', 'GAS.MC', 
 'GRF.MC', 'IAG.MC', 'IDR.MC', 'IME.MC', 'IND.MC', 'ITX.MC', 'MAP.MC', 'MEL.MC', 'MER.MC', 'MRL.MC', 'MTS.MC', 'NAT.MC', 'NHK.MC', 'OHL.MC', 'PHP.MC', 'POP.MC', 'REE.MC', 
 'REP.MC', 'SAB.MC', 'SAN.MC', 'SGC.MC', 'SOL.MC', 'SPM.MC', 'TDE.MC', 'TEF.MC', 'TGS.MC', 'VIS.MC', 'CSH2.PA', 'PUST.PA', 'GOLD.AS', 'LQQ.PA', 'IEO', 'XOP', 'IEZ', 'SPOG', 'OILG', 'XLE', 
 'VDE', 'FENY', 'IYE', 'RSPG', 'IXC']

    
    selected_tickers = st.multiselect(
        "Sélectionnez des tickers à comparer",
        options=popular_tickers,
        default=['AAPL', 'MSFT', 'TSLA'],
        help="Sélectionnez jusqu'à 10 tickers"
    )

with col_chart2:
    timeframe = st.selectbox(
        "Timeframe",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y','10y', '15y', '20y', '25y', 'All'],
        index=3,
        help="Période d'analyse"
    )

# Fonction pour normaliser en %
def normalize_to_percentage(df):
    """Normalise les prix à 100% au début"""
    if df is None or len(df) == 0:
        return None
    return (df / df.iloc[0]) * 100

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

# Génération du graphique
if selected_tickers:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Couleurs Bloomberg style
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00', 
              '#FF1493', '#00CED1', '#32CD32', '#FFD700']
    
    # Stocker les données pour la corrélation
    correlation_data = pd.DataFrame()
    
    with st.spinner('📊 Chargement des données...'):
        for idx, ticker in enumerate(selected_tickers[:10]):
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


# ===== SIMULATEUR DE PORTEFEUILLE =====
st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
st.markdown("#### 💼 PORTFOLIO SIMULATOR")

# Interface de configuration du portefeuille
st.markdown("**Configurez les poids de chaque actif dans votre portefeuille :**")

# Créer une ligne de sliders pour chaque ticker sélectionné
weight_cols = st.columns(min(len(selected_tickers), 5))
weights = {}

for idx, ticker in enumerate(selected_tickers[:10]):
    with weight_cols[idx % 5]:
        weights[ticker] = st.slider(
            f"{ticker}",
            min_value=0.0,
            max_value=100.0,
            value=100.0 / len(selected_tickers),  # Répartition équitable par défaut
            step=1.0,
            key=f"weight_{ticker}"
        )

# Vérifier que la somme fait 100%
total_weight = sum(weights.values())
if abs(total_weight - 100.0) > 0.1:
    st.warning(f"⚠️ La somme des poids est de {total_weight:.1f}% (devrait être 100%)")
else:
    st.success(f"✅ Portefeuille équilibré : {total_weight:.1f}%")

# Calculer la performance du portefeuille
if total_weight > 0:
    portfolio_performance = pd.Series(dtype=float)
    portfolio_data = pd.DataFrame()
    
    # Normaliser les poids pour qu'ils totalisent 100%
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    for ticker, weight in normalized_weights.items():
        hist = get_historical_data(ticker, timeframe)
        if hist is not None and len(hist) > 0:
            normalized = normalize_to_percentage(hist['Close'])
            if portfolio_performance.empty:
                portfolio_performance = normalized * (weight)
            else:
                portfolio_performance = portfolio_performance.add(
                    normalized * (weight), 
                    fill_value=0
                )
            portfolio_data[ticker] = hist['Close']
    
    # Créer le graphique du portefeuille
    if not portfolio_performance.empty:
        fig_portfolio = go.Figure()
        
        # Ajouter les performances individuelles en transparence
        for idx, ticker in enumerate(selected_tickers[:10]):
            hist = get_historical_data(ticker, timeframe)
            if hist is not None and len(hist) > 0:
                normalized = normalize_to_percentage(hist['Close'])
                fig_portfolio.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=f"{ticker} ({weights[ticker]:.1f}%)",
                    line=dict(color=colors[idx % len(colors)], width=1, dash='dot'),
                    opacity=0.3,
                    showlegend=True
                ))
        
        # Ajouter la performance du portefeuille
        fig_portfolio.add_trace(go.Scatter(
            x=portfolio_performance.index,
            y=portfolio_performance.values,
            mode='lines',
            name='PORTFOLIO',
            line=dict(color='#FFAA00', width=4),
            hovertemplate='<b>PORTFOLIO</b><br>%{y:.2f}%<br>%{x}<extra></extra>'
        ))
        
        fig_portfolio.update_layout(
            title=f"Portfolio Performance vs Individual Assets - {timeframe.upper()}",
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
            height=500
        )
        
        # Ligne horizontale à 100%
        fig_portfolio.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=100, y1=100,
            line=dict(color="#666", width=1, dash="dash")
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Statistiques du portefeuille
        st.markdown("#### 📊 PORTFOLIO STATISTICS")
        
        start_value = portfolio_performance.iloc[0]
        end_value = portfolio_performance.iloc[-1]
        total_return = end_value - start_value
        
        # Volatilité (écart-type des rendements quotidiens)
        daily_returns = portfolio_performance.pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualisée
        
        # Sharpe Ratio (simplifié, sans taux sans risque)
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        stat_cols = st.columns(5)
        
        with stat_cols[0]:
            st.metric("Total Return", f"{total_return:+.2f}%")
        
        with stat_cols[1]:
            st.metric("Volatility (ann.)", f"{volatility:.2f}%")
        
        with stat_cols[2]:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with stat_cols[3]:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        with stat_cols[4]:
            end_date = portfolio_performance.index[-1].strftime('%Y-%m-%d')
            st.metric("End Date", end_date)
        
        # Composition du portefeuille
        st.markdown("**📋 Portfolio Composition:**")
        comp_text = " | ".join([f"{ticker}: {weight:.1f}%" for ticker, weight in weights.items() if weight > 0])
        st.text(comp_text)




# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES EN TEMPS RÉEL • YAHOO FINANCE + COINMARKETCAP<br>
        🔄 RAFRAÎCHISSEMENT AUTOMATIQUE: 3 SECONDES • AUCUNE ACTION REQUISE
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 CONNEXION: PARIS, FRANCE • SYSTÈME OPÉRATIONNEL
    </div>
    """, unsafe_allow_html=True)
# ===== PUB FOOTER =====
add_footer_ad()

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

