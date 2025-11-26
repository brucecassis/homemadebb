import streamlit as st
from datetime import datetime
import time

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Market Screener",
    page_icon="🔍",
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
    
    /* Style pour les tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #111;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #222;
        color: #FFAA00;
        border: 1px solid #333;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
    
    /* Style pour les selectbox */
    .stSelectbox > div > div {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #FFAA00 !important;
    }
    
    /* Iframe container */
    .screener-container {
        background: #000;
        border: 2px solid #FFAA00;
        border-radius: 0;
        padding: 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - MARKET SCREENER</div>
    </div>
    <div>{current_time} UTC • TRADINGVIEW DATA</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# BOUTONS DE NAVIGATION
# =============================================
nav_cols = st.columns(6)

with nav_cols[0]:
    if st.button("📊 MARKETS", use_container_width=True):
        st.switch_page("app.py")

with nav_cols[1]:
    if st.button("🔍 SCREENER", use_container_width=True):
        pass  # Déjà sur cette page

with nav_cols[2]:
    if st.button("📰 NEWS", use_container_width=True):
        st.switch_page("pages/NEWS.py")

with nav_cols[3]:
    if st.button("📁 EDGAR", use_container_width=True):
        st.switch_page("pages/EDGAR.py")

with nav_cols[4]:
    if st.button("💰 PRICING", use_container_width=True):
        st.switch_page("pages/PRICING.py")

with nav_cols[5]:
    if st.button("🤖 CHATBOT", use_container_width=True):
        st.switch_page("pages/CHATBOT.py")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# TITRE DE LA PAGE
# =============================================
st.markdown("### 🔍 MARKET SCREENER - TRADINGVIEW")
st.markdown("""
<div style="color:#888;font-size:11px;margin-bottom:15px;">
    Screener multi-marchés powered by TradingView • Filtrez par indicateurs techniques et fondamentaux
</div>
""", unsafe_allow_html=True)

# =============================================
# SÉLECTION DU TYPE DE MARCHÉ
# =============================================
col_market, col_theme, col_height = st.columns([2, 1, 1])

with col_market:
    market_type = st.selectbox(
        "📊 TYPE DE MARCHÉ",
        options=[
            "🇺🇸 US Stocks",
            "🇫🇷 France (Euronext Paris)",
            "🇬🇧 UK (London Stock Exchange)",
            "🇩🇪 Germany (XETRA)",
            "🇨🇭 Switzerland (SIX)",
            "🇯🇵 Japan (TSE)",
            "🇨🇳 China (SSE)",
            "🇭🇰 Hong Kong (HKEX)",
            "🇦🇺 Australia (ASX)",
            "🇨🇦 Canada (TSX)",
            "🇧🇷 Brazil (B3)",
            "🇮🇳 India (NSE)",
            "💱 Forex",
            "₿ Crypto Pairs",
            "🪙 Cryptocurrency Market",
        ],
        index=0,
        help="Sélectionnez le marché à screener"
    )

with col_theme:
    color_theme = st.selectbox(
        "🎨 THÈME",
        options=["dark", "light"],
        index=0,
        help="Thème du widget"
    )

with col_height:
    widget_height = st.selectbox(
        "📐 HAUTEUR",
        options=[500, 600, 700, 800, 900, 1000],
        index=3,
        help="Hauteur du widget en pixels"
    )

# =============================================
# MAPPING DES MARCHÉS
# =============================================
market_mapping = {
    "🇺🇸 US Stocks": ("america", "Stock Screener - US Markets"),
    "🇫🇷 France (Euronext Paris)": ("france", "Stock Screener - France"),
    "🇬🇧 UK (London Stock Exchange)": ("uk", "Stock Screener - UK"),
    "🇩🇪 Germany (XETRA)": ("germany", "Stock Screener - Germany"),
    "🇨🇭 Switzerland (SIX)": ("switzerland", "Stock Screener - Switzerland"),
    "🇯🇵 Japan (TSE)": ("japan", "Stock Screener - Japan"),
    "🇨🇳 China (SSE)": ("china", "Stock Screener - China"),
    "🇭🇰 Hong Kong (HKEX)": ("hongkong", "Stock Screener - Hong Kong"),
    "🇦🇺 Australia (ASX)": ("australia", "Stock Screener - Australia"),
    "🇨🇦 Canada (TSX)": ("canada", "Stock Screener - Canada"),
    "🇧🇷 Brazil (B3)": ("brazil", "Stock Screener - Brazil"),
    "🇮🇳 India (NSE)": ("india", "Stock Screener - India"),
    "💱 Forex": ("forex", "Forex Screener"),
    "₿ Crypto Pairs": ("crypto", "Crypto Pairs Screener"),
    "🪙 Cryptocurrency Market": ("crypto_mkt", "Cryptocurrency Market"),
}

selected_market, market_title = market_mapping[market_type]

# =============================================
# SÉLECTION DES COLONNES ET ÉCRANS
# =============================================
col_column, col_screen = st.columns(2)

with col_column:
    default_column = st.selectbox(
        "📋 COLONNES PAR DÉFAUT",
        options=[
            "overview",
            "performance",
            "oscillators",
            "moving_averages",
            "Ede35e23",  # Valuation
        ],
        format_func=lambda x: {
            "overview": "📊 Overview",
            "performance": "📈 Performance",
            "oscillators": "🔄 Oscillators",
            "moving_averages": "📉 Moving Averages",
            "Ede35e23": "💰 Valuation",
        }.get(x, x),
        index=0,
        help="Type d'affichage par défaut"
    )

with col_screen:
    default_screen = st.selectbox(
        "🎯 FILTRE PAR DÉFAUT",
        options=[
            "most_capitalized",
            "volume_leaders",
            "top_gainers",
            "top_losers",
            "ath",
            "atl",
            "above_52wk_high",
            "below_52wk_low",
            "monthly_gainers",
            "monthly_losers",
            "most_volatile",
            "unusual_volume",
            "overbought",
            "oversold",
        ],
        format_func=lambda x: {
            "most_capitalized": "💰 Plus grandes capitalisations",
            "volume_leaders": "📊 Leaders en volume",
            "top_gainers": "🟢 Top Gainers",
            "top_losers": "🔴 Top Losers",
            "ath": "🏔️ All-Time High",
            "atl": "🕳️ All-Time Low",
            "above_52wk_high": "📈 Au-dessus du plus haut 52 semaines",
            "below_52wk_low": "📉 En-dessous du plus bas 52 semaines",
            "monthly_gainers": "📅 Gainers du mois",
            "monthly_losers": "📅 Losers du mois",
            "most_volatile": "⚡ Plus volatiles",
            "unusual_volume": "🔥 Volume inhabituel",
            "overbought": "🔴 Surachat (RSI)",
            "oversold": "🟢 Survente (RSI)",
        }.get(x, x),
        index=0,
        help="Filtre par défaut du screener"
    )

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# WIDGET TRADINGVIEW - SCREENER
# =============================================
st.markdown(f"#### 📊 {market_title.upper()}")

# Différents widgets selon le type de marché
if selected_market == "crypto_mkt":
    # Widget spécifique pour Cryptocurrency Market
    tradingview_widget = f'''
    <div class="tradingview-widget-container" style="height:{widget_height}px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
        <div class="tradingview-widget-copyright">
            <a href="https://www.tradingview.com/markets/cryptocurrencies/" rel="noopener nofollow" target="_blank">
                <span class="blue-text">Cryptocurrency Markets</span>
            </a> by TradingView
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
        {{
            "width": "100%",
            "height": "100%",
            "defaultColumn": "{default_column}",
            "screener_type": "crypto_mkt",
            "displayCurrency": "USD",
            "colorTheme": "{color_theme}",
            "locale": "fr",
            "isTransparent": true
        }}
        </script>
    </div>
    '''
elif selected_market in ["forex", "crypto"]:
    # Widget pour Forex et Crypto pairs
    tradingview_widget = f'''
    <div class="tradingview-widget-container" style="height:{widget_height}px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
        <div class="tradingview-widget-copyright">
            <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                <span class="blue-text">Track all markets on TradingView</span>
            </a>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
        {{
            "width": "100%",
            "height": "100%",
            "defaultColumn": "{default_column}",
            "defaultScreen": "{default_screen}",
            "showToolbar": true,
            "locale": "fr",
            "market": "{selected_market}",
            "colorTheme": "{color_theme}",
            "isTransparent": true
        }}
        </script>
    </div>
    '''
else:
    # Widget standard pour les actions
    tradingview_widget = f'''
    <div class="tradingview-widget-container" style="height:{widget_height}px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
        <div class="tradingview-widget-copyright">
            <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
                <span class="blue-text">Track all markets on TradingView</span>
            </a>
        </div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
        {{
            "width": "100%",
            "height": "100%",
            "defaultColumn": "{default_column}",
            "defaultScreen": "{default_screen}",
            "showToolbar": true,
            "locale": "fr",
            "market": "{selected_market}",
            "colorTheme": "{color_theme}",
            "isTransparent": true
        }}
        </script>
    </div>
    '''

# Afficher le widget
st.components.v1.html(tradingview_widget, height=widget_height + 500, scrolling=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# SECTION HEATMAPS
# =============================================
st.markdown("### 🗺️ HEATMAPS")

heatmap_tabs = st.tabs(["📊 Stock Heatmap", "₿ Crypto Heatmap", "💱 Forex Heatmap", "📈 ETF Heatmap"])

with heatmap_tabs[0]:
    st.markdown("#### 📊 S&P 500 HEATMAP")
    stock_heatmap = f'''
    <div class="tradingview-widget-container" style="height:600px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
        {{
            "exchanges": [],
            "dataSource": "SPX500",
            "grouping": "sector",
            "blockSize": "market_cap_basic",
            "blockColor": "change",
            "locale": "fr",
            "symbolUrl": "",
            "colorTheme": "{color_theme}",
            "hasTopBar": true,
            "isDataSet498": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "isMonoSize": false,
            "width": "100%",
            "height": "100%"
        }}
        </script>
    </div>
    '''
    st.components.v1.html(stock_heatmap, height=650, scrolling=True)

with heatmap_tabs[1]:
    st.markdown("#### ₿ CRYPTO COINS HEATMAP")
    crypto_heatmap = f'''
    <div class="tradingview-widget-container" style="height:600px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js" async>
        {{
            "dataSource": "Crypto",
            "blockSize": "market_cap_calc",
            "blockColor": "change",
            "locale": "fr",
            "symbolUrl": "",
            "colorTheme": "{color_theme}",
            "hasTopBar": true,
            "isDataSetEnabled": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "width": "100%",
            "height": "100%"
        }}
        </script>
    </div>
    '''
    st.components.v1.html(crypto_heatmap, height=650, scrolling=True)

with heatmap_tabs[2]:
    st.markdown("#### 💱 FOREX HEATMAP")
    forex_heatmap = f'''
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-forex-heat-map.js" async>
        {{
            "width": "100%",
            "height": "100%",
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
            "isTransparent": true,
            "colorTheme": "{color_theme}",
            "locale": "fr"
        }}
        </script>
    </div>
    '''
    st.components.v1.html(forex_heatmap, height=550, scrolling=True)

with heatmap_tabs[3]:
    st.markdown("#### 📈 ETF HEATMAP")
    etf_heatmap = f'''
    <div class="tradingview-widget-container" style="height:600px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-etf-heatmap.js" async>
        {{
            "dataSource": "AllUSEtf",
            "blockSize": "aum_basic",
            "blockColor": "change",
            "grouping": "asset_class",
            "locale": "fr",
            "symbolUrl": "",
            "colorTheme": "{color_theme}",
            "hasTopBar": true,
            "isDataSetEnabled": true,
            "isZoomEnabled": true,
            "hasSymbolTooltip": true,
            "width": "100%",
            "height": "100%"
        }}
        </script>
    </div>
    '''
    st.components.v1.html(etf_heatmap, height=650, scrolling=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# SECTION FOREX CROSS RATES
# =============================================
st.markdown("### 💱 FOREX CROSS RATES")

forex_cross = f'''
<div class="tradingview-widget-container" style="height:400px;width:100%;">
    <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-forex-cross-rates.js" async>
    {{
        "width": "100%",
        "height": "100%",
        "currencies": [
            "EUR",
            "USD",
            "JPY",
            "GBP",
            "CHF",
            "AUD",
            "CAD",
            "NZD",
            "CNY",
            "HKD"
        ],
        "isTransparent": true,
        "colorTheme": "{color_theme}",
        "locale": "fr"
    }}
    </script>
</div>
'''
st.components.v1.html(forex_cross, height=450, scrolling=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# SECTION CALENDRIER ÉCONOMIQUE
# =============================================
st.markdown("### 📅 ECONOMIC CALENDAR")

economic_calendar = f'''
<div class="tradingview-widget-container" style="height:500px;width:100%;">
    <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
    {{
        "colorTheme": "{color_theme}",
        "isTransparent": true,
        "width": "100%",
        "height": "100%",
        "locale": "fr",
        "importanceFilter": "-1,0,1",
        "countryFilter": "us,eu,gb,jp,cn,ch,fr,de"
    }}
    </script>
</div>
'''
st.components.v1.html(economic_calendar, height=550, scrolling=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# LÉGENDE ET AIDE
# =============================================
with st.expander("📖 GUIDE D'UTILISATION DU SCREENER"):
    st.markdown("""
    ### 🔍 Comment utiliser le Market Screener
    
    **1. Sélection du marché:**
    - Choisissez le marché que vous souhaitez analyser (US, Europe, Crypto, Forex, etc.)
    - Chaque marché a ses propres caractéristiques et filtres disponibles
    
    **2. Colonnes disponibles:**
    - **Overview**: Vue d'ensemble avec prix, variation, volume
    - **Performance**: Performances sur différentes périodes (1j, 1s, 1m, 3m, 6m, 1a)
    - **Oscillators**: RSI, MACD, Stochastic, etc.
    - **Moving Averages**: SMA, EMA sur différentes périodes
    - **Valuation**: P/E, P/B, EV/EBITDA, etc.
    
    **3. Filtres prédéfinis:**
    - 🟢 **Top Gainers**: Actions avec les plus fortes hausses
    - 🔴 **Top Losers**: Actions avec les plus fortes baisses
    - 💰 **Most Capitalized**: Plus grandes capitalisations
    - 📊 **Volume Leaders**: Leaders en volume
    - ⚡ **Most Volatile**: Actions les plus volatiles
    - 🔴 **Overbought**: RSI > 70 (surachat)
    - 🟢 **Oversold**: RSI < 30 (survente)
    
    **4. Heatmaps:**
    - Visualisez rapidement les performances de tout un secteur
    - Taille = Capitalisation boursière
    - Couleur = Performance (vert = hausse, rouge = baisse)
    
    **5. Forex Cross Rates:**
    - Tableau croisé des taux de change entre devises majeures
    - Idéal pour le trading de paires de devises
    
    **6. Calendrier économique:**
    - Événements économiques à venir
    - Impact sur les marchés (faible, moyen, fort)
    """)

# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES EN TEMPS RÉEL • TRADINGVIEW<br>
        🔄 SCREENER INTERACTIF • MULTI-MARCHÉS
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 SYSTÈME OPÉRATIONNEL
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | TRADINGVIEW DATA | SYSTÈME OPÉRATIONNEL<br>
    MARKET SCREENER • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
