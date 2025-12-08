import streamlit as st
from datetime import datetime
from auth_utils import init_session_state

init_session_state()

if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Vous devez être connecté pour accéder à cette page.")
    st.stop()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Stock Screener",
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
    
    .stApp {
        background-color: #000000 !important;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
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
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    iframe {
        border: 2px solid #FFAA00 !important;
        background: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>🔍 BLOOMBERG ENS® - STOCK SCREENER</div>
    </div>
    <div>{current_time} UTC • TRADINGVIEW SCREENER</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# OPTIONS DE SCREENER
# =============================================
st.markdown("### 🎯 SÉLECTION DU SCREENER")

col1, col2 = st.columns([3, 9])

with col1:
    screener_type = st.selectbox(
        "Type de screener",
        options=[
            "Stock Screener (US)",
            "Stock Screener (Europe)",
            "Crypto Screener",
            "Forex Screener",
            "Top Gainers US",
            "Top Losers US"
        ],
        index=0
    )

# =============================================
# WIDGET TRADINGVIEW (Alternative fonctionnelle)
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

if screener_type == "Stock Screener (US)":
    st.markdown("#### 📊 US STOCK SCREENER")
    
    # TradingView Stock Screener Widget
    screener_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
      {
      "width": "100%",
      "height": "800",
      "defaultColumn": "overview",
      "screener_type": "crypto_mkt",
      "displayCurrency": "USD",
      "colorTheme": "dark",
      "locale": "en",
      "isTransparent": false
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.markdown(screener_html, unsafe_allow_html=True)

elif screener_type == "Stock Screener (Europe)":
    st.markdown("#### 🇪🇺 EUROPEAN STOCK SCREENER")
    
    screener_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
      {
      "width": "100%",
      "height": "800",
      "defaultColumn": "overview",
      "screener_type": "crypto_mkt",
      "displayCurrency": "EUR",
      "colorTheme": "dark",
      "locale": "en"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.markdown(screener_html, unsafe_allow_html=True)

elif screener_type == "Crypto Screener":
    st.markdown("#### ₿ CRYPTOCURRENCY SCREENER")
    
    screener_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
      {
      "width": "100%",
      "height": "800",
      "defaultColumn": "overview",
      "screener_type": "crypto_mkt",
      "displayCurrency": "USD",
      "colorTheme": "dark",
      "locale": "en"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.markdown(screener_html, unsafe_allow_html=True)

elif screener_type == "Forex Screener":
    st.markdown("#### 💱 FOREX SCREENER")
    
    screener_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
      {
      "width": "100%",
      "height": "800",
      "defaultColumn": "overview",
      "screener_type": "forex_mkt",
      "displayCurrency": "USD",
      "colorTheme": "dark",
      "locale": "en"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.markdown(screener_html, unsafe_allow_html=True)

elif screener_type == "Top Gainers US":
    st.markdown("#### 🚀 TOP GAINERS - US STOCKS")
    
    # Market Overview Widget avec filtre Gainers
    screener_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-hotlists.js" async>
      {
      "colorTheme": "dark",
      "dateRange": "1D",
      "exchange": "US",
      "showChart": true,
      "locale": "en",
      "largeChartUrl": "",
      "isTransparent": false,
      "showSymbolLogo": true,
      "showFloatingTooltip": true,
      "width": "100%",
      "height": "800",
      "plotLineColorGrowing": "rgba(41, 98, 255, 1)",
      "plotLineColorFalling": "rgba(41, 98, 255, 1)",
      "gridLineColor": "rgba(240, 243, 250, 0)",
      "scaleFontColor": "rgba(106, 109, 120, 1)",
      "belowLineFillColorGrowing": "rgba(41, 98, 255, 0.12)",
      "belowLineFillColorFalling": "rgba(41, 98, 255, 0.12)",
      "belowLineFillColorGrowingBottom": "rgba(41, 98, 255, 0)",
      "belowLineFillColorFallingBottom": "rgba(41, 98, 255, 0)",
      "symbolActiveColor": "rgba(41, 98, 255, 0.12)"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.markdown(screener_html, unsafe_allow_html=True)

elif screener_type == "Top Losers US":
    st.markdown("#### 📉 TOP LOSERS - US STOCKS")
    
    screener_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-hotlists.js" async>
      {
      "colorTheme": "dark",
      "dateRange": "1D",
      "exchange": "US",
      "showChart": true,
      "locale": "en",
      "width": "100%",
      "height": "800",
      "plotLineColorGrowing": "rgba(255, 82, 82, 1)",
      "plotLineColorFalling": "rgba(255, 82, 82, 1)",
      "gridLineColor": "rgba(240, 243, 250, 0)",
      "scaleFontColor": "rgba(106, 109, 120, 1)",
      "belowLineFillColorGrowing": "rgba(255, 82, 82, 0.12)",
      "belowLineFillColorFalling": "rgba(255, 82, 82, 0.12)",
      "belowLineFillColorGrowingBottom": "rgba(255, 82, 82, 0)",
      "belowLineFillColorFallingBottom": "rgba(255, 82, 82, 0)",
      "symbolActiveColor": "rgba(255, 82, 82, 0.12)",
      "tabs": [
        {
          "title": "Losers",
          "sortField": "change",
          "sortOrder": "asc"
        }
      ]
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.markdown(screener_html, unsafe_allow_html=True)

# Ajouter de l'espace pour le widget
st.markdown('<div style="height: 850px;"></div>', unsafe_allow_html=True)

# =============================================
# INFORMATIONS
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

st.info("""
📌 **À PROPOS DU SCREENER:**

✅ **Fonctionnalités:**
- Filtrage avancé multi-critères
- Données en temps réel
- Graphiques intégrés
- Analyse technique complète
- Export des résultats

🔍 **Comment utiliser:**
- Cliquez sur les en-têtes de colonnes pour trier
- Utilisez les filtres dans le widget
- Double-cliquez sur une action pour voir le détail
- Les données sont actualisées en temps réel

🌐 **Marchés disponibles:**
- 🇺🇸 Actions américaines (NYSE, NASDAQ, AMEX)
- 🇪🇺 Actions européennes
- ₿ Cryptomonnaies
- 💱 Forex

Powered by **TradingView** - Professional Charting Platform
""")

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | POWERED BY TRADINGVIEW | SYSTÈME OPÉRATIONNEL<br>
    STOCK SCREENER ACTIF • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
