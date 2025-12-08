import streamlit as st
from datetime import datetime
from auth_utils import init_session_state
import streamlit.components.v1 as components

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
        border-radius: 0px !important;
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
    <div>{current_time} UTC • POWERED BY TAKEPROFIT.COM</div>
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
            "Stock Screener (Global)",
            "Top Gainers",
            "Top Losers",
            "Most Active",
            "Dividend Stocks"
        ],
        index=0
    )

# =============================================
# WIDGET TAKEPROFIT
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

# Widget HTML TakeProfit Stock Screener
# Documentation: https://takeprofit.com/docs/guide/platform/stock-screener/Stock-screener-overview

if screener_type == "Stock Screener (US)":
    st.markdown("#### 📊 US STOCK SCREENER")
    
    # Widget TakeProfit pour US stocks
    takeprofit_html = """
    <!-- TakeProfit.com Widget BEGIN -->
    <script type="text/javascript" src="https://files.takeprofit.com/tools/widgets/stock_screener/js/init.js"></script>
    <div 
        class="tp-stock-screener" 
        data-countries="US"
        data-theme="dark"
        data-width="100%"
        data-height="800"
    ></div>
    <!-- TakeProfit.com Widget END -->
    """
    
    components.html(takeprofit_html, height=850, scrolling=True)

elif screener_type == "Stock Screener (Global)":
    st.markdown("#### 🌍 GLOBAL STOCK SCREENER")
    
    takeprofit_html = """
    <!-- TakeProfit.com Widget BEGIN -->
    <script type="text/javascript" src="https://files.takeprofit.com/tools/widgets/stock_screener/js/init.js"></script>
    <div 
        class="tp-stock-screener" 
        data-countries="US,GB,DE,FR,IT,ES,CH,NL"
        data-theme="dark"
        data-width="100%"
        data-height="800"
    ></div>
    <!-- TakeProfit.com Widget END -->
    """
    
    components.html(takeprofit_html, height=850, scrolling=True)

elif screener_type == "Top Gainers":
    st.markdown("#### 🚀 TOP GAINERS TODAY")
    
    takeprofit_html = """
    <!-- TakeProfit.com Widget BEGIN -->
    <script type="text/javascript" src="https://files.takeprofit.com/tools/widgets/stock_screener/js/init.js"></script>
    <div 
        class="tp-stock-screener" 
        data-countries="US"
        data-filter="gainers"
        data-theme="dark"
        data-width="100%"
        data-height="800"
    ></div>
    <!-- TakeProfit.com Widget END -->
    """
    
    components.html(takeprofit_html, height=850, scrolling=True)

elif screener_type == "Top Losers":
    st.markdown("#### 📉 TOP LOSERS TODAY")
    
    takeprofit_html = """
    <!-- TakeProfit.com Widget BEGIN -->
    <script type="text/javascript" src="https://files.takeprofit.com/tools/widgets/stock_screener/js/init.js"></script>
    <div 
        class="tp-stock-screener" 
        data-countries="US"
        data-filter="losers"
        data-theme="dark"
        data-width="100%"
        data-height="800"
    ></div>
    <!-- TakeProfit.com Widget END -->
    """
    
    components.html(takeprofit_html, height=850, scrolling=True)

elif screener_type == "Most Active":
    st.markdown("#### 🔥 MOST ACTIVE STOCKS")
    
    takeprofit_html = """
    <!-- TakeProfit.com Widget BEGIN -->
    <script type="text/javascript" src="https://files.takeprofit.com/tools/widgets/stock_screener/js/init.js"></script>
    <div 
        class="tp-stock-screener" 
        data-countries="US"
        data-filter="active"
        data-theme="dark"
        data-width="100%"
        data-height="800"
    ></div>
    <!-- TakeProfit.com Widget END -->
    """
    
    components.html(takeprofit_html, height=850, scrolling=True)

elif screener_type == "Dividend Stocks":
    st.markdown("#### 💰 DIVIDEND PAYING STOCKS")
    
    takeprofit_html = """
    <!-- TakeProfit.com Widget BEGIN -->
    <script type="text/javascript" src="https://files.takeprofit.com/tools/widgets/stock_screener/js/init.js"></script>
    <div 
        class="tp-stock-screener" 
        data-countries="US"
        data-filter="dividend"
        data-theme="dark"
        data-width="100%"
        data-height="800"
    ></div>
    <!-- TakeProfit.com Widget END -->
    """
    
    components.html(takeprofit_html, height=850, scrolling=True)

# =============================================
# INFORMATIONS
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

st.info("""
📌 **À PROPOS DU SCREENER:**

✅ **Fonctionnalités:**
- Filtrage avancé par secteur, capitalisation, P/E ratio, etc.
- Données en temps réel
- Export des résultats
- Graphiques intégrés
- Analyse technique

🔍 **Navigation:**
- Utilisez les filtres dans le widget pour affiner votre recherche
- Cliquez sur une action pour voir les détails
- Les résultats sont mis à jour automatiquement

🌐 **Marchés disponibles:**
- 🇺🇸 États-Unis (NYSE, NASDAQ, AMEX)
- 🇬🇧 Royaume-Uni (LSE)
- 🇩🇪 Allemagne (XETRA)
- 🇫🇷 France (Euronext Paris)
- 🇨🇭 Suisse (SIX)
- Et plus...

Powered by **TakeProfit.com** - Professional Trading Tools
""")

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | POWERED BY TAKEPROFIT.COM | SYSTÈME OPÉRATIONNEL<br>
    STOCK SCREENER ACTIF • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
