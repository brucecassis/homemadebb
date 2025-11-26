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
        <div>⬛ BLOOMBERG ENS® TERMINAL - OBLIGATION SCREENER</div>
    </div>
    <div>{current_time} UTC • TRADINGVIEW DATA</div>
</div>
""", unsafe_allow_html=True)


# =============================================
# TITRE DE LA PAGE
# =============================================
st.markdown("### 🔍 OBLIGATION SCREENER - TRADINGVIEW")

# =============================================
# WIDGET TRADINGVIEW - OBLIGATION SCREENER
# =============================================
obligation_screener = f'''
<div class="tradingview-widget-container" style="height:800px;width:100%;">
    <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%;"></div>
    <div class="tradingview-widget-copyright">
        <a href="https://www.tradingview.com/markets/bonds/" rel="noopener nofollow" target="_blank">
            <span class="blue-text">Bonds</span>
        </a> by TradingView
    </div>
    <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
    {{
        "width": "100%",
        "height": "100%",
        "defaultColumn": "overview",
        "screener_type": "bonds",
        "displayCurrency": "USD",
        "colorTheme": "dark",
        "locale": "fr",
        "isTransparent": true
    }}
    </script>
</div>
'''

# Afficher le widget
st.components.v1.html(obligation_screener, height=850, scrolling=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# LÉGENDE ET AIDE
# =============================================
with st.expander("📖 GUIDE D'UTILISATION DU SCREENER"):
    st.markdown("""
    ### 🔍 Comment utiliser le Obligation Screener

    **1. Sélection des filtres:**
    - Vous pouvez choisir les filtres pour affiner vos recherches.

    **2. Visualisation des données:**
    - Les données sont présentées sous forme de tableau.

    **3. Informations détaillées:**
    - Vous pouvez cliquer sur chaque ligne pour obtenir plus d'informations.
    """)

# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES EN TEMPS RÉEL • TRADINGVIEW<br>
        🔄 SCREENER INTERACTIF • OBLIGATIONS
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
    2025 BLOOMBERG ENS | TRADINGVIEW DATA |
