import streamlit as st
import pandas as pd
import yfinance as yf
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
    
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 10px !important;
    }
    
    /* Style pour les lignes du tableau */
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: #000 !important;
    }
    
    /* Hover sur les lignes */
    .stDataFrame tbody tr:hover {
        background-color: #1a1a1a !important;
    }
    
    /* Style pour les heatmaps */
    .heatmap-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0px;
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
        <div>🔍 BLOOMBERG ENS® - MARKET HEATMAPS & SCREENER</div>
    </div>
    <div>{current_time} UTC • TRADINGVIEW</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# HEATMAPS TRADINGVIEW
# =============================================
st.markdown("### 🔥 MARKET HEATMAPS - REAL TIME")

# Sélecteur de heatmap
heatmap_tabs = st.tabs(["📊 STOCKS US", "🌍 ETF GLOBAL", "₿ CRYPTO"])

with heatmap_tabs[0]:
    st.markdown("#### 📊 US STOCKS HEATMAP")
    
    # Widget TradingView Stock Heatmap
    stock_heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
      {
        "exchanges": [],
        "dataSource": "SPX500",
        "grouping": "sector",
        "blockSize": "market_cap_basic",
        "blockColor": "change",
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": true,
        "isDataSetEnabled": true,
        "isZoomEnabled": true,
        "hasSymbolTooltip": true,
        "width": "100%",
        "height": "600"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(stock_heatmap, height=650)
    
    st.caption("""
    **📖 Comment lire la heatmap :**
    - 🟢 **Vert** : Performance positive
    - 🔴 **Rouge** : Performance négative
    - **Taille des blocs** : Proportionnelle à la capitalisation boursière
    - **Groupement** : Par secteur (Technology, Healthcare, Finance, etc.)
    """)

with heatmap_tabs[1]:
    st.markdown("#### 🌍 GLOBAL ETF HEATMAP")
    
    # Widget TradingView ETF Heatmap
    etf_heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-etf-heatmap.js" async>
      {
        "dataSource": "AllUSEtf",
        "blockSize": "aum",
        "blockColor": "change",
        "grouping": "asset_class",
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": true,
        "isDataSetEnabled": true,
        "isZoomEnabled": true,
        "hasSymbolTooltip": true,
        "width": "100%",
        "height": "600"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(etf_heatmap, height=650)
    
    st.caption("""
    **📖 Comment lire la heatmap :**
    - 🟢 **Vert** : Performance positive
    - 🔴 **Rouge** : Performance négative
    - **Taille des blocs** : Proportionnelle aux actifs sous gestion (AUM)
    - **Groupement** : Par classe d'actifs (Equity, Bond, Commodity, etc.)
    """)

with heatmap_tabs[2]:
    st.markdown("#### ₿ CRYPTOCURRENCY HEATMAP")
    
    # Widget TradingView Crypto Heatmap
    crypto_heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js" async>
      {
        "dataSource": "Crypto",
        "blockSize": "market_cap_calc",
        "blockColor": "change",
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": true,
        "isDataSetEnabled": true,
        "isZoomEnabled": true,
        "hasSymbolTooltip": true,
        "width": "100%",
        "height": "600"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(crypto_heatmap, height=650)
    
    st.caption("""
    **📖 Comment lire la heatmap :**
    - 🟢 **Vert** : Performance positive
    - 🔴 **Rouge** : Performance négative
    - **Taille des blocs** : Proportionnelle à la capitalisation boursière
    - **Principales cryptos** : BTC, ETH, BNB, SOL, XRP, ADA, DOGE, etc.)
    """)

st.markdown('<hr style="border-color: #FFAA00; margin: 30px 0;">', unsafe_allow_html=True)

# =============================================
# SCREENER TRADINGVIEW
# =============================================
st.markdown("### 🔍 STOCK SCREENER - TRADINGVIEW")

st.markdown("""
<div style="color:#666;font-size:10px;margin:10px 0 20px 0;">
    📊 Screener professionnel TradingView • Filtres avancés • Données temps réel • Tri multi-critères
</div>
""", unsafe_allow_html=True)

# Widget TradingView Stock Screener
screener_widget = """
<!DOCTYPE html>
<html>
<head>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        html, body {
            background-color: #000 !important;
            height: 100%;
            width: 100%;
            overflow: hidden;
        }
        .tradingview-widget-container {
            width: 100% !important;
            height: 100vh !important;
            background-color: #000 !important;
        }
    </style>
</head>
<body>
    <div style="background:#000;border:2px solid #FFAA00;padding:20px;margin:0;height:100vh;display:flex;flex-direction:column;">
        <div style="background:#FFAA00;color:#000;padding:10px 15px;font-weight:bold;font-size:14px;margin:-20px -20px 20px -20px;text-transform:uppercase;letter-spacing:2px;flex-shrink:0;">
            🔍 STOCK SCREENER - TRADINGVIEW
        </div>
        
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="flex-grow:1;">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js" async>
            {
              "width": "100%",
              "height": "100%",
              "defaultColumn": "overview",
              "defaultScreen": "general",
              "market": "america",
              "showToolbar": true,
              "colorTheme": "dark",
              "locale": "en",
              "isTransparent": false
            }
            </script>
        </div>
        <!-- TradingView Widget END -->
        
        <div style="background:#111;color:#666;padding:8px;font-size:9px;text-align:center;flex-shrink:0;margin:-20px;margin-top:10px;">
            💡 Cliquez sur les colonnes pour trier • Utilisez les filtres en haut pour affiner votre recherche • Double-cliquez sur une ligne pour voir le graphique
        </div>
    </div>
</body>
</html>
"""

st.components.v1.html(screener_widget, height=1000, scrolling=False)

st.markdown("""
<div style="color:#666;font-size:9px;margin-top:15px;text-align:center;">
    📊 Source: TradingView Professional Screener • Données temps réel • 
    Filtrez par secteur, capitalisation, P/E, dividendes, performance et plus encore
</div>
""", unsafe_allow_html=True)

# =============================================
# AIDE
# =============================================
with st.expander("📖 AIDE - COMMENT UTILISER LE SCREENER TRADINGVIEW"):
    st.markdown("""
    **🔍 UTILISATION DU SCREENER:**
    
    **1. FILTRES RAPIDES (Barre du haut)**
    - Cliquez sur les boutons pour filtrer par secteur, capitalisation, etc.
    - Combinez plusieurs filtres pour affiner votre recherche
    
    **2. COLONNES PERSONNALISABLES**
    - Cliquez sur les en-têtes de colonnes pour trier
    - Utilisez le menu (⚙️) pour ajouter/retirer des colonnes
    - Colonnes disponibles : P/E, EPS, Market Cap, Volume, etc.
    
    **3. FILTRES AVANCÉS**
    - Cliquez sur "Add filter" pour créer des critères personnalisés
    - Exemples : P/E < 20, Div Yield > 3%, Market Cap > 10B
    
    **4. SAUVEGARDER VOS RECHERCHES**
    - Les filtres sont sauvegardés localement dans votre navigateur
    - Utilisez les presets pour accéder rapidement à vos recherches favorites
    
    **📊 INDICATEURS CLÉS:**
    
    - **P/E Ratio** : Prix / Bénéfice par action (< 15 = sous-évalué)
    - **EPS** : Bénéfice par action
    - **Market Cap** : Capitalisation boursière
    - **Volume** : Volume d'échanges quotidien
    - **Div Yield** : Rendement du dividende (% annuel)
    - **Change %** : Variation sur la période sélectionnée
    - **52W High/Low** : Plus haut/bas sur 52 semaines
    
    **💡 ASTUCES:**
    
    - Double-cliquez sur une ligne pour voir le graphique complet
    - Utilisez Ctrl/Cmd + Click pour sélectionner plusieurs lignes
    - Exportez vos résultats via le menu (⚙️) → Export
    """)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | TRADINGVIEW WIDGETS | SYSTÈME OPÉRATIONNEL<br>
    HEATMAPS & SCREENER ACTIFS • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
