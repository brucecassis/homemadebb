# pages/NEWS.py
# Bloomberg Terminal - News Feed avec Yahoo Finance

import streamlit as st
import yfinance as yf
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

# =============================================
# AUTO-REFRESH TOUTES LES 60 SECONDES
# =============================================
count = st_autorefresh(interval=60000, limit=None, key="news_refresh")

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - News",
    page_icon="📰",
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
    
    body, .main, .stApp {
        font-family: 'Courier New', monospace;
        background: #000 !important;
        color: #FFAA00;
        font-size: 12px;
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
    
    .stMultiSelect > div {
        background-color: #111 !important;
        border: 1px solid #FFAA00 !important;
        color: #FFAA00 !important;
    }
    
    .stSelectbox > div {
        background-color: #111 !important;
        border: 1px solid #FFAA00 !important;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    /* Style des news */
    .news-card {
        background: #111;
        border: 1px solid #333;
        border-left: 4px solid #FFAA00;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s;
    }
    
    .news-card:hover {
        border-left-color: #00FF00;
        background: #1a1a1a;
    }
    
    .news-title {
        color: #FFAA00;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .news-title a {
        color: #FFAA00;
        text-decoration: none;
    }
    
    .news-title a:hover {
        color: #00FF00;
        text-decoration: underline;
    }
    
    .news-meta {
        color: #666;
        font-size: 10px;
        margin-bottom: 8px;
    }
    
    .news-source {
        color: #00FFFF;
        font-weight: bold;
    }
    
    .news-ticker {
        background: #FFAA00;
        color: #000;
        padding: 2px 6px;
        font-size: 9px;
        font-weight: bold;
        margin-right: 5px;
    }
    
    .news-summary {
        color: #CCC;
        font-size: 11px;
        line-height: 1.5;
    }
    
    .category-header {
        background: #FFAA00;
        color: #000;
        padding: 8px 15px;
        font-weight: bold;
        font-size: 12px;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    hr {
        border-color: #333;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# FONCTION RÉCUPÉRATION NEWS
# =============================================
@st.cache_data(ttl=60)
def get_ticker_news(ticker):
    """Récupère les news d'un ticker via Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except Exception as e:
        return []

def format_timestamp(timestamp):
    """Convertit un timestamp en date lisible"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return "Date inconnue"

def display_news_card(news_item, ticker=""):
    """Affiche une carte de news style Bloomberg"""
    title = news_item.get('title', 'Sans titre')
    link = news_item.get('link', '#')
    publisher = news_item.get('publisher', 'Source inconnue')
    timestamp = news_item.get('providerPublishTime', 0)
    
    # Thumbnail si disponible
    thumbnail = ""
    if 'thumbnail' in news_item and news_item['thumbnail']:
        try:
            thumb_url = news_item['thumbnail']['resolutions'][0]['url']
            thumbnail = f'<img src="{thumb_url}" style="width:100px;height:60px;object-fit:cover;float:right;margin-left:10px;border:1px solid #333;">'
        except:
            pass
    
    st.markdown(f"""
    <div class="news-card">
        {thumbnail}
        <span class="news-ticker">{ticker}</span>
        <span class="news-meta"><span class="news-source">{publisher}</span> • {format_timestamp(timestamp)}</span>
        <div class="news-title"><a href="{link}" target="_blank">{title}</a></div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® | NEWS TERMINAL</div>
        <a href="accueil.html" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • AUTO-REFRESH: 60s</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# SÉLECTION DES TICKERS
# =============================================
st.markdown("### 📰 MARKET NEWS FEED")

col_select, col_cat = st.columns([3, 1])

with col_select:
    # Catégories prédéfinies
    ticker_categories = {
        "🇺🇸 US TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "🇺🇸 US FINANCE": ["JPM", "BAC", "GS", "MS", "WFC"],
        "🇺🇸 US INDICES": ["SPY", "QQQ", "DIA", "IWM"],
        "🇪🇺 EUROPE": ["MC.PA", "OR.PA", "SAP", "ASML", "NESN.SW"],
        "₿ CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "⛽ ENERGY": ["XOM", "CVX", "COP", "TTE.PA"],
        "💊 PHARMA": ["JNJ", "PFE", "UNH", "LLY", "ABBV"]
    }
    
    selected_category = st.selectbox(
        "Catégorie",
        options=list(ticker_categories.keys()),
        index=0
    )

with col_cat:
    max_news = st.selectbox(
        "News par ticker",
        options=[5, 10, 15, 20],
        index=1
    )

# Tickers de la catégorie sélectionnée
default_tickers = ticker_categories[selected_category]

selected_tickers = st.multiselect(
    "Sélectionnez les tickers à suivre",
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX",
             "JPM", "BAC", "GS", "MS", "WFC", "C",
             "XOM", "CVX", "COP", "SLB", "TTE.PA",
             "JNJ", "PFE", "UNH", "LLY", "ABBV", "MRK",
             "MC.PA", "OR.PA", "BNP.PA", "SAP", "ASML", "NESN.SW",
             "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
             "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV"],
    default=default_tickers,
    help="Sélectionnez jusqu'à 10 tickers"
)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# AFFICHAGE DES NEWS
# =============================================
if selected_tickers:
    
    # Option d'affichage
    view_mode = st.radio(
        "Mode d'affichage",
        options=["📋 Par ticker", "🕐 Chronologique"],
        horizontal=True
    )
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    all_news = []
    
    with st.spinner("📡 Chargement des news..."):
        for ticker in selected_tickers[:10]:
            news_list = get_ticker_news(ticker)
            for news in news_list[:max_news]:
                news['_ticker'] = ticker
                all_news.append(news)
    
    if view_mode == "📋 Par ticker":
        # Affichage par ticker
        for ticker in selected_tickers[:10]:
            ticker_news = [n for n in all_news if n['_ticker'] == ticker]
            
            if ticker_news:
                st.markdown(f'<div class="category-header">📊 {ticker} - {len(ticker_news)} NEWS</div>', unsafe_allow_html=True)
                
                for news in ticker_news:
                    display_news_card(news, ticker)
    
    else:
        # Affichage chronologique
        # Trier par date décroissante
        all_news_sorted = sorted(
            all_news,
            key=lambda x: x.get('providerPublishTime', 0),
            reverse=True
        )
        
        st.markdown(f'<div class="category-header">🕐 TOUTES LES NEWS - {len(all_news_sorted)} ARTICLES</div>', unsafe_allow_html=True)
        
        for news in all_news_sorted:
            display_news_card(news, news['_ticker'])

else:
    st.warning("⚠️ Sélectionnez au moins un ticker pour voir les news")

# =============================================
# STATISTIQUES
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    st.markdown(f"""
    <div style="background:#111;border:1px solid #333;padding:15px;text-align:center;">
        <div style="color:#FFAA00;font-size:24px;font-weight:bold;">{len(selected_tickers)}</div>
        <div style="color:#666;font-size:10px;">TICKERS SUIVIS</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown(f"""
    <div style="background:#111;border:1px solid #333;padding:15px;text-align:center;">
        <div style="color:#00FF00;font-size:24px;font-weight:bold;">{len(all_news) if selected_tickers else 0}</div>
        <div style="color:#666;font-size:10px;">ARTICLES CHARGÉS</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown(f"""
    <div style="background:#111;border:1px solid #333;padding:15px;text-align:center;">
        <div style="color:#00FFFF;font-size:24px;font-weight:bold;">{datetime.now().strftime("%H:%M:%S")}</div>
        <div style="color:#666;font-size:10px;">DERNIÈRE MAJ</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE NEWS API | SYSTÈME OPÉRATIONNEL<br>
    AUTO-REFRESH: 60 SECONDES • DERNIÈRE MAJ: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
