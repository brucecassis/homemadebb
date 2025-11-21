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
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body, .main, .stApp {
        font-family: 'Courier New', monospace;
        background: #000 !important;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .block-container { padding: 0rem 1rem !important; }
    
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
        border-radius: 0px !important;
        font-size: 10px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 2px solid #FFAA00 !important;
        border-radius: 0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #111;
        border-bottom: 2px solid #FFAA00;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #222;
        color: #FFAA00;
        border: 1px solid #333;
        border-bottom: none;
        padding: 10px 30px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 12px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
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
        font-size: 13px;
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
        margin-bottom: 5px;
    }
    
    .news-source {
        color: #00FFFF;
        font-weight: bold;
    }
    
    .news-ticker {
        background: #FFAA00;
        color: #000;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 8px;
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
    
    .search-box {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 20px;
        margin: 20px 0;
    }
    
    hr { border-color: #333; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS
# =============================================
@st.cache_data(ttl=60)
def get_ticker_news(ticker):
    """Récupère les news d'un ticker via Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news if news else []
    except:
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
    
    st.markdown(f"""
    <div class="news-card">
        <div>
            <span class="news-ticker">{ticker}</span>
            <span class="news-meta"><span class="news-source">{publisher}</span> • {format_timestamp(timestamp)}</span>
        </div>
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
# ONGLETS PRINCIPAUX
# =============================================
tab_global, tab_search = st.tabs(["📰 GLOBAL FEED", "🔍 SEARCH TICKER"])

# =============================================
# ONGLET 1 : GLOBAL FEED
# =============================================
with tab_global:
    st.markdown("### 🌍 GLOBAL MARKET NEWS")
    
    # Liste des tickers pour le feed global
    global_tickers = {
        "US TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "US FINANCE": ["JPM", "BAC", "GS"],
        "INDICES": ["SPY", "QQQ"],
        "CRYPTO": ["BTC-USD", "ETH-USD"],
        "EUROPE": ["MC.PA", "OR.PA", "ASML"],
        "ENERGY": ["XOM", "CVX"]
    }
    
    # Collecter toutes les news
    all_news = []
    
    with st.spinner("📡 Chargement du feed global..."):
        for category, tickers in global_tickers.items():
            for ticker in tickers:
                news_list = get_ticker_news(ticker)
                for news in news_list[:5]:  # 5 news max par ticker
                    news['_ticker'] = ticker
                    news['_category'] = category
                    all_news.append(news)
    
    # Trier par date décroissante
    all_news_sorted = sorted(
        all_news,
        key=lambda x: x.get('providerPublishTime', 0),
        reverse=True
    )
    
    # Supprimer les doublons (même titre)
    seen_titles = set()
    unique_news = []
    for news in all_news_sorted:
        title = news.get('title', '')
        if title not in seen_titles:
            seen_titles.add(title)
            unique_news.append(news)
    
    # Stats
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("ARTICLES", len(unique_news))
    with col_s2:
        st.metric("TICKERS", sum(len(t) for t in global_tickers.values()))
    with col_s3:
        st.metric("MAJ", datetime.now().strftime("%H:%M:%S"))
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Afficher les news
    st.markdown(f'<div class="category-header">🕐 LATEST NEWS - {len(unique_news)} ARTICLES</div>', unsafe_allow_html=True)
    
    for news in unique_news[:50]:  # Limite à 50 articles
        display_news_card(news, news['_ticker'])

# =============================================
# ONGLET 2 : SEARCH TICKER
# =============================================
with tab_search:
    st.markdown("### 🔍 SEARCH NEWS BY TICKER")
    
    st.markdown("""
    <div class="search-box">
        <div style="color:#FFAA00;font-size:12px;margin-bottom:10px;">
            Entrez un symbole ticker pour rechercher ses actualités
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        search_ticker = st.text_input(
            "Ticker",
            placeholder="Ex: AAPL, MSFT, BTC-USD, MC.PA...",
            label_visibility="collapsed",
            key="search_input"
        )
    
    with col_btn:
        search_btn = st.button("🔍 SEARCH", use_container_width=True)
    
    # Exemples de tickers
    st.markdown("""
    <div style="color:#666;font-size:10px;margin:10px 0;">
        <b>EXEMPLES:</b> AAPL • MSFT • GOOGL • TSLA • NVDA • META • JPM • BTC-USD • ETH-USD • MC.PA • OR.PA • NESN.SW
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Recherche
    if search_ticker:
        ticker_clean = search_ticker.upper().strip()
        
        with st.spinner(f"📡 Recherche des news pour {ticker_clean}..."):
            search_news = get_ticker_news(ticker_clean)
        
        if search_news:
            # Info sur le ticker
            try:
                stock = yf.Ticker(ticker_clean)
                info = stock.info
                company_name = info.get('shortName', ticker_clean)
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                
                st.markdown(f"""
                <div style="background:#111;border:2px solid #FFAA00;padding:15px;margin-bottom:20px;">
                    <div style="color:#FFAA00;font-size:18px;font-weight:bold;">{ticker_clean}</div>
                    <div style="color:#FFF;font-size:12px;">{company_name}</div>
                    <div style="color:#00FF00;font-size:16px;margin-top:5px;">
                        ${current_price if isinstance(current_price, (int, float)) else 'N/A'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except:
                pass
            
            st.markdown(f'<div class="category-header">📊 {ticker_clean} - {len(search_news)} NEWS</div>', unsafe_allow_html=True)
            
            for news in search_news:
                display_news_card(news, ticker_clean)
        else:
            st.warning(f"⚠️ Aucune news trouvée pour {ticker_clean}. Vérifiez le symbole du ticker.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:50px;color:#666;">
            <div style="font-size:40px;margin-bottom:20px;">🔍</div>
            <div style="font-size:14px;">Entrez un ticker ci-dessus pour rechercher ses actualités</div>
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
