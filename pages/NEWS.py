# pages/NEWS.py
# Bloomberg Terminal - News Feed avec Finnhub API

import streamlit as st
import requests
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh

# =============================================
# CONFIGURATION FINNHUB
# =============================================
FINNHUB_API_KEY = "d14re49r01qop9mf2algd14re49r01qop9mf2am0"

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
    
    .news-category {
        background: #00FFFF;
        color: #000;
        padding: 2px 8px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .news-summary {
        color: #AAA;
        font-size: 11px;
        line-height: 1.5;
        margin-top: 8px;
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
# FONCTIONS FINNHUB
# =============================================
@st.cache_data(ttl=60)
def get_market_news(category="general"):
    """Récupère les news générales du marché via Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Erreur Finnhub: {e}")
        return []

@st.cache_data(ttl=60)
def get_company_news(ticker, days_back=7):
    """Récupère les news d'une entreprise spécifique via Finnhub"""
    try:
        today = datetime.now()
        from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Erreur Finnhub: {e}")
        return []

def format_timestamp(timestamp):
    """Convertit un timestamp en date lisible"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return "Date inconnue"

def display_news_card(news_item, ticker="", show_summary=True):
    """Affiche une carte de news style Bloomberg"""
    headline = news_item.get('headline', 'Sans titre')
    url = news_item.get('url', '#')
    source = news_item.get('source', 'Source inconnue')
    timestamp = news_item.get('datetime', 0)
    summary = news_item.get('summary', '')
    category = news_item.get('category', '')
    image = news_item.get('image', '')
    
    # Image si disponible
    img_html = ""
    if image:
        img_html = f'<img src="{image}" style="width:120px;height:80px;object-fit:cover;float:right;margin-left:15px;border:1px solid #333;">'
    
    # Badge ticker ou catégorie
    badge = ""
    if ticker:
        badge = f'<span class="news-ticker">{ticker}</span>'
    elif category:
        badge = f'<span class="news-category">{category.upper()}</span>'
    
    # Summary
    summary_html = ""
    if show_summary and summary:
        short_summary = summary[:200] + "..." if len(summary) > 200 else summary
        summary_html = f'<div class="news-summary">{short_summary}</div>'
    
    st.markdown(f"""
    <div class="news-card">
        {img_html}
        <div>
            {badge}
            <span class="news-meta"><span class="news-source">{source}</span> • {format_timestamp(timestamp)}</span>
        </div>
        <div class="news-title"><a href="{url}" target="_blank">{headline}</a></div>
        {summary_html}
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
    <div>{current_time} UTC • FINNHUB API • AUTO-REFRESH: 60s</div>
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
    st.markdown("### 🌍 GLOBAL MARKET NEWS - FINNHUB")
    
    # Sélection de catégorie
    col_cat, col_info = st.columns([2, 4])
    
    with col_cat:
        category = st.selectbox(
            "Catégorie",
            options=["general", "forex", "crypto", "merger"],
            format_func=lambda x: {
                "general": "📊 GENERAL MARKET",
                "forex": "💱 FOREX",
                "crypto": "₿ CRYPTO",
                "merger": "🤝 MERGERS & ACQUISITIONS"
            }.get(x, x)
        )
    
    with col_info:
        st.markdown(f"""
        <div style="color:#666;font-size:10px;padding:15px 0;">
            📡 NEWS EN TEMPS RÉEL • FINNHUB API • RAFRAÎCHISSEMENT AUTO: 60 SEC
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Récupérer les news
    with st.spinner("📡 Chargement des news..."):
        market_news = get_market_news(category)
    
    if market_news:
        # Stats
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("ARTICLES", len(market_news))
        with col_s2:
            st.metric("CATÉGORIE", category.upper())
        with col_s3:
            st.metric("MAJ", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Afficher les news
        st.markdown(f'<div class="category-header">🕐 LATEST NEWS - {len(market_news)} ARTICLES</div>', unsafe_allow_html=True)
        
        for news in market_news[:50]:
            display_news_card(news, show_summary=True)
    else:
        st.warning(⚠️ Aucune news disponible pour le moment")

# =============================================
# ONGLET 2 : SEARCH TICKER
# =============================================
with tab_search:
    st.markdown("### 🔍 SEARCH NEWS BY TICKER")
    
    st.markdown("""
    <div class="search-box">
        <div style="color:#FFAA00;font-size:12px;margin-bottom:10px;">
            Entrez un symbole ticker US pour rechercher ses actualités (ex: AAPL, MSFT, TSLA)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_days, col_btn = st.columns([3, 1, 1])
    
    with col_input:
        search_ticker = st.text_input(
            "Ticker",
            placeholder="Ex: AAPL, MSFT, GOOGL, TSLA...",
            label_visibility="collapsed",
            key="search_input"
        )
    
    with col_days:
        days_back = st.selectbox(
            "Période",
            options=[7, 14, 30, 60, 90],
            format_func=lambda x: f"{x} jours",
            label_visibility="collapsed"
        )
    
    with col_btn:
        search_btn = st.button("🔍 SEARCH", use_container_width=True)
    
    # Exemples de tickers
    st.markdown("""
    <div style="color:#666;font-size:10px;margin:10px 0;">
        <b>EXEMPLES:</b> AAPL • MSFT • GOOGL • TSLA • NVDA • META • AMZN • JPM • BAC • XOM • JNJ • V • WMT • DIS • NFLX
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Recherche
    if search_ticker:
        ticker_clean = search_ticker.upper().strip()
        
        with st.spinner(f"📡 Recherche des news pour {ticker_clean}..."):
            company_news = get_company_news(ticker_clean, days_back)
        
        if company_news:
            # Stats
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("TICKER", ticker_clean)
            with col_s2:
                st.metric("ARTICLES", len(company_news))
            with col_s3:
                st.metric("PÉRIODE", f"{days_back} jours")
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="category-header">📊 {ticker_clean} - {len(company_news)} NEWS</div>', unsafe_allow_html=True)
            
            for news in company_news[:50]:
                display_news_card(news, ticker=ticker_clean, show_summary=True)
        else:
            st.warning(f"⚠️ Aucune news trouvée pour {ticker_clean}. Vérifiez le symbole (tickers US uniquement).")
    else:
        st.markdown("""
        <div style="text-align:center;padding:50px;color:#666;">
            <div style="font-size:40px;margin-bottom:20px;">🔍</div>
            <div style="font-size:14px;">Entrez un ticker ci-dessus pour rechercher ses actualités</div>
            <div style="font-size:11px;margin-top:10px;color:#444;">Note: Finnhub supporte principalement les tickers US</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | FINNHUB API | SYSTÈME OPÉRATIONNEL<br>
    AUTO-REFRESH: 60 SECONDES • DERNIÈRE MAJ: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
