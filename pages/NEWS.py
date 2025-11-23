# pages/NEWS.py
# Bloomberg Terminal - News Feed avec Finnhub API + Calendrier Économique (Polygon.io)

import streamlit as st
import requests
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import pandas as pd

# =============================================
# CONFIGURATION APIs
# =============================================
FINNHUB_API_KEY = "d14re49r01qop9mf2algd14re49r01qop9mf2am0"
POLYGON_API_KEY = "F95xsJcPxjI5WHlyVfcWWTFy1mK9cfEi"

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
    
    .stTextInput > div > div > input, .stDateInput > div > div > input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 2px solid #FFAA00 !important;
        border-radius: 0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
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
    
    .event-card {
        background: #0a0a0a;
        border: 1px solid #333;
        border-left: 4px solid #00FFFF;
        padding: 12px 15px;
        margin: 8px 0;
        transition: all 0.3s;
    }
    
    .event-card:hover {
        border-left-color: #00FF00;
        background: #151515;
    }
    
    .event-date {
        background: #FFAA00;
        color: #000;
        padding: 4px 10px;
        font-weight: bold;
        font-size: 10px;
        margin-right: 15px;
        display: inline-block;
        min-width: 100px;
        text-align: center;
    }
    
    .event-title {
        color: #FFAA00;
        font-size: 12px;
        font-weight: bold;
        margin: 8px 0 4px 0;
    }
    
    .event-details {
        color: #888;
        font-size: 10px;
    }
    
    .ipo-card {
        background: #0a0a0a;
        border: 1px solid #333;
        border-left: 4px solid #FF00FF;
        padding: 12px 15px;
        margin: 8px 0;
        transition: all 0.3s;
    }
    
    .ipo-card:hover {
        border-left-color: #00FF00;
        background: #151515;
    }
    
    .ipo-ticker {
        background: #FF00FF;
        color: #FFF;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 10px;
        display: inline-block;
    }
    
    .ipo-name {
        color: #FFAA00;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    
    .ipo-details {
        color: #888;
        font-size: 10px;
        margin-top: 5px;
    }
    
    .earnings-card {
        background: #0a0a0a;
        border: 1px solid #333;
        border-left: 4px solid #00FF00;
        padding: 12px 15px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s;
    }
    
    .earnings-card:hover {
        border-left-color: #FFAA00;
        background: #151515;
    }
    
    .earnings-card.past {
        border-left-color: #666;
        opacity: 0.8;
    }
    
    .earnings-ticker {
        background: #00FF00;
        color: #000;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: bold;
        margin-right: 15px;
        min-width: 70px;
        text-align: center;
    }
    
    .earnings-ticker.past {
        background: #666;
        color: #FFF;
    }
    
    .earnings-info {
        flex: 1;
    }
    
    .earnings-name {
        color: #FFAA00;
        font-size: 12px;
        font-weight: bold;
    }
    
    .earnings-date {
        color: #888;
        font-size: 10px;
        margin-top: 3px;
    }
    
    .earnings-eps {
        color: #00FFFF;
        font-size: 10px;
        font-weight: bold;
        text-align: right;
        min-width: 200px;
    }
    
    .earnings-beat {
        color: #00FF00 !important;
    }
    
    .earnings-miss {
        color: #FF0000 !important;
    }
    
    .dividend-card {
        background: #0a0a0a;
        border: 1px solid #333;
        border-left: 4px solid #00FFFF;
        padding: 12px 15px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s;
    }
    
    .dividend-card:hover {
        border-left-color: #00FF00;
        background: #151515;
    }
    
    .dividend-amount {
        background: #00FFFF;
        color: #000;
        padding: 6px 12px;
        font-size: 13px;
        font-weight: bold;
        min-width: 80px;
        text-align: center;
    }
    
    .dividend-info {
        flex: 1;
        margin-left: 15px;
    }
    
    .dividend-date {
        color: #FFAA00;
        font-size: 11px;
        font-weight: bold;
    }
    
    .dividend-type {
        color: #888;
        font-size: 10px;
        margin-top: 3px;
    }
    
    hr { border-color: #333; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS FINNHUB - NEWS
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

# =============================================
# FONCTIONS FINNHUB - EARNINGS
# =============================================
@st.cache_data(ttl=1800)
def get_earnings_calendar_finnhub(from_date, to_date):
    """Calendrier résultats trimestriels via Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/calendar/earnings?from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {"earningsCalendar": []}
    except Exception as e:
        st.error(f"Erreur calendrier earnings: {e}")
        return {"earningsCalendar": []}

# =============================================
# FONCTIONS POLYGON.IO
# =============================================
@st.cache_data(ttl=3600)
def get_economic_events_polygon():
    """Calendrier économique via Polygon.io"""
    try:
        # Note: Polygon n'a pas d'endpoint direct pour les événements économiques
        # On peut utiliser les market holidays et status
        url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Erreur Polygon economic: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_market_holidays_polygon():
    """Jours fériés du marché via Polygon.io"""
    try:
        url = f"https://api.polygon.io/v1/marketstatus/upcoming?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Erreur Polygon holidays: {e}")
        return []

@st.cache_data(ttl=3600)
def get_dividends_polygon(ticker):
    """Dividendes via Polygon.io"""
    try:
        url = f"https://api.polygon.io/v3/reference/dividends?ticker={ticker}&limit=100&apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        return []
    except Exception as e:
        st.error(f"Erreur Polygon dividends: {e}")
        return []

@st.cache_data(ttl=3600)
def get_stock_splits_polygon(ticker):
    """Stock splits via Polygon.io"""
    try:
        url = f"https://api.polygon.io/v3/reference/splits?ticker={ticker}&limit=50&apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        return []
    except Exception as e:
        st.error(f"Erreur Polygon splits: {e}")
        return []

@st.cache_data(ttl=3600)
def get_ticker_details_polygon(ticker):
    """Détails d'un ticker via Polygon.io"""
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', {})
        return {}
    except Exception as e:
        return {}

# =============================================
# FONCTIONS UTILITAIRES
# =============================================
def format_timestamp(timestamp):
    """Convertit un timestamp en date lisible"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return "Date inconnue"

def format_date(date_str):
    """Formate une date au format DD/MM/YYYY"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except:
        return date_str

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
    <div>{current_time} UTC • FINNHUB + POLYGON.IO • AUTO-REFRESH: 60s</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# ONGLETS PRINCIPAUX
# =============================================
tab_global, tab_search, tab_calendar = st.tabs(["📰 GLOBAL FEED", "🔍 SEARCH TICKER", "📅 CALENDAR"])

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
        st.warning("⚠️ Aucune news disponible pour le moment")

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
# ONGLET 3 : CALENDAR
# =============================================
with tab_calendar:
    st.markdown("### 📅 ECONOMIC & CORPORATE CALENDAR")
    
    # Sous-onglets du calendrier
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 MARKET STATUS",
        "🚀 TICKER INFO",
        "💰 EARNINGS CALENDAR",
        "💵 DIVIDENDS & SPLITS"
    ])
    
    # ========== SOUS-ONGLET 1: MARKET STATUS (Polygon) ==========
    with sub_tab1:
        st.markdown("#### 📊 MARKET STATUS & HOLIDAYS")
        
        st.markdown("""
        <div style="color:#666;font-size:10px;margin:10px 0;">
            Statut du marché en temps réel et jours fériés à venir (Polygon.io)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown('<div class="category-header">🔴 MARKET STATUS NOW</div>', unsafe_allow_html=True)
            
            with st.spinner("📡 Chargement du statut du marché..."):
                market_status = get_economic_events_polygon()
            
            if market_status:
                market_open = market_status.get('market', 'unknown')
                server_time = market_status.get('serverTime', 'N/A')
                exchanges = market_status.get('exchanges', {})
                
                status_color = "#00FF00" if market_open == "open" else "#FF0000"
                status_text = "OPEN" if market_open == "open" else "CLOSED"
                
                st.markdown(f"""
                <div class="event-card">
                    <div style="text-align:center;">
                        <div style="font-size:40px;margin:20px 0;">
                            <span style="color:{status_color};">●</span>
                        </div>
                        <div style="color:#FFAA00;font-size:18px;font-weight:bold;margin-bottom:10px;">
                            MARKET IS {status_text}
                        </div>
                        <div style="color:#888;font-size:10px;">
                            Server Time: {server_time}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if exchanges:
                    st.markdown('<div style="color:#FFAA00;font-size:11px;margin-top:20px;font-weight:bold;">EXCHANGES STATUS:</div>', unsafe_allow_html=True)
                    for exchange, status in exchanges.items():
                        ex_status = "OPEN ✅" if status == "open" else "CLOSED ❌"
                        st.markdown(f"""
                        <div style="background:#111;padding:8px;margin:5px 0;border-left:3px solid #00FFFF;">
                            <span style="color:#00FFFF;font-weight:bold;">{exchange.upper()}</span>: {ex_status}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Impossible de récupérer le statut du marché")
        
        with col_m2:
            st.markdown('<div class="category-header">📅 UPCOMING MARKET HOLIDAYS</div>', unsafe_allow_html=True)
            
            with st.spinner("📡 Chargement des jours fériés..."):
                holidays = get_market_holidays_polygon()
            
            if holidays and isinstance(holidays, list):
                for holiday in holidays[:10]:
                    h_date = holiday.get('date', 'N/A')
                    h_name = holiday.get('name', 'Holiday')
                    h_status = holiday.get('status', 'closed')
                    
                    st.markdown(f"""
                    <div class="event-card">
                        <span class="event-date">{format_date(h_date) if h_date != 'N/A' else h_date}</span>
                        <div class="event-title">{h_name}</div>
                        <div class="event-details">Market Status: {h_status.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:30px;color:#666;">
                    <div style="font-size:30px;margin-bottom:15px;">📅</div>
                    <div style="font-size:12px;">Aucun jour férié à afficher</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ========== SOUS-ONGLET 2: TICKER INFO (Polygon) ==========
    with sub_tab2:
        st.markdown("#### 🚀 TICKER DETAILS & INFORMATION")
        
        st.markdown("""
        <div class="search-box">
            <div style="color:#FFAA00;font-size:12px;margin-bottom:10px;">
                Recherchez des informations détaillées sur un ticker US (Polygon.io)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns([4, 1])
        
        with col_t1:
            ticker_info_search = st.text_input(
                "Ticker",
                placeholder="Ex: AAPL, MSFT, TSLA...",
                label_visibility="collapsed",
                key="ticker_info"
            )
        
        with col_t2:
            ticker_search_btn = st.button("🔍 SEARCH", use_container_width=True, key="ticker_search")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        if ticker_info_search:
            ticker_clean = ticker_info_search.upper().strip()
            
            with st.spinner(f"📡 Recherche d'informations pour {ticker_clean}..."):
                ticker_details = get_ticker_details_polygon(ticker_clean)
            
            if ticker_details:
                name = ticker_details.get('name', 'N/A')
                market = ticker_details.get('market', 'N/A')
                locale = ticker_details.get('locale', 'N/A')
                primary_exchange = ticker_details.get('primary_exchange', 'N/A')
                ticker_type = ticker_details.get('type', 'N/A')
                active = ticker_details.get('active', False)
                currency = ticker_details.get('currency_name', 'N/A')
                cik = ticker_details.get('cik', 'N/A')
                composite_figi = ticker_details.get('composite_figi', 'N/A')
                share_class_figi = ticker_details.get('share_class_figi', 'N/A')
                market_cap = ticker_details.get('market_cap', 'N/A')
                phone = ticker_details.get('phone_number', 'N/A')
                address = ticker_details.get('address', {})
                description = ticker_details.get('description', 'N/A')
                homepage = ticker_details.get('homepage_url', 'N/A')
                total_employees = ticker_details.get('total_employees', 'N/A')
                list_date = ticker_details.get('list_date', 'N/A')
                
                # Affichage
                st.markdown(f"""
                <div class="ipo-card">
                    <div style="display:flex;align-items:center;margin-bottom:12px;">
                        <span class="ipo-ticker">{ticker_clean}</span>
                        <span class="ipo-name">{name}</span>
                    </div>
                    <div class="ipo-details">
                        <b>Type:</b> {ticker_type} • <b>Market:</b> {market} • <b>Exchange:</b> {primary_exchange}<br>
                        <b>Currency:</b> {currency} • <b>Active:</b> {'✅ Yes' if active else '❌ No'}<br>
                        <b>Listed Since:</b> {format_date(list_date) if list_date != 'N/A' else 'N/A'}<br>
                        <b>Market Cap:</b> ${market_cap:,} • <b>Employees:</b> {total_employees:,}<br>
                        <b>CIK:</b> {cik} • <b>Phone:</b> {phone}<br>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if description and description != 'N/A':
                    st.markdown(f"""
                    <div style="background:#111;border:1px solid #333;padding:15px;margin:15px 0;">
                        <div style="color:#FFAA00;font-weight:bold;margin-bottom:8px;">DESCRIPTION:</div>
                        <div style="color:#AAA;font-size:11px;line-height:1.6;">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if homepage and homepage != 'N/A':
                    st.markdown(f"""
                    <div style="margin:10px 0;">
                        <a href="{homepage}" target="_blank" style="color:#00FFFF;text-decoration:none;">
                            🌐 Visit Company Website →
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Adresse
                if address:
                    addr1 = address.get('address1', '')
                    city = address.get('city', '')
                    state = address.get('state', '')
                    postal = address.get('postal_code', '')
                    
                    if any([addr1, city, state]):
                        st.markdown(f"""
                        <div style="background:#0a0a0a;border:1px solid #333;padding:10px;margin:10px 0;">
                            <div style="color:#FFAA00;font-size:10px;font-weight:bold;">ADDRESS:</div>
                            <div style="color:#888;font-size:10px;">{addr1}, {city}, {state} {postal}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Aucune information trouvée pour {ticker_clean}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px;color:#666;">
                <div style="font-size:40px;margin-bottom:20px;">🔍</div>
                <div style="font-size:14px;">Entrez un ticker pour voir ses informations détaillées</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== SOUS-ONGLET 3: EARNINGS CALENDAR (Finnhub amélioré) ==========
    with sub_tab3:
        st.markdown("#### 💰 EARNINGS CALENDAR - QUARTERLY RESULTS")
        
        st.markdown("""
        <div style="color:#666;font-size:10px;margin:10px 0;">
            Calendrier des résultats trimestriels avec estimations et résultats réels (Finnhub)
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection de période personnalisée
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            from_date_earn = st.date_input(
                "Date de début",
                value=datetime.now() - timedelta(days=30),
                key="earn_from"
            )
        
        with col_e2:
            to_date_earn = st.date_input(
                "Date de fin",
                value=datetime.now() + timedelta(days=30),
                key="earn_to"
            )
        
        with col_e3:
            earn_refresh = st.button("🔄 REFRESH", use_container_width=True, key="earn_refresh")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        from_str = from_date_earn.strftime("%Y-%m-%d")
        to_str = to_date_earn.strftime("%Y-%m-%d")
        
        with st.spinner("📡 Chargement du calendrier des résultats..."):
            earn_data = get_earnings_calendar_finnhub(from_str, to_str)
        
        earnings_events = earn_data.get("earningsCalendar", [])
        
        if earnings_events:
            # Séparer passés et futurs
            today = datetime.now().date()
            past_earnings = []
            future_earnings = []
            
            for e in earnings_events:
                e_date_str = e.get('date', '')
                try:
                    e_date = datetime.strptime(e_date_str, "%Y-%m-%d").date()
                    if e_date < today:
                        past_earnings.append(e)
                    else:
                        future_earnings.append(e)
                except:
                    future_earnings.append(e)
            
            # Stats
            col_ea1, col_ea2, col_ea3, col_ea4 = st.columns(4)
            with col_ea1:
                st.metric("TOTAL", len(earnings_events))
            with col_ea2:
                st.metric("PASSÉS", len(past_earnings))
            with col_ea3:
                st.metric("À VENIR", len(future_earnings))
            with col_ea4:
                st.metric("MAJ", datetime.now().strftime("%H:%M"))
            
            st.markdown('<hr>', unsafe_allow_html=True)
            
            # Filtre d'affichage
            display_filter = st.radio(
                "Afficher",
                options=["Tous", "À venir uniquement", "Passés uniquement"],
                horizontal=True,
                key="earn_filter"
            )
            
            # Déterminer quoi afficher
            if display_filter == "À venir uniquement":
                to_display = future_earnings
                header_text = "📅 UPCOMING EARNINGS"
            elif display_filter == "Passés uniquement":
                to_display = past_earnings
                header_text = "📊 PAST EARNINGS"
            else:
                to_display = earnings_events
                header_text = "📈 ALL EARNINGS"
            
            st.markdown(f'<div class="category-header">{header_text} - {len(to_display)} RESULTS</div>', unsafe_allow_html=True)
            
            for earning in to_display[:100]:
                date = earning.get("date", "N/A")
                symbol = earning.get("symbol", "N/A")
                eps_actual = earning.get("epsActual")
                eps_estimate = earning.get("epsEstimate")
                hour = earning.get("hour", "N/A")
                quarter = earning.get("quarter", "N/A")
                revenue_actual = earning.get("revenueActual")
                revenue_estimate = earning.get("revenueEstimate")
                year = earning.get("year", "N/A")
                
                # Déterminer si passé
                is_past = False
                try:
                    e_date = datetime.strptime(date, "%Y-%m-%d").date()
                    is_past = e_date < today
                except:
                    pass
                
                past_class = "past" if is_past else ""
                
                # EPS Info avec comparaison
                eps_html = ""
                if eps_actual is not None and eps_estimate is not None:
                    diff = eps_actual - eps_estimate
                    beat_miss_class = "earnings-beat" if diff >= 0 else "earnings-miss"
                    beat_miss_text = f"(Beat: +${abs(diff):.2f})" if diff >= 0 else f"(Miss: -${abs(diff):.2f})"
                    eps_html = f"""
                    <span class="{beat_miss_class}">
                        EPS: ${eps_actual:.2f} vs ${eps_estimate:.2f} est. {beat_miss_text}
                    </span>
                    """
                elif eps_estimate is not None:
                    eps_html = f"EPS: ${eps_estimate:.2f} (est.)"
                elif eps_actual is not None:
                    eps_html = f"EPS: ${eps_actual:.2f} (actual)"
                else:
                    eps_html = "EPS: N/A"
                
                # Revenue Info avec comparaison
                rev_html = ""
                if revenue_actual is not None and revenue_estimate is not None:
                    diff_rev = revenue_actual - revenue_estimate
                    beat_miss_class_rev = "earnings-beat" if diff_rev >= 0 else "earnings-miss"
                    beat_miss_text_rev = f"(Beat: +${abs(diff_rev):.0f}M)" if diff_rev >= 0 else f"(Miss: -${abs(diff_rev):.0f}M)"
                    rev_html = f"""
                    <br><span class="{beat_miss_class_rev}">
                        REV: ${revenue_actual:.0f}M vs ${revenue_estimate:.0f}M est. {beat_miss_text_rev}
                    </span>
                    """
                elif revenue_estimate is not None:
                    rev_html = f"<br>REV: ${revenue_estimate:.0f}M (est.)"
                elif revenue_actual is not None:
                    rev_html = f"<br>REV: ${revenue_actual:.0f}M (actual)"
                
                st.markdown(f"""
                <div class="earnings-card {past_class}">
                    <div class="earnings-ticker {past_class}">{symbol}</div>
                    <div class="earnings-info">
                        <div class="earnings-name">Q{quarter} {year} EARNINGS</div>
                        <div class="earnings-date">📅 {format_date(date)} • {hour}</div>
                    </div>
                    <div class="earnings-eps">
                        {eps_html}
                        {rev_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Aucun résultat trimestriel dans la période sélectionnée")
    
    # ========== SOUS-ONGLET 4: DIVIDENDS & SPLITS (Polygon) ==========
    with sub_tab4:
        st.markdown("#### 💵 DIVIDENDS & STOCK SPLITS")
        
        st.markdown("""
        <div class="search-box">
            <div style="color:#FFAA00;font-size:12px;margin-bottom:10px;">
                Recherchez l'historique des dividendes et stock splits d'un ticker (Polygon.io)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_d1, col_d2 = st.columns([4, 1])
        
        with col_d1:
            div_ticker = st.text_input(
                "Ticker",
                placeholder="Ex: AAPL, MSFT, JNJ, KO...",
                label_visibility="collapsed",
                key="div_ticker"
            )
        
        with col_d2:
            div_search_btn = st.button("🔍 SEARCH", use_container_width=True, key="div_search")
        
        st.markdown("""
        <div style="color:#666;font-size:10px;margin:10px 0;">
            <b>EXEMPLES DE DIVIDEND ARISTOCRATS:</b> JNJ • PG • KO • PEP • MCD • WMT • XOM • CVX • MMM • CAT
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        if div_ticker:
            ticker_div_clean = div_ticker.upper().strip()
            
            # Onglets dividendes et splits
            div_tab, split_tab = st.tabs(["💰 DIVIDENDS", "📊 STOCK SPLITS"])
            
            with div_tab:
                with st.spinner(f"📡 Chargement des dividendes pour {ticker_div_clean}..."):
                    dividends = get_dividends_polygon(ticker_div_clean)
                
                if dividends:
                    col_dv1, col_dv2, col_dv3 = st.columns(3)
                    with col_dv1:
                        st.metric("TICKER", ticker_div_clean)
                    with col_dv2:
                        st.metric("PAIEMENTS", len(dividends))
                    with col_dv3:
                        total_div = sum([d.get('cash_amount', 0) for d in dividends if d.get('cash_amount')])
                        st.metric("TOTAL", f"${total_div:.2f}")
                    
                    st.markdown('<hr>', unsafe_allow_html=True)
                    st.markdown(f'<div class="category-header">💰 {ticker_div_clean} - DIVIDEND HISTORY ({len(dividends)})</div>', unsafe_allow_html=True)
                    
                    for div in dividends[:50]:
                        amount = div.get("cash_amount", 0)
                        currency = div.get("currency", "USD")
                        ex_date = div.get("ex_dividend_date", "N/A")
                        payment_date = div.get("pay_date", "N/A")
                        record_date = div.get("record_date", "N/A")
                        declaration_date = div.get("declaration_date", "N/A")
                        frequency = div.get("frequency", "N/A")
                        div_type = div.get("dividend_type", "N/A")
                        
                        st.markdown(f"""
                        <div class="dividend-card">
                            <div class="dividend-amount">${amount:.4f}</div>
                            <div class="dividend-info">
                                <div class="dividend-date">💰 Payment: {format_date(payment_date) if payment_date != 'N/A' else 'N/A'}</div>
                                <div class="dividend-type">
                                    Ex-Date: {format_date(ex_date) if ex_date != 'N/A' else 'N/A'} • 
                                    Record: {format_date(record_date) if record_date != 'N/A' else 'N/A'} • 
                                    Type: {div_type} • Freq: {frequency}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ Aucun dividende trouvé pour {ticker_div_clean}")
            
            with split_tab:
                with st.spinner(f"📡 Chargement des splits pour {ticker_div_clean}..."):
                    splits = get_stock_splits_polygon(ticker_div_clean)
                
                if splits:
                    st.metric("STOCK SPLITS", len(splits))
                    st.markdown('<hr>', unsafe_allow_html=True)
                    st.markdown(f'<div class="category-header">📊 {ticker_div_clean} - STOCK SPLITS HISTORY</div>', unsafe_allow_html=True)
                    
                    for split in splits:
                        exec_date = split.get("execution_date", "N/A")
                        split_from = split.get("split_from", 1)
                        split_to = split.get("split_to", 1)
                        
                        ratio = f"{split_to}:{split_from}"
                        ratio_decimal = split_to / split_from if split_from > 0 else 0
                        
                        st.markdown(f"""
                        <div class="event-card">
                            <span class="event-date">{format_date(exec_date) if exec_date != 'N/A' else 'N/A'}</span>
                            <div class="event-title">STOCK SPLIT: {ratio} ({ratio_decimal:.2f}x)</div>
                            <div class="event-details">
                                Chaque action divisée en {split_to} nouvelles actions
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"ℹ️ Aucun stock split trouvé pour {ticker_div_clean}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px;color:#666;">
                <div style="font-size:40px;margin-bottom:20px;">💵</div>
                <div style="font-size:14px;">Entrez un ticker pour voir ses dividendes et stock splits</div>
                <div style="font-size:11px;margin-top:10px;color:#444;">Recherchez l'historique complet des distributions</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | FINNHUB + POLYGON.IO APIs | SYSTÈME OPÉRATIONNEL<br>
    AUTO-REFRESH: 60 SECONDES • DERNIÈRE MAJ: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
