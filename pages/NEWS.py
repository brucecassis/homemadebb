# pages/NEWS.py
# Bloomberg Terminal - News Feed avec Finnhub API + Calendrier Économique

import streamlit as st
import requests
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import pandas as pd

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
    
    .event-card {
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
        min-width: 90px;
        text-align: center;
    }
    
    .event-info {
        flex: 1;
    }
    
    .event-title {
        color: #FFAA00;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 4px;
    }
    
    .event-details {
        color: #888;
        font-size: 10px;
    }
    
    .event-impact {
        padding: 3px 8px;
        font-size: 9px;
        font-weight: bold;
        border: 1px solid;
    }
    
    .impact-high {
        background: #FF0000;
        color: #FFF;
        border-color: #FF0000;
    }
    
    .impact-medium {
        background: #FFAA00;
        color: #000;
        border-color: #FFAA00;
    }
    
    .impact-low {
        background: #00FF00;
        color: #000;
        border-color: #00FF00;
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
        min-width: 100px;
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
    
    /* DataFrame styling */
    .dataframe {
        background: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 10px !important;
    }
    
    .dataframe th {
        background: #FFAA00 !important;
        color: #000 !important;
        font-weight: bold !important;
        padding: 8px !important;
        text-align: left !important;
    }
    
    .dataframe td {
        background: #111 !important;
        color: #FFAA00 !important;
        padding: 6px !important;
        border-bottom: 1px solid #222 !important;
    }
    
    .dataframe tr:hover td {
        background: #1a1a1a !important;
    }
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
# FONCTIONS FINNHUB - CALENDRIER
# =============================================
@st.cache_data(ttl=3600)
def get_economic_calendar():
    """Calendrier économique"""
    try:
        url = f"https://finnhub.io/api/v1/calendar/economic?token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {"economicCalendar": []}
    except Exception as e:
        st.error(f"Erreur calendrier économique: {e}")
        return {"economicCalendar": []}

@st.cache_data(ttl=3600)
def get_ipo_calendar(from_date, to_date):
    """Calendrier IPO"""
    try:
        url = f"https://finnhub.io/api/v1/calendar/ipo?from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {"ipoCalendar": []}
    except Exception as e:
        st.error(f"Erreur calendrier IPO: {e}")
        return {"ipoCalendar": []}

@st.cache_data(ttl=3600)
def get_earnings_calendar(from_date, to_date):
    """Calendrier résultats trimestriels"""
    try:
        url = f"https://finnhub.io/api/v1/calendar/earnings?from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return {"earningsCalendar": []}
    except Exception as e:
        st.error(f"Erreur calendrier earnings: {e}")
        return {"earningsCalendar": []}

@st.cache_data(ttl=3600)
def get_dividends(ticker, from_date, to_date):
    """Dividendes pour un ticker"""
    try:
        url = f"https://finnhub.io/api/v1/stock/dividend?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Erreur dividendes: {e}")
        return []

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
    <div>{current_time} UTC • FINNHUB API • AUTO-REFRESH: 60s</div>
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
        "📊 ECONOMIC EVENTS",
        "🚀 IPO CALENDAR",
        "💰 EARNINGS CALENDAR",
        "💵 DIVIDENDS"
    ])
    
    # ========== SOUS-ONGLET 1: CALENDRIER ÉCONOMIQUE ==========
    with sub_tab1:
        st.markdown("#### 📊 ECONOMIC EVENTS & INDICATORS")
        
        st.markdown("""
        <div style="color:#666;font-size:10px;margin:10px 0;">
            Événements économiques majeurs, indicateurs macro-économiques et annonces des banques centrales
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        with st.spinner("📡 Chargement du calendrier économique..."):
            eco_data = get_economic_calendar()
        
        eco_events = eco_data.get("economicCalendar", [])
        
        if eco_events:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.metric("ÉVÉNEMENTS", len(eco_events))
            with col_e2:
                st.metric("MAJ", datetime.now().strftime("%H:%M:%S"))
            
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="category-header">📅 UPCOMING ECONOMIC EVENTS</div>', unsafe_allow_html=True)
            
            for event in eco_events[:30]:
                event_time = event.get("time", "N/A")
                country = event.get("country", "N/A")
                event_name = event.get("event", "Événement non spécifié")
                impact = event.get("impact", "").lower()
                actual = event.get("actual", "N/A")
                estimate = event.get("estimate", "N/A")
                previous = event.get("previous", "N/A")
                
                # Déterminer la classe d'impact
                impact_class = "impact-low"
                if impact == "high":
                    impact_class = "impact-high"
                elif impact == "medium":
                    impact_class = "impact-medium"
                
                st.markdown(f"""
                <div class="event-card">
                    <div class="event-date">{event_time}</div>
                    <div class="event-info">
                        <div class="event-title">{country} - {event_name}</div>
                        <div class="event-details">
                            ACTUAL: {actual} • FORECAST: {estimate} • PREVIOUS: {previous}
                        </div>
                    </div>
                    <div class="event-impact {impact_class}">{impact.upper() if impact else "N/A"}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Aucun événement économique disponible actuellement")
    
    # ========== SOUS-ONGLET 2: CALENDRIER IPO ==========
    with sub_tab2:
        st.markdown("#### 🚀 IPO CALENDAR - UPCOMING LISTINGS")
        
        # Sélection de période
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            ipo_days_ahead = st.selectbox(
                "Période à afficher",
                options=[7, 14, 30, 60, 90],
                format_func=lambda x: f"Prochains {x} jours",
                key="ipo_period"
            )
        
        from_date_ipo = datetime.now().strftime("%Y-%m-%d")
        to_date_ipo = (datetime.now() + timedelta(days=ipo_days_ahead)).strftime("%Y-%m-%d")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        with st.spinner("📡 Chargement des IPOs à venir..."):
            ipo_data = get_ipo_calendar(from_date_ipo, to_date_ipo)
        
        ipo_events = ipo_data.get("ipoCalendar", [])
        
        if ipo_events:
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.metric("IPOs", len(ipo_events))
            with col_i2:
                st.metric("PÉRIODE", f"{ipo_days_ahead} jours")
            with col_i3:
                st.metric("MAJ", datetime.now().strftime("%H:%M:%S"))
            
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="category-header">🚀 UPCOMING IPOs</div>', unsafe_allow_html=True)
            
            for ipo in ipo_events[:30]:
                ipo_date = ipo.get("date", "N/A")
                exchange = ipo.get("exchange", "N/A")
                name = ipo.get("name", "N/A")
                price_low = ipo.get("priceLow", "N/A")
                price_high = ipo.get("priceHigh", "N/A")
                shares = ipo.get("numberOfShares", "N/A")
                total_shares = ipo.get("totalSharesValue", "N/A")
                status = ipo.get("status", "N/A")
                symbol = ipo.get("symbol", "N/A")
                
                price_range = f"${price_low} - ${price_high}" if price_low != "N/A" and price_high != "N/A" else "N/A"
                
                st.markdown(f"""
                <div class="ipo-card">
                    <div style="display:flex;align-items:center;margin-bottom:8px;">
                        <span class="ipo-ticker">{symbol}</span>
                        <span class="ipo-name">{name}</span>
                    </div>
                    <div class="ipo-details">
                        📅 {format_date(ipo_date)} • 🏦 {exchange} • 💰 {price_range} • 📊 {shares} shares • STATUS: {status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Aucune IPO prévue dans la période sélectionnée")
    
    # ========== SOUS-ONGLET 3: CALENDRIER EARNINGS ==========
    with sub_tab3:
        st.markdown("#### 💰 EARNINGS CALENDAR - QUARTERLY RESULTS")
        
        # Sélection de période
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            earn_days_ahead = st.selectbox(
                "Période à afficher",
                options=[7, 14, 30, 60],
                format_func=lambda x: f"Prochains {x} jours",
                key="earn_period"
            )
        
        from_date_earn = datetime.now().strftime("%Y-%m-%d")
        to_date_earn = (datetime.now() + timedelta(days=earn_days_ahead)).strftime("%Y-%m-%d")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        with st.spinner("📡 Chargement du calendrier des résultats..."):
            earn_data = get_earnings_calendar(from_date_earn, to_date_earn)
        
        earnings_events = earn_data.get("earningsCalendar", [])
        
        if earnings_events:
            col_ea1, col_ea2, col_ea3 = st.columns(3)
            with col_ea1:
                st.metric("ENTREPRISES", len(earnings_events))
            with col_ea2:
                st.metric("PÉRIODE", f"{earn_days_ahead} jours")
            with col_ea3:
                st.metric("MAJ", datetime.now().strftime("%H:%M:%S"))
            
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="category-header">📈 UPCOMING EARNINGS RELEASES</div>', unsafe_allow_html=True)
            
            for earning in earnings_events[:50]:
                date = earning.get("date", "N/A")
                symbol = earning.get("symbol", "N/A")
                eps_actual = earning.get("epsActual", "N/A")
                eps_estimate = earning.get("epsEstimate", "N/A")
                hour = earning.get("hour", "N/A")
                quarter = earning.get("quarter", "N/A")
                revenue_actual = earning.get("revenueActual", "N/A")
                revenue_estimate = earning.get("revenueEstimate", "N/A")
                year = earning.get("year", "N/A")
                
                eps_info = f"EPS: {eps_estimate} (est.)" if eps_estimate != "N/A" else "EPS: N/A"
                revenue_info = f"REV: {revenue_estimate}M (est.)" if revenue_estimate != "N/A" else ""
                
                st.markdown(f"""
                <div class="earnings-card">
                    <div class="earnings-ticker">{symbol}</div>
                    <div class="earnings-info">
                        <div class="earnings-name">Q{quarter} {year} EARNINGS</div>
                        <div class="earnings-date">📅 {format_date(date)} • {hour}</div>
                    </div>
                    <div class="earnings-eps">
                        {eps_info}<br>
                        {revenue_info}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Aucun résultat trimestriel prévu dans la période sélectionnée")
    
    # ========== SOUS-ONGLET 4: DIVIDENDES ==========
    with sub_tab4:
        st.markdown("#### 💵 DIVIDEND CALENDAR")
        
        st.markdown("""
        <div class="search-box">
            <div style="color:#FFAA00;font-size:12px;margin-bottom:10px;">
                Entrez un ticker US pour voir son historique et calendrier de dividendes
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_d1, col_d2, col_d3 = st.columns([3, 1, 1])
        
        with col_d1:
            div_ticker = st.text_input(
                "Ticker pour dividendes",
                placeholder="Ex: AAPL, MSFT, JNJ, KO...",
                label_visibility="collapsed",
                key="div_ticker"
            )
        
        with col_d2:
            div_years = st.selectbox(
                "Période",
                options=[1, 2, 3, 5],
                format_func=lambda x: f"{x} an{'s' if x>1 else ''}",
                label_visibility="collapsed",
                key="div_years"
            )
        
        with col_d3:
            div_search_btn = st.button("🔍 SEARCH", use_container_width=True, key="div_search")
        
        st.markdown("""
        <div style="color:#666;font-size:10px;margin:10px 0;">
            <b>EXEMPLES DE DIVIDEND ARISTOCRATS:</b> JNJ • PG • KO • PEP • MCD • WMT • XOM • CVX • MMM • CAT
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        if div_ticker:
            ticker_div_clean = div_ticker.upper().strip()
            
            from_date_div = (datetime.now() - timedelta(days=365*div_years)).strftime("%Y-%m-%d")
            to_date_div = (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d")
            
            with st.spinner(f"📡 Chargement des dividendes pour {ticker_div_clean}..."):
                dividends = get_dividends(ticker_div_clean, from_date_div, to_date_div)
            
            if dividends:
                col_dv1, col_dv2, col_dv3 = st.columns(3)
                with col_dv1:
                    st.metric("TICKER", ticker_div_clean)
                with col_dv2:
                    st.metric("PAIEMENTS", len(dividends))
                with col_dv3:
                    total_div = sum([d.get('amount', 0) for d in dividends])
                    st.metric("TOTAL", f"${total_div:.2f}")
                
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown(f'<div class="category-header">💰 {ticker_div_clean} - DIVIDEND HISTORY</div>', unsafe_allow_html=True)
                
                for div in dividends:
                    amount = div.get("amount", 0)
                    currency = div.get("currency", "USD")
                    date = div.get("date", "N/A")
                    declaration_date = div.get("declarationDate", "N/A")
                    ex_date = div.get("exDate", "N/A")
                    payment_date = div.get("payDate", "N/A")
                    record_date = div.get("recordDate", "N/A")
                    
                    st.markdown(f"""
                    <div class="dividend-card">
                        <div class="dividend-amount">${amount:.4f}</div>
                        <div class="dividend-info">
                            <div class="dividend-date">📅 Payment Date: {format_date(payment_date)}</div>
                            <div class="dividend-type">
                                Ex-Date: {format_date(ex_date)} • Record: {format_date(record_date)} • Declaration: {format_date(declaration_date)}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Aucun dividende trouvé pour {ticker_div_clean}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px;color:#666;">
                <div style="font-size:40px;margin-bottom:20px;">💵</div>
                <div style="font-size:14px;">Entrez un ticker ci-dessus pour voir son calendrier de dividendes</div>
                <div style="font-size:11px;margin-top:10px;color:#444;">Recherchez des entreprises qui distribuent régulièrement des dividendes</div>
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
