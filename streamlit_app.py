import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time

# Configuration de la page
st.set_page_config(
    page_title="Bloomberg Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé STYLE BLOOMBERG + HORLOGE EN TEMPS RÉEL
st.markdown("""
<style>
    /* Fond noir total */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    /* Barre orange Bloomberg */
    .bloomberg-header {
        background: linear-gradient(90deg, #FFAA00 0%, #FF8C00 100%);
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 16px;
        font-family: 'Courier New', monospace;
        margin-bottom: 0px;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Navigation / Commande */
    .nav-section {
        background: #000;
        padding: 10px 20px;
        border-bottom: 1px solid #333;
        display: flex;
        gap: 15px;
        align-items: center;
    }
    
    .nav-btn {
        background: #333;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        padding: 6px 15px;
        font-size: 11px;
        cursor: pointer;
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        font-weight: bold;
    }
    
    .nav-btn:hover {
        background: #FFAA00;
        color: #000;
    }
    
    /* Titres orange */
    h1, h2, h3, h4 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 14px !important;
        margin: 10px 0 !important;
    }
    
    /* Metrics Bloomberg style */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFAA00 !important;
        font-size: 11px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        font-family: 'Courier New', monospace !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* News style Bloomberg */
    .news-item {
        background-color: #0a0a0a;
        border-left: 3px solid #FFAA00;
        padding: 10px 12px;
        margin-bottom: 6px;
        border-bottom: 1px solid #222;
        font-family: 'Courier New', monospace;
    }
    
    .news-title {
        color: #FFAA00;
        font-size: 12px;
        font-weight: 600;
        margin: 0;
        line-height: 1.4;
    }
    
    .news-meta {
        color: #666;
        font-size: 10px;
        margin-top: 4px;
    }
    
    .news-category {
        color: #FFAA00;
        font-weight: bold;
    }
    
    /* Boutons Bloomberg */
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        font-weight: bold;
        border: 1px solid #FFAA00;
        padding: 8px 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 11px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    /* Input boxes */
    .stTextInput input {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    
    .stTextInput input:focus {
        border-color: #FFF;
        box-shadow: 0 0 3px #FFAA00;
    }
    
    /* Lignes de séparation */
    hr {
        border-color: #333333;
        margin: 8px 0;
    }
    
    /* Supprimer le padding par défaut */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Horloge en temps réel */
    .live-clock {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    /* Command line */
    .command-line {
        background: #000;
        padding: 6px 12px;
        border: 1px solid #FFAA00;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #FFAA00;
        margin: 10px 0;
    }
    
    .prompt {
        color: #FFAA00;
        font-weight: bold;
        margin-right: 8px;
    }
</style>

<script>
    // Horloge en temps réel JavaScript
    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        const timeString = hours + ':' + minutes + ':' + seconds + ' PARIS';
        
        const clockElements = document.querySelectorAll('.live-clock');
        clockElements.forEach(el => {
            el.textContent = timeString;
        });
    }
    
    // Mise à jour toutes les secondes
    setInterval(updateClock, 1000);
    updateClock();
</script>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données réelles
def get_market_data(ticker):
    """Récupère les données réelles de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None, None, None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return current_price, change_percent, hist
        
    except:
        return None, None, None

# Fonction pour récupérer les news réelles
def get_real_news(ticker):
    """Récupère les vraies news de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:8] if news else []
    except:
        return []

# ===== HEADER BLOOMBERG avec horloge =====
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG TERMINAL</div>
    <div class="live-clock">{current_time.strftime("%H:%M:%S")} PARIS</div>
</div>
''', unsafe_allow_html=True)

# ===== BARRE DE NAVIGATION / COMMANDE =====
st.markdown("""
<div class="command-line">
    <span class="prompt">COMMAND></span>
    <span style="color: #666;">Tapez une fonction: PRICING, NEWS, SCREENER, PORTFOLIO, HELP...</span>
</div>
""", unsafe_allow_html=True)

# Barre de recherche avec navigation
col_nav1, col_nav2, col_nav3 = st.columns([3, 1, 1])

with col_nav1:
    search_command = st.text_input("", placeholder="Entrez une commande (ex: PRICING, NEWS, SCREENER...)", label_visibility="collapsed", key="nav_search")

with col_nav2:
    if st.button("🔍 EXECUTER", use_container_width=True):
        cmd = search_command.upper().strip()
        
        if cmd == "PRICING" or cmd == "PRICE":
            st.switch_page("pages/pricing.py")
        elif cmd == "NEWS" or cmd == "N":
            st.info("📰 Page NEWS en construction...")
        elif cmd == "SCREENER" or cmd == "SCREEN":
            st.info("📊 Page SCREENER en construction...")
        elif cmd == "PORTFOLIO" or cmd == "PORT":
            st.info("💼 Page PORTFOLIO en construction...")
        elif cmd == "HELP":
            st.info("""
            **Commandes disponibles:**
            - PRICING : Options pricing calculator
            - NEWS : Market news
            - SCREENER : Stock screener
            - PORTFOLIO : Portfolio tracker
            - HELP : Afficher cette aide
            """)
        elif cmd:
            st.warning(f"⚠️ Commande '{cmd}' non reconnue. Tapez HELP pour voir les commandes disponibles.")

with col_nav3:
    if st.button("🔄 REFRESH", use_container_width=True):
        st.rerun()

st.markdown("---")

# ===== MARKET OVERVIEW =====
st.markdown("### 📊 GLOBAL MARKETS - LIVE DATA")

markets = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    'EUR/USD': 'EURUSD=X'
}

cols = st.columns(4)

for idx, (name, ticker) in enumerate(markets.items()):
    with cols[idx]:
        current, change, hist = get_market_data(ticker)
        
        if current is not None:
            if 'USD' in ticker or 'EUR' in ticker:
                value_display = f"{current:.4f}"
            else:
                value_display = f"{current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOADING...", delta="0%")

st.markdown("---")

# ===== COMMODITIES & CRYPTO =====
st.markdown("### 💰 COMMODITIES & DIGITAL ASSETS")

commodities = {
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'CRUDE OIL': 'CL=F',
    'NAT GAS': 'NG=F',
    'BITCOIN': 'BTC-USD',
    'ETHEREUM': 'ETH-USD'
}

cols_comm = st.columns(6)

for idx, (name, ticker) in enumerate(commodities.items()):
    with cols_comm[idx]:
        current, change, _ = get_market_data(ticker)
        
        if current is not None:
            if 'BTC' in ticker or 'ETH' in ticker:
                value_display = f"${current:,.0f}"
            else:
                value_display = f"${current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOADING...")

st.markdown("---")

# ===== MAIN CONTENT =====
col_main, col_sidebar = st.columns([2.5, 1])

with col_main:
    st.markdown("### 📰 MARKET NEWS & HEADLINES")
    
    # Récupérer vraies news
    all_news = []
    for ticker in ['^GSPC', '^IXIC', '^DJI']:
        news = get_real_news(ticker)
        all_news.extend(news)
    
    if all_news:
        all_news = sorted(all_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:10]
        
        for item in all_news:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            link = item.get('link', '#')
            
            pub_time = item.get('providerPublishTime', 0)
            if pub_time:
                time_ago = datetime.now() - datetime.fromtimestamp(pub_time)
                hours_ago = int(time_ago.total_seconds() / 3600)
                if hours_ago < 1:
                    time_str = f"{int(time_ago.total_seconds() / 60)}MIN"
                elif hours_ago < 24:
                    time_str = f"{hours_ago}H"
                else:
                    time_str = f"{int(hours_ago / 24)}D"
            else:
                time_str = "NOW"
            
            st.markdown(f"""
            <div class="news-item">
                <div class="news-title">
                    <a href='{link}' target='_blank' style='color: #FFAA00; text-decoration: none;'>
                        ▸ {title}
                    </a>
                </div>
                <div class="news-meta">
                    <span class="news-category">{publisher.upper()}</span> • {time_str} AGO
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📡 Loading market news...")

with col_sidebar:
    st.markdown("### 🔍 QUICK TICKER SEARCH")
    
    custom_ticker = st.text_input("", placeholder="AAPL, MSFT, TSLA...", label_visibility="collapsed", key="ticker_search")
    
    if custom_ticker:
        current, change, hist = get_market_data(custom_ticker.upper())
        
        if current is not None:
            st.metric(
                label=custom_ticker.upper(),
                value=f"${current:,.2f}",
                delta=f"{change:+.2f}%"
            )
            
            if hist is not None and len(hist) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 170, 0, 0.2)'
                ))
                fig.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.error(f"❌ SYMBOL NOT FOUND")
    
    st.markdown("---")
    st.markdown("### 🛠️ TERMINAL FUNCTIONS")
    
    if st.button("📊 OPTIONS PRICING", use_container_width=True, key="btn_pricing"):
        st.switch_page("pages/pricing.py")
    
    if st.button("📈 STOCK SCREENER", use_container_width=True, key="btn_screener"):
        st.info("Coming soon...")
    
    if st.button("💼 PORTFOLIO", use_container_width=True, key="btn_portfolio"):
        st.info("Coming soon...")
    
    if st.button("📅 ECO CALENDAR", use_container_width=True, key="btn_calendar"):
        st.info("Coming soon...")
    
    if st.button("⭐ WATCHLIST", use_container_width=True, key="btn_watchlist"):
        st.info("Coming soon...")

# ===== FOOTER =====
st.markdown("---")
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 10px; font-family: "Courier New", monospace; padding: 8px;'>
    © 2025 BLOOMBERG TERMINAL CLONE | DATA: YAHOO FINANCE | LAST UPDATE: {last_update}
</div>
""", unsafe_allow_html=True)
