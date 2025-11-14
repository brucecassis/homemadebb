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

# CSS personnalisé STYLE BLOOMBERG AUTHENTIQUE
st.markdown("""
<style>
    /* Fond noir total */
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stApp {
        background-color: #000000;
    }
    
    /* Barre orange Bloomberg */
    .bloomberg-header {
        background: linear-gradient(90deg, #f47920 0%, #f8981d 100%);
        padding: 12px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 0px;
        letter-spacing: 2px;
        border-bottom: 2px solid #f8981d;
    }
    
    /* Titres orange */
    h1, h2, h3, h4 {
        color: #f47920 !important;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metrics Bloomberg style */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #ffffff;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        color: #f47920;
        font-size: 13px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: bold;
    }
    
    /* Cartes de données */
    .market-card {
        background-color: #1a1a1a;
        padding: 15px;
        border-left: 4px solid #f47920;
        border-radius: 0px;
        margin-bottom: 10px;
    }
    
    /* News style Bloomberg */
    .news-item {
        background-color: #0a0a0a;
        border-left: 3px solid #f47920;
        padding: 12px 15px;
        margin-bottom: 8px;
        border-bottom: 1px solid #333;
    }
    
    .news-title {
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
        margin: 0;
        line-height: 1.4;
    }
    
    .news-meta {
        color: #888888;
        font-size: 12px;
        margin-top: 5px;
    }
    
    .news-category {
        color: #f47920;
        font-weight: bold;
    }
    
    /* Boutons Bloomberg */
    .stButton > button {
        background-color: #f47920;
        color: #000000;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 13px;
    }
    
    .stButton > button:hover {
        background-color: #ff9933;
        color: #000000;
    }
    
    /* Barre de temps */
    .time-bar {
        background-color: #1a1a1a;
        padding: 8px 20px;
        color: #f47920;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        border-bottom: 1px solid #333;
        text-align: right;
    }
    
    /* Supprimer le padding par défaut */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* Sidebar sombre */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }
    
    /* Input boxes */
    .stTextInput input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #f47920;
        border-radius: 0px;
    }
    
    /* Lignes de séparation */
    hr {
        border-color: #333333;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# AUTO-REFRESH toutes les 3 secondes
st.markdown("""
<script>
    setTimeout(function() {
        window.location.reload();
    }, 3000);
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

# ===== HEADER BLOOMBERG =====
st.markdown('<div class="bloomberg-header">⬛ BLOOMBERG TERMINAL</div>', unsafe_allow_html=True)

# Barre de temps avec auto-update
current_time = datetime.now()
st.markdown(f'<div class="time-bar">🕐 {current_time.strftime("%H:%M:%S")} | 📅 {current_time.strftime("%A, %B %d, %Y").upper()} | LIVE DATA ●</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===== MARKET OVERVIEW =====
st.markdown("### 📊 GLOBAL MARKETS")

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
        all_news = sorted(all_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:8]
        
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
                    <a href='{link}' target='_blank' style='color: #ffffff; text-decoration: none;'>
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
    
    custom_ticker = st.text_input("", placeholder="Enter symbol (AAPL, MSFT...)", label_visibility="collapsed")
    
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
                    line=dict(color='#f47920', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(244, 121, 32, 0.2)'
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
    st.button("📊 EQUITY SCREENER", use_container_width=True)
    st.button("💼 PORTFOLIO", use_container_width=True)
    st.button("📈 TECHNICAL ANALYSIS", use_container_width=True)
    st.button("📅 ECO CALENDAR", use_container_width=True)
    st.button("⭐ WATCHLIST", use_container_width=True)

# ===== FOOTER =====
st.markdown("---")
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 11px; padding: 10px;'>
    © 2025 BLOOMBERG TERMINAL CLONE | DATA: YAHOO FINANCE | LAST UPDATE: {last_update} | AUTO-REFRESH: 3S
</div>
""", unsafe_allow_html=True)
