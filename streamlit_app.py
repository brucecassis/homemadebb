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
    initial_sidebar_state="expanded"
)

# CSS personnalisé STYLE BLOOMBERG AUTHENTIQUE
st.markdown("""
<style>
    /* Fond noir total */
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
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    /* Barre orange Bloomberg */
    .bloomberg-header {
        background: #FFAA00;
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 14px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }
    
    /* Titres orange */
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
    
    /* Metrics Bloomberg style */
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
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
    
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* News style Bloomberg */
    .news-item {
        background-color: #0a0a0a;
        border-left: 3px solid #FFAA00;
        padding: 8px 10px;
        margin-bottom: 5px;
        border-bottom: 1px solid #222;
        font-family: 'Courier New', monospace;
    }
    
    .news-title {
        color: #FFAA00;
        font-size: 11px;
        font-weight: 600;
        margin: 0;
        line-height: 1.3;
    }
    
    .news-meta {
        color: #666;
        font-size: 9px;
        margin-top: 3px;
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
        padding: 6px 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 10px;
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
        font-size: 11px;
    }
    
    .stTextInput input:focus {
        border-color: #FFF;
        box-shadow: 0 0 3px #FFAA00;
    }
    
    /* Lignes de séparation */
    hr {
        border-color: #333333;
        margin: 5px 0;
    }
    
    /* Supprimer le padding par défaut */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Horloge en temps réel */
    .live-clock {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    /* Command line style */
    .command-line {
        background: #000;
        padding: 5px 10px;
        border: 1px solid #333;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #FFAA00;
        margin: 5px 0;
    }
    
    .prompt {
        color: #FFAA00;
        font-weight: bold;
        margin-right: 8px;
    }
    
    /* Sidebar navigation */
    [data-testid="stSidebar"] {
        background-color: #000;
        border-right: 2px solid #FFAA00;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FFAA00;
        font-family: 'Courier New', monospace;
    }
    
    /* Terminal text style */
    p, div, span {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
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
    
    setInterval(updateClock, 1000);
    updateClock();
    
    // Auto-refresh toutes les 3 secondes
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
        return news[:10] if news else []
    except:
        return []

# ===== HEADER BLOOMBERG avec horloge =====
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL</div>
    <div class="live-clock">{current_time.strftime("%H:%M:%S")} PARIS</div>
</div>
''', unsafe_allow_html=True)

# ===== BARRE DE COMMANDE =====
st.markdown("""
<div class="command-line">
    <span class="prompt">NAV></span>
    <span style="color: #666;">OPTIONS PRICING disponible dans le menu à gauche ←</span>
</div>
""", unsafe_allow_html=True)

col_nav1, col_nav2 = st.columns([4, 1])

with col_nav1:
    search_command = st.text_input("", placeholder="Commande: HELP pour aide", label_visibility="collapsed", key="nav_search")

with col_nav2:
    if st.button("EXEC", use_container_width=True):
        cmd = search_command.upper().strip()
        
        if cmd == "HELP":
            st.info("📋 **COMMANDES:**\nUtilisez le menu à gauche pour OPTIONS PRICING\nTapez un ticker dans la recherche rapide")
        elif cmd:
            st.warning(f"⚠️ '{cmd}' - Utilisez le menu latéral")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 5px 0;"></div>', unsafe_allow_html=True)

# ===== MARKET OVERVIEW =====
st.markdown("### 📊 GLOBAL MARKETS - LIVE")

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
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

# ===== COMMODITIES =====
st.markdown("### 💰 COMMODITIES & CRYPTO")

commodities = {
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'OIL': 'CL=F',
    'GAS': 'NG=F',
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD'
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
            st.metric(label=name, value="LOAD...")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)

# ===== MAIN CONTENT =====
col_main, col_sidebar = st.columns([2.5, 1])

with col_main:
    st.markdown("### 📰 MARKET NEWS")
    
    all_news = []
    for ticker in ['^GSPC', '^IXIC']:
        news = get_real_news(ticker)
        all_news.extend(news)
    
    if all_news:
        all_news = sorted(all_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:12]
        
        for item in all_news:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            link = item.get('link', '#')
            
            pub_time = item.get('providerPublishTime', 0)
            if pub_time:
                time_ago = datetime.now() - datetime.fromtimestamp(pub_time)
                hours_ago = int(time_ago.total_seconds() / 3600)
                if hours_ago < 1:
                    time_str = f"{int(time_ago.total_seconds() / 60)}M"
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
                    <span class="news-category">{publisher.upper()}</span> • {time_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📡 Loading...")

with col_sidebar:
    st.markdown("### 🔍 TICKER SEARCH")
    
    custom_ticker = st.text_input("", placeholder="AAPL, TSLA...", label_visibility="collapsed", key="ticker_search")
    
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
                    height=120,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.error(f"❌ NOT FOUND")
    
    st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0; padding-top: 10px;"></div>', unsafe_allow_html=True)
    st.markdown("### 🛠️ QUICK ACCESS")
    st.markdown("**→ OPTIONS PRICING**")
    st.caption("Voir menu latéral ←")

# ===== FOOTER =====
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE | UPDATE: {last_update} | REFRESH: 3S
</div>
""", unsafe_allow_html=True)
