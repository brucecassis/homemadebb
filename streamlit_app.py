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

# CSS personnalisé style Bloomberg
st.markdown("""
<style>
    .main {background-color: #0a0a0a;}
    .stApp {background-color: #0a0a0a;}
    h1, h2, h3 {color: #ff8c00;}
    .stMetric {background-color: #1a1a1a; padding: 10px; border-radius: 5px;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données réelles
@st.cache_data(ttl=60)  # Cache pendant 60 secondes pour éviter trop de requêtes
def get_market_data(ticker):
    """Récupère les données réelles de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Récupérer les données historiques des 5 derniers jours
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None, None, None
        
        # Prix actuel (dernier close)
        current_price = hist['Close'].iloc[-1]
        
        # Prix de la veille
        previous_close = hist['Close'].iloc[-2]
        
        # Calcul du changement en %
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return current_price, change_percent, hist
        
    except Exception as e:
        st.error(f"Erreur pour {ticker}: {str(e)}")
        return None, None, None

# Fonction pour récupérer les news réelles
@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_real_news(ticker):
    """Récupère les vraies news de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        return news[:5] if news else []
    except:
        return []

# Header avec heure actuelle
col1, col2, col3 = st.columns([2,4,2])
with col1:
    st.title("📊 BLOOMBERG")
with col3:
    # Affichage de l'heure en temps réel
    placeholder_time = st.empty()
    current_time = datetime.now()
    placeholder_time.write(f"🕐 {current_time.strftime('%H:%M:%S')}")
    st.write(f"📅 {current_time.strftime('%d %B %Y')}")

st.markdown("---")

# Market Overview avec données RÉELLES
st.header("📈 Market Overview - Live Data")

# Bouton de refresh
col_refresh, col_empty = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Définition des marchés à suivre
markets = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW JONES': '^DJI',
    'EUR/USD': 'EURUSD=X'
}

# Affichage des métriques en temps réel
cols = st.columns(4)

for idx, (name, ticker) in enumerate(markets.items()):
    with cols[idx]:
        with st.spinner(f"Loading {name}..."):
            current, change, hist = get_market_data(ticker)
            
            if current is not None:
                # Formatage selon le type d'actif
                if 'USD' in ticker or 'EUR' in ticker:
                    value_display = f"{current:.4f}"
                else:
                    value_display = f"{current:,.2f}"
                
                # Affichage avec delta coloré
                st.metric(
                    label=name,
                    value=value_display,
                    delta=f"{change:+.2f}%",
                    delta_color="normal"
                )
                
                # Mini graphique sparkline
                if hist is not None and len(hist) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        line=dict(color='#ff8c00', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255, 140, 0, 0.1)'
                    ))
                    fig.update_layout(
                        height=100,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.metric(label=name, value="N/A", delta="0%")
                st.caption("⚠️ Données non disponibles")

st.markdown("---")

# Section des commodités et crypto
st.subheader("💰 Commodities & Crypto - Live Data")

commodities = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Crude Oil': 'CL=F',
    'Natural Gas': 'NG=F',
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD'
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
            st.metric(label=name, value="N/A")

st.markdown("---")

# Main Content avec vraies news
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    st.subheader("📰 Latest Market News")
    
    # Récupération des vraies news depuis Yahoo Finance
    all_news = []
    for ticker in ['^GSPC', '^IXIC', '^DJI']:
        news = get_real_news(ticker)
        all_news.extend(news)
    
    # Affichage des news ou placeholder si pas de news
    if all_news:
        # Trier par date et prendre les 8 premières
        all_news = sorted(all_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:8]
        
        for item in all_news:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            link = item.get('link', '#')
            
            # Calcul du temps écoulé
            pub_time = item.get('providerPublishTime', 0)
            if pub_time:
                time_ago = datetime.now() - datetime.fromtimestamp(pub_time)
                hours_ago = int(time_ago.total_seconds() / 3600)
                if hours_ago < 1:
                    time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                elif hours_ago < 24:
                    time_str = f"{hours_ago}h ago"
                else:
                    time_str = f"{int(hours_ago / 24)}d ago"
            else:
                time_str = "Recently"
            
            st.markdown(f"""
            <div style='background-color: #1a1a1a; padding: 15px; margin-bottom: 10px; border-left: 3px solid #ff8c00;'>
                <h4 style='margin: 0; color: white;'>
                    <a href='{link}' target='_blank' style='color: white; text-decoration: none;'>
                        {title}
                    </a>
                </h4>
                <p style='margin: 5px 0 0 0; color: #888;'>
                    <span style='color: #ff8c00;'>{publisher}</span> • {time_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📡 Chargement des actualités en cours...")
        # Fallback avec quelques news génériques
        st.markdown("""
        <div style='background-color: #1a1a1a; padding: 15px; margin-bottom: 10px; border-left: 3px solid #ff8c00;'>
            <h4 style='margin: 0; color: white;'>Market Data Loading...</h4>
            <p style='margin: 5px 0 0 0; color: #888;'>Actualisation des données en cours</p>
        </div>
        """, unsafe_allow_html=True)

with col_sidebar:
    st.subheader("🛠️ Quick Tools")
    
    st.button("📊 Stock Screener", use_container_width=True, type="primary")
    st.button("💼 Portfolio Tracker", use_container_width=True)
    st.button("📅 Economic Calendar", use_container_width=True)
    st.button("⭐ Watchlist", use_container_width=True)
    st.button("📈 Market Analysis", use_container_width=True)
    
    st.markdown("---")
    
    # Section recherche de ticker personnalisé
    st.subheader("🔍 Quick Search")
    custom_ticker = st.text_input("Enter ticker (ex: AAPL, MSFT, TSLA)", "")
    
    if custom_ticker:
        with st.spinner(f"Fetching {custom_ticker.upper()}..."):
            current, change, hist = get_market_data(custom_ticker.upper())
            
            if current is not None:
                st.metric(
                    label=custom_ticker.upper(),
                    value=f"${current:,.2f}",
                    delta=f"{change:+.2f}%"
                )
                
                if hist is not None and len(hist) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        increasing_line_color='#00ff00',
                        decreasing_line_color='#ff0000'
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        xaxis_rangeslider_visible=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='#1a1a1a'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ Ticker '{custom_ticker.upper()}' non trouvé")

# Footer avec timestamp de dernière mise à jour
st.markdown("---")
last_update = datetime.now().strftime('%H:%M:%S')
st.caption(f"© 2025 Bloomberg Terminal Clone • Données Yahoo Finance • Dernière MàJ: {last_update}")
st.caption("⚠️ Les données sont fournies à titre informatif uniquement. Actualisez pour obtenir les derniers cours.")
