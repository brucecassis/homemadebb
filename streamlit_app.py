import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time
import requests

# =============================================
# AUTO-REFRESH TOUTES LES 3 SECONDES
# =============================================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 3:
    st.session_state.last_refresh = time.time()
    st.rerun()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Markets",
    page_icon="📊",
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
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    .stApp {
        background-color: #000000;
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
    
    [data-testid="stMetricDelta"] {
        font-size: 11px !important;
        font-weight: bold !important;
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
        transform: translateY(-2px) !important;
    }
    
    hr {
        border-color: #333333;
        margin: 8px 0;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .section-box {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FFAA00;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# API COINMARKETCAP
# =============================================
CMC_API_KEY = "09e527de-bfea-4816-8afe-ae6a37bf5799"

@st.cache_data(ttl=3)
def get_crypto_data_cmc(symbols):
    """Récupère les données crypto depuis CoinMarketCap"""
    try:
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        parameters = {
            'symbol': ','.join(symbols),
            'convert': 'USD'
        }
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': CMC_API_KEY,
        }
        response = requests.get(url, headers=headers, params=parameters)
        data = response.json()
        
        crypto_data = {}
        for symbol in symbols:
            if symbol in data['data']:
                quote = data['data'][symbol]['quote']['USD']
                crypto_data[symbol] = {
                    'price': quote['price'],
                    'change_24h': quote['percent_change_24h']
                }
        return crypto_data
    except Exception as e:
        return None

# =============================================
# FONCTION DONNÉES MARCHÉ
# =============================================
@st.cache_data(ttl=3)
def get_market_data(ticker):
    """Récupère les données réelles de Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None, None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return current_price, change_percent
    except:
        return None, None

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - LIVE MARKETS</div>
        <a href="accueil.html" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • AUTO-REFRESH: 3s</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# INDICES GLOBAUX
# =============================================
st.markdown("### 📊 GLOBAL INDICES - LIVE")

indices = {
    'NASDAQ': '^IXIC',
    'S&P 500': '^GSPC',
    'DOW JONES': '^DJI',
    'RUSSELL 2000': '^RUT',
    'VIX': '^VIX',
    'CAC 40': '^FCHI',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'SMI (SIX)': '^SSMI',
    'FTSE MIB': 'FTSEMIB.MI',
    'NIKKEI 225': '^N225',
    'SSE (CHINA)': '000001.SS',
    'DXY (USD)': 'DX-Y.NYB'
}

cols_indices = st.columns(6)

for idx, (name, ticker) in enumerate(indices.items()):
    with cols_indices[idx % 6]:
        current, change = get_market_data(ticker)
        
        if current is not None:
            if name == 'VIX':
                value_display = f"{current:.2f}"
            else:
                value_display = f"{current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# FOREX
# =============================================
st.markdown("### 💱 FOREX - LIVE")

forex = {
    'EUR/USD': 'EURUSD=X',
    'CHF/USD': 'CHFUSD=X',
    'CHF/EUR': 'CHFEUR=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'JPY=X',
    'USD/CNY': 'CNY=X'
}

cols_fx = st.columns(6)

for idx, (name, ticker) in enumerate(forex.items()):
    with cols_fx[idx]:
        current, change = get_market_data(ticker)
        
        if current is not None:
            st.metric(
                label=name,
                value=f"{current:.4f}",
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# COMMODITIES
# =============================================
st.markdown("### 💰 COMMODITIES - LIVE")

commodities = {
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'PLATINUM': 'PL=F',
    'COPPER': 'HG=F',
    'OIL (WTI)': 'CL=F',
    'NAT GAS': 'NG=F',
    'BRENT OIL': 'BZ=F',
    'ALUMINUM': 'ALI=F'
}

cols_comm = st.columns(4)

for idx, (name, ticker) in enumerate(commodities.items()):
    with cols_comm[idx % 4]:
        current, change = get_market_data(ticker)
        
        if current is not None:
            value_display = f"${current:,.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%"
            )
        else:
            st.metric(label=name, value="LOAD...")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# CRYPTO - COINMARKETCAP
# =============================================
st.markdown("### ₿ CRYPTO - COINMARKETCAP LIVE")

crypto_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'AAVE', 'LINK']
crypto_data_cmc = get_crypto_data_cmc(crypto_symbols)

cols_crypto = st.columns(6)

crypto_pairs = [
    ('BTC', 'BTCUSDT'),
    ('ETH', 'ETHUSDT'),
    ('SOL', 'SOLUSDT'),
    ('XRP', 'XRPUSDT'),
    ('AAVE', 'AAVEUSDT'),
    ('LINK', 'LINKUSDT')
]

if crypto_data_cmc:
    for idx, (symbol, pair) in enumerate(crypto_pairs):
        with cols_crypto[idx]:
            if symbol in crypto_data_cmc:
                price = crypto_data_cmc[symbol]['price']
                change = crypto_data_cmc[symbol]['change_24h']
                
                if price >= 100:
                    value_display = f"${price:,.2f}"
                else:
                    value_display = f"${price:.4f}"
                
                st.metric(
                    label=pair,
                    value=value_display,
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=pair, value="ERROR", delta="0%")
else:
    for idx, (_, pair) in enumerate(crypto_pairs):
        with cols_crypto[idx]:
            st.metric(label=pair, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# GRAPHIQUE COMPARATIF + MATRICE CORRÉLATION
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("### 📈 COMPARATIVE CHART - PERFORMANCE %")

col_chart1, col_chart2 = st.columns([3, 1])

with col_chart1:
    popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'DIS',
                       'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'XOM', 'CVX', 'COP', 'SLB',
                       'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'LLY', 'MRK', 'BMY', 'AMGN',
                       'MC.PA', 'OR.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'TTE.PA', 'SAF.PA',
                       'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ZURN.SW',
                       'BTC-USD', 'ETH-USD', 'SOL-USD', 'SPY', 'QQQ', 'IWM', 'DIA']
    
    selected_tickers = st.multiselect(
        "Sélectionnez des tickers à comparer",
        options=popular_tickers,
        default=['AAPL', 'MSFT', 'TSLA'],
        help="Sélectionnez jusqu'à 10 tickers"
    )

with col_chart2:
    timeframe = st.selectbox(
        "Timeframe",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y'],
        index=3,
        help="Période d'analyse"
    )

# Fonction pour normaliser en %
def normalize_to_percentage(df):
    """Normalise les prix à 100% au début"""
    if df is None or len(df) == 0:
        return None
    return (df / df.iloc[0]) * 100

# Fonction pour récupérer les données historiques
@st.cache_data(ttl=300)
def get_historical_data(ticker, period):
    """Récupère les données historiques pour un ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except:
        return None

# Génération du graphique
if selected_tickers:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Couleurs Bloomberg style
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00', 
              '#FF1493', '#00CED1', '#32CD32', '#FFD700']
    
    # Stocker les données pour la corrélation
    correlation_data = pd.DataFrame()
    
    with st.spinner('📊 Chargement des données...'):
        for idx, ticker in enumerate(selected_tickers[:10]):
            hist = get_historical_data(ticker, timeframe)
            
            if hist is not None and len(hist) > 0:
                # Normaliser à 100%
                normalized = normalize_to_percentage(hist['Close'])
                
                # Stocker pour corrélation
                correlation_data[ticker] = hist['Close']
                
                # Ajouter la courbe
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'<b>{ticker}</b><br>%{{y:.2f}}%<br>%{{x}}<extra></extra>'
                ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title=f"Performance Comparison - {timeframe.upper()}",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        xaxis=dict(
            gridcolor='#333',
            showgrid=True,
            title="Date"
        ),
        yaxis=dict(
            gridcolor='#333',
            showgrid=True,
            title="Performance (%)"
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Ligne horizontale à 100%
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="paper",
        y0=100, y1=100,
        line=dict(color="#666", width=1, dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistiques des performances
    st.markdown("#### 📊 PERFORMANCE SUMMARY")
    
    cols_perf = st.columns(min(len(selected_tickers), 10))
    
    for idx, ticker in enumerate(selected_tickers[:10]):
        with cols_perf[idx]:
            hist = get_historical_data(ticker, timeframe)
            if hist is not None and len(hist) > 1:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                perf = ((end_price - start_price) / start_price) * 100
                
                st.metric(
                    label=ticker,
                    value=f"{end_price:.2f}",
                    delta=f"{perf:+.2f}%"
                )
    
    # ===== MATRICE DE CORRÉLATION =====
    if len(correlation_data.columns) > 1:
        st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 CORRELATION MATRIX")
        
        # Calculer la matrice de corrélation
        corr_matrix = correlation_data.corr()
        
        # Créer la heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0, '#FF0000'],      # Rouge pour corrélation négative
                [0.5, '#000000'],    # Noir pour 0
                [1, '#00FF00']       # Vert pour corrélation positive
            ],
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10, "color": "#FFAA00"},
            showscale=True,
            colorbar=dict(
                title="Corr",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0', '0.5', '1.0']
            )
        ))
        
        fig_corr.update_layout(
            title="Correlation Matrix (1.0 = parfaitement corrélé, -1.0 = inversement corrélé)",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                tickfont=dict(color='#FFAA00', size=9),
                showgrid=False
            ),
            yaxis=dict(
                tickfont=dict(color='#FFAA00', size=9),
                showgrid=False
            ),
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Interprétation
        st.caption("""
        **📖 Comment lire la matrice :**
        - 🟢 **Vert (proche de 1.0)** : Les actions évoluent ensemble (forte corrélation positive)
        - ⚫ **Noir (proche de 0)** : Pas de relation claire
        - 🔴 **Rouge (proche de -1.0)** : Les actions évoluent en sens inverse (corrélation négative)
        """)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 DONNÉES EN TEMPS RÉEL • YAHOO FINANCE + COINMARKETCAP<br>
        🔄 RAFRAÎCHISSEMENT AUTOMATIQUE: 3 SECONDES • AUCUNE ACTION REQUISE
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 CONNEXION: PARIS, FRANCE • SYSTÈME OPÉRATIONNEL
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE • COINMARKETCAP | SYSTÈME OPÉRATIONNEL<br>
    DONNÉES DE MARCHÉ DISPONIBLES • REFRESH AUTO: 3s • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)

# Auto-refresh JavaScript
st.markdown("""
<script>
    setTimeout(function(){
        window.parent.location.reload();
    }, 3000);
</script>
""", unsafe_allow_html=True)
