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

# CSS + JavaScript pour horloge en temps réel
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
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
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
    
    .stMultiSelect {
        background-color: #000;
    }
    
    .stMultiSelect > div {
        background-color: #000;
        border: 1px solid #FFAA00;
    }
    
    .stSelectbox select {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        font-family: 'Courier New', monospace;
    }
    
    hr {
        border-color: #333333;
        margin: 5px 0;
    }
    
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .live-clock {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
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
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    /* Chart container */
    .chart-section {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
    }
</style>

<script>
    // Horloge en temps réel
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
</script>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données réelles
@st.cache_data(ttl=60)
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

# Fonction pour normaliser en %
def normalize_to_percentage(df):
    """Normalise les prix à 100% au début"""
    if df is None or len(df) == 0:
        return None
    return (df / df.iloc[0]) * 100

# Fonction pour récupérer les news réelles
@st.cache_data(ttl=300)
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

# ===== BARRE DE RECHERCHE / NAVIGATION =====
st.markdown("""
<div class="command-line">
    <span class="prompt">FUNCTION></span>
    <span style="color: #666;">Tapez: PRICE, NEWS, SCREENER, PORTFOLIO, HELP...</span>
</div>
""", unsafe_allow_html=True)

# Formulaire pour gérer l'entrée
with st.form(key="nav_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        search_command = st.text_input(
            "",
            placeholder="Entrez une fonction (PRICE, NEWS, HELP...)",
            label_visibility="collapsed",
            key="command_input"
        )
    
    with col2:
        submit = st.form_submit_button("EXEC", use_container_width=True)
    
    if submit and search_command:
        cmd = search_command.upper().strip()
        
        if cmd == "PRICE" or cmd == "PRICING":
            st.info("📊 Page PRICING : Cliquez sur le bouton dans la sidebar →")
        elif cmd == "NEWS" or cmd == "N":
            st.info("📰 Page NEWS en construction...")
        elif cmd == "SCREENER" or cmd == "SCREEN":
            st.info("📊 Page SCREENER en construction...")
        elif cmd == "PORTFOLIO" or cmd == "PORT":
            st.info("💼 Page PORTFOLIO en construction...")
        elif cmd == "HELP" or cmd == "H":
            st.info("""
            **📋 FONCTIONS DISPONIBLES:**
            - PRICE / PRICING : Options pricing calculator
            - NEWS / N : Market news
            - SCREENER / SCREEN : Stock screener
            - PORTFOLIO / PORT : Portfolio tracker
            - HELP / H : Afficher cette aide
            """)
        else:
            st.warning(f"⚠️ Fonction '{cmd}' non reconnue. Tapez HELP pour voir les commandes.")

st.markdown('<div style="border-bottom: 1px solid #333; margin: 5px 0;"></div>', unsafe_allow_html=True)

# Bouton refresh manuel
col_r1, col_r2 = st.columns([5, 1])
with col_r2:
    if st.button("🔄 REFRESH", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===== MARKET OVERVIEW =====
st.markdown("### 📈 COMPARATIVE CHART - PERFORMANCE %")

col_chart1, col_chart2, col_chart3 = st.columns([3, 1, 1])

with col_chart1:
    # Liste de tickers populaires
    popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT', 
                       'SPY', 'QQQ', 'BTC-USD', 'ETH-USD']
    
    selected_tickers = st.multiselect(
        "Sélectionnez des tickers à comparer",
        options=popular_tickers,
        default=['AAPL', 'MSFT', 'TSLA'],
        help="Sélectionnez jusqu'à 5 tickers"
    )

with col_chart2:
    timeframe = st.selectbox(
        "Timeframe",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '5y'],
        index=3,
        help="Période d'analyse"
    )

with col_chart3:
    if st.button("📊 UPDATE", use_container_width=True, key="update_chart"):
        st.cache_data.clear()

# Génération du graphique
if selected_tickers:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Couleurs Bloomberg style
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00', 
              '#FF1493', '#00CED1', '#32CD32', '#FFD700']
    
    # Stocker les données pour la corrélation
    correlation_data = pd.DataFrame()
    
    with st.spinner('📊 Chargement des données...'):
        for idx, ticker in enumerate(selected_tickers[:10]):  # Limite à 10 tickers
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

st.markdown('<div style="border-bottom: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)

# ===== MAIN CONTENT =====

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

# ===== GRAPHIQUE MULTI-TICKERS COMPARATIF =====
st.markdown("### 📈 COMPARATIVE CHART - PERFORMANCE %")

# Liste COMPLÈTE des tickers S&P 500 (503 actions)
sp500_tickers = ['A', 'AAL', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBNY', 'SBUX', 'SCHW', 'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS']

# NASDAQ 100 (100 actions tech)
nasdaq_tickers = ['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'AZN', 'BIDU', 'BIIB', 'BKNG', 'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EBAY', 'ENPH', 'EXC', 'FANG', 'FAST', 'FISV', 'FTNT', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX', 'NTES', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'RIVN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZM', 'ZS']

# CAC 40 complet (40 actions)
cac40_tickers = ['MC.PA', 'OR.PA', 'SAN.PA', 'AIR.PA', 'AI.PA', 'BNP.PA', 'CS.PA', 'SU.PA', 'TTE.PA', 'BN.PA', 'CA.PA', 'ACA.PA', 'SGO.PA', 'GLE.PA', 'VIE.PA', 'DSY.PA', 'EN.PA', 'CAP.PA', 'DG.PA', 'RMS.PA', 'SAF.PA', 'ORA.PA', 'PUB.PA', 'KER.PA', 'URW.PA', 'RI.PA', 'ML.PA', 'STM.PA', 'TEP.PA', 'VIV.PA', 'SOP.PA', 'WLN.PA', 'BOL.PA', 'ERF.PA', 'EL.PA', 'BVI.PA', 'GET.PA', 'FP.PA', 'LR.PA', 'STLA.PA']

# SIX Swiss SMI (20 actions principales)
six_tickers = ['NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ABBN.SW', 'ZURN.SW', 'SREN.SW', 'GIVN.SW', 'CSGN.SW', 'SLHN.SW', 'LOGN.SW', 'SIKA.SW', 'GEBN.SW', 'SCMN.SW', 'ALC.SW', 'LONN.SW', 'CFR.SW', 'PGHN.SW', 'HOLN.SW', 'VATN.SW']

# Combiner toutes les listes
all_tickers = ['--- CRYPTO ---', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'MATIC-USD', 'DOT-USD', 'AVAX-USD',
               '--- NASDAQ 100 ---'] + sorted(list(set(nasdaq_tickers))) + \
              ['--- S&P 500 ---'] + sorted(list(set(sp500_tickers))) + \
              ['--- CAC 40 ---'] + sorted(cac40_tickers) + \
              ['--- SIX SWISS ---'] + sorted(six_tickers)

col_chart1, col_chart2, col_chart3 = st.columns([3, 1, 1])

with col_chart1:
    # Input texte pour contourner la limite du multiselect
    tickers_input = st.text_input(
        "Entrez les tickers séparés par des virgules (ex: AAPL, MSFT, TSLA, MC.PA, NESN.SW)",
        value="AAPL, MSFT, GOOGL",
        help="Max 10 tickers | S&P500: AAPL, MSFT | NASDAQ: NVDA, AMD | CAC40: MC.PA, OR.PA | SIX: NESN.SW, ROG.SW | Crypto: BTC-USD, ETH-USD"
    )
    # Convertir en liste et nettoyer
    selected_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    # Expander avec liste complète pour référence
    with st.expander("📋 Voir tous les tickers disponibles"):
        st.markdown("**🔷 CRYPTO:** BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD")
        st.markdown("**🔷 NASDAQ 100:** AAPL, MSFT, GOOGL, META, NVDA, TSLA, AMD, INTC, NFLX, AMZN...")
        st.markdown("**🔷 S&P 500:** Toutes les actions US (503 actions)")
        st.markdown("**🔷 CAC 40:** MC.PA (LVMH), OR.PA (L'Oréal), SAN.PA (Sanofi), AIR.PA (Airbus)...")
        st.markdown("**🔷 SIX Swiss:** NESN.SW (Nestlé), ROG.SW (Roche), NOVN.SW (Novartis)...")
        st.caption("Pour voir la liste complète des tickers, visitez : finance.yahoo.com")
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
    st.caption("📊 OPTIONS PRICING")
    st.caption("📈 SCREENER (soon)")
    st.caption("💼 PORTFOLIO (soon)")

# ===== FOOTER =====
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
last_update = datetime.now().strftime('%H:%M:%S')
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE | LAST UPDATE: {last_update}
</div>
""", unsafe_allow_html=True)
