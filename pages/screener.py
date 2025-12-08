import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from auth_utils import init_session_state
import time

init_session_state()

if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Vous devez être connecté pour accéder à cette page.")
    st.stop()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Stock Screener",
    page_icon="🔍",
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
    
    .stApp {
        background-color: #000000 !important;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
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
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 10px !important;
        color: #FFAA00 !important;
        background-color: #111 !important;
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
# HEADER
# =============================================
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>🔍 BLOOMBERG ENS® - STOCK SCREENER</div>
    </div>
    <div>{current_time} UTC • YAHOO FINANCE (100% GRATUIT)</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# LISTE DE TICKERS PRÉ-DÉFINIE
# =============================================
# Liste complète des tickers populaires US
POPULAR_TICKERS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE', 
                   'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'IBM', 'QCOM', 'INTU', 'TXN', 'NOW'],
    'Financial Services': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SCHW', 'AXP',
                          'BLK', 'SPGI', 'C', 'CB', 'MMC', 'PGR', 'AON', 'USB', 'BK', 'TFC'],
    'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
                  'AMGN', 'GILD', 'CVS', 'MDT', 'CI', 'REGN', 'VRTX', 'ZTS', 'HUM', 'ISRG'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
                         'MAR', 'GM', 'F', 'ABNB', 'ROST', 'HLT', 'YUM', 'DRI', 'ULTA', 'ORLY'],
    'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'T', 'VZ', 'CMCSA', 'TMUS', 'CHTR', 'EA',
                               'TTWO', 'MTCH', 'SPOT', 'PINS', 'SNAP', 'PARA', 'WBD', 'OMC', 'IPG', 'FOXA'],
    'Consumer Defensive': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'GIS',
                          'KMB', 'SYY', 'HSY', 'K', 'TSN', 'CAG', 'STZ', 'TAP', 'CPB', 'HRL'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
              'BKR', 'WMB', 'KMI', 'HAL', 'DVN', 'FANG', 'MRO', 'APA', 'CTRA', 'OVV'],
    'Industrials': ['UPS', 'HON', 'RTX', 'UNP', 'BA', 'CAT', 'GE', 'LMT', 'DE', 'MMM',
                   'NOC', 'FDX', 'CSX', 'GD', 'NSC', 'EMR', 'ETN', 'ITW', 'PH', 'TDG'],
    'Basic Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'NUE', 'CTVA', 'DOW', 'DD',
                       'PPG', 'VMC', 'MLM', 'ALB', 'CF', 'MOS', 'IFF', 'CE', 'FMC', 'EMN'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'VICI',
                   'AVB', 'EQR', 'SPG', 'WY', 'INVH', 'ARE', 'VTR', 'EXR', 'MAA', 'SUI'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PCG', 'ED',
                 'WEC', 'ES', 'AWK', 'DTE', 'PEG', 'EIX', 'PPL', 'FE', 'AEE', 'CMS']
}

# Liste des indices
INDICES = {
    'S&P 500': ['^GSPC'],
    'DOW JONES': ['^DJI'],
    'NASDAQ': ['^IXIC'],
    'RUSSELL 2000': ['^RUT']
}

# =============================================
# FONCTIONS
# =============================================
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    """Récupère les données d'une action"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        return {
            'symbol': ticker,
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'price': current_price,
            'change': change_percent,
            'volume': hist['Volume'].iloc[-1],
            'marketCap': info.get('marketCap', 0),
            'beta': info.get('beta', 0),
            'dividendYield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'pe': info.get('trailingPE', 0),
            'high': hist['High'].iloc[-1],
            'low': hist['Low'].iloc[-1],
            'avgVolume': info.get('averageVolume', 0)
        }
    except:
        return None

def filter_stocks(stocks_data, filters):
    """Applique les filtres"""
    df = pd.DataFrame([s for s in stocks_data if s is not None])
    
    if df.empty:
        return df
    
    # Filtrer par prix
    if filters['price_min'] > 0:
        df = df[df['price'] >= filters['price_min']]
    if filters['price_max'] < 10000:
        df = df[df['price'] <= filters['price_max']]
    
    # Filtrer par market cap
    if filters['market_cap_min']:
        df = df[df['marketCap'] >= filters['market_cap_min']]
    if filters['market_cap_max']:
        df = df[df['marketCap'] <= filters['market_cap_max']]
    
    # Filtrer par volume
    if filters['volume_min']:
        df = df[df['volume'] >= filters['volume_min']]
    if filters['volume_max']:
        df = df[df['volume'] <= filters['volume_max']]
    
    # Filtrer par variation
    if filters['change_min'] is not None:
        df = df[df['change'] >= filters['change_min']]
    if filters['change_max'] is not None:
        df = df[df['change'] <= filters['change_max']]
    
    # Filtrer par dividend yield
    if filters['dividend_min'] > 0:
        df = df[df['dividendYield'] >= filters['dividend_min']]
    
    # Filtrer par beta
    if filters['beta_min'] is not None:
        df = df[df['beta'] >= filters['beta_min']]
    if filters['beta_max'] is not None:
        df = df[df['beta'] <= filters['beta_max']]
    
    # Filtrer par secteur
    if filters['sector'] != "All":
        df = df[df['sector'] == filters['sector']]
    
    return df

# =============================================
# INTERFACE DE FILTRAGE
# =============================================
st.markdown("### 🎯 FILTRES DE RECHERCHE")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**📊 MARCHÉ**")
    
    sector = st.selectbox(
        "Secteur",
        options=["All"] + list(POPULAR_TICKERS.keys()),
        index=0
    )

with col2:
    st.markdown("**💰 CAPITALISATION**")
    
    market_cap_range = st.select_slider(
        "Market Cap",
        options=["All", "Small (<2B)", "Mid (2B-10B)", "Large (10B-200B)", "Mega (>200B)"],
        value="All"
    )
    
    market_cap_min, market_cap_max = None, None
    if market_cap_range == "Small (<2B)":
        market_cap_max = 2000000000
    elif market_cap_range == "Mid (2B-10B)":
        market_cap_min = 2000000000
        market_cap_max = 10000000000
    elif market_cap_range == "Large (10B-200B)":
        market_cap_min = 10000000000
        market_cap_max = 200000000000
    elif market_cap_range == "Mega (>200B)":
        market_cap_min = 200000000000

with col3:
    st.markdown("**💹 PRIX & PERFORMANCE**")
    
    price_min = st.number_input("Prix min ($)", min_value=0.0, value=0.0, step=1.0)
    price_max = st.number_input("Prix max ($)", min_value=0.0, value=10000.0, step=10.0)
    
    change_range = st.selectbox(
        "Variation 24h",
        options=["All", "Strong Up (>5%)", "Up (0-5%)", "Down (0-5%)", "Strong Down (<-5%)"],
        index=0
    )
    
    change_min, change_max = None, None
    if change_range == "Strong Up (>5%)":
        change_min = 5.0
    elif change_range == "Up (0-5%)":
        change_min = 0.0
        change_max = 5.0
    elif change_range == "Down (0-5%)":
        change_min = -5.0
        change_max = 0.0
    elif change_range == "Strong Down (<-5%)":
        change_max = -5.0

with col4:
    st.markdown("**📈 DIVIDENDES & VOLATILITÉ**")
    
    dividend_min = st.number_input("Dividend Yield min (%)", min_value=0.0, value=0.0, step=0.5)
    
    beta_range = st.selectbox(
        "Volatilité (Beta)",
        options=["All", "Low Risk (<0.8)", "Medium (0.8-1.2)", "High Risk (>1.2)"],
        index=0
    )
    
    beta_min, beta_max = None, None
    if beta_range == "Low Risk (<0.8)":
        beta_max = 0.8
    elif beta_range == "Medium (0.8-1.2)":
        beta_min = 0.8
        beta_max = 1.2
    elif beta_range == "High Risk (>1.2)":
        beta_min = 1.2
    
    volume_min = st.number_input("Volume min", min_value=0, value=0, step=100000)

st.markdown('<hr>', unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns([2, 8])

with col_btn1:
    search_button = st.button("🔍 LANCER LA RECHERCHE", use_container_width=True)

with col_btn2:
    reset_button = st.button("🔄 RESET", use_container_width=True)

if reset_button:
    st.rerun()

# =============================================
# RÉSULTATS
# =============================================
if search_button:
    st.markdown("### 📊 RÉSULTATS DE LA RECHERCHE")
    
    # Sélectionner les tickers à analyser
    if sector == "All":
        tickers_to_analyze = []
        for sector_tickers in POPULAR_TICKERS.values():
            tickers_to_analyze.extend(sector_tickers[:10])  # 10 par secteur
    else:
        tickers_to_analyze = POPULAR_TICKERS[sector]
    
    with st.spinner(f'🔍 Analyse de {len(tickers_to_analyze)} actions...'):
        # Récupérer les données
        stocks_data = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(tickers_to_analyze):
            data = get_stock_data(ticker)
            if data:
                stocks_data.append(data)
            progress_bar.progress((idx + 1) / len(tickers_to_analyze))
            time.sleep(0.1)  # Éviter de surcharger Yahoo Finance
        
        progress_bar.empty()
        
        # Appliquer les filtres
        filters = {
            'price_min': price_min,
            'price_max': price_max,
            'market_cap_min': market_cap_min,
            'market_cap_max': market_cap_max,
            'volume_min': volume_min,
            'volume_max': None,
            'change_min': change_min,
            'change_max': change_max,
            'dividend_min': dividend_min,
            'beta_min': beta_min,
            'beta_max': beta_max,
            'sector': sector
        }
        
        filtered_df = filter_stocks(stocks_data, filters)
    
    if not filtered_df.empty:
        st.success(f"✅ **{len(filtered_df)} actions trouvées**")
        
        # Statistiques
        st.markdown("#### 📊 STATISTIQUES")
        
        stat_cols = st.columns(6)
        
        with stat_cols[0]:
            avg_change = filtered_df['change'].mean()
            st.metric("Var. moyenne", f"{avg_change:+.2f}%")
        
        with stat_cols[1]:
            avg_volume = filtered_df['volume'].mean()
            st.metric("Vol. moyen", f"{avg_volume/1e6:.2f}M")
        
        with stat_cols[2]:
            avg_price = filtered_df['price'].mean()
            st.metric("Prix moyen", f"${avg_price:.2f}")
        
        with stat_cols[3]:
            total_mcap = filtered_df['marketCap'].sum()
            st.metric("Cap. totale", f"${total_mcap/1e9:.2f}B")
        
        with stat_cols[4]:
            avg_div = filtered_df['dividendYield'].mean()
            st.metric("Div. moyen", f"{avg_div:.2f}%")
        
        with stat_cols[5]:
            avg_beta = filtered_df['beta'].mean()
            st.metric("Beta moyen", f"{avg_beta:.2f}")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Préparer le DataFrame pour l'affichage
        df_display = filtered_df.copy()
        df_display = df_display.rename(columns={
            'symbol': 'Ticker',
            'name': 'Company',
            'sector': 'Sector',
            'price': 'Price ($)',
            'change': 'Change (%)',
            'marketCap': 'Market Cap',
            'volume': 'Volume',
            'dividendYield': 'Div. Yield (%)',
            'beta': 'Beta',
            'pe': 'P/E'
        })
        
        # Formater
        df_display['Market Cap'] = df_display['Market Cap'].apply(
            lambda x: f"{x/1e9:.2f}B" if x > 1e9 else f"{x/1e6:.2f}M"
        )
        df_display['Volume'] = df_display['Volume'].apply(
            lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.2f}K"
        )
        df_display['Price ($)'] = df_display['Price ($)'].apply(lambda x: f"${x:.2f}")
        df_display['Change (%)'] = df_display['Change (%)'].apply(lambda x: f"{x:+.2f}%")
        df_display['Div. Yield (%)'] = df_display['Div. Yield (%)'].apply(lambda x: f"{x:.2f}%")
        df_display['Beta'] = df_display['Beta'].apply(lambda x: f"{x:.2f}")
        df_display['P/E'] = df_display['P/E'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
        
        # Colonnes à afficher
        display_cols = ['Ticker', 'Company', 'Sector', 'Price ($)', 'Change (%)', 
                       'Market Cap', 'Volume', 'Div. Yield (%)', 'Beta', 'P/E']
        
        st.markdown("#### 📋 LISTE DES ACTIONS")
        
        col_sort1, col_sort2 = st.columns([3, 9])
        
        with col_sort1:
            sort_by = st.selectbox("Trier par", options=display_cols, index=4)
        
        # Trier (utiliser les valeurs originales)
        sort_map = {
            'Ticker': 'symbol', 'Company': 'name', 'Sector': 'sector',
            'Price ($)': 'price', 'Change (%)': 'change', 'Market Cap': 'marketCap',
            'Volume': 'volume', 'Div. Yield (%)': 'dividendYield', 'Beta': 'beta', 'P/E': 'pe'
        }
        filtered_df_sorted = filtered_df.sort_values(by=sort_map[sort_by], ascending=False)
        df_display_sorted = df_display.loc[filtered_df_sorted.index]
        
        st.dataframe(df_display_sorted[display_cols], use_container_width=True, height=400)
        
        # Export CSV
        csv = df_display_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 EXPORTER EN CSV",
            csv,
            f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
        
        # Top Gainers/Losers
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("#### 🏆 TOP GAINERS & LOSERS")
        
        col_gain, col_lose = st.columns(2)
        
        with col_gain:
            st.markdown("**🟢 TOP 5 GAINERS**")
            top_gainers = filtered_df.nlargest(5, 'change')
            for _, row in top_gainers.iterrows():
                st.metric(f"{row['symbol']}", f"${row['price']:.2f}", f"{row['change']:+.2f}%")
        
        with col_lose:
            st.markdown("**🔴 TOP 5 LOSERS**")
            top_losers = filtered_df.nsmallest(5, 'change')
            for _, row in top_losers.iterrows():
                st.metric(f"{row['symbol']}", f"${row['price']:.2f}", f"{row['change']:+.2f}%")
        
        # Graphique par secteur
        if len(filtered_df['sector'].unique()) > 1:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("#### 📊 RÉPARTITION PAR SECTEUR")
            
            sector_counts = filtered_df['sector'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                hole=0.3,
                marker=dict(colors=['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000'])
            )])
            
            fig.update_layout(
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Aucun résultat. Modifiez vos critères.")

# Footer
st.markdown('<hr>', unsafe_allow_html=True)
st.info("""
📌 **NOTE:** Ce screener utilise Yahoo Finance (100% gratuit).
- Base de données : ~200 actions populaires US
- Données en temps réel
- Aucune limitation d'API
""")

st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | YAHOO FINANCE | {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
