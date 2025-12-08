import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from auth_utils import init_session_state

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
    }
    
    /* Style pour les lignes du tableau */
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: #000 !important;
    }
    
    /* Hover sur les lignes */
    .stDataFrame tbody tr:hover {
        background-color: #1a1a1a !important;
    }
    
    /* Style pour les heatmaps */
    .heatmap-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0px;
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
        <div>🔍 BLOOMBERG ENS® - MARKET HEATMAPS & SCREENER</div>
    </div>
    <div>{current_time} UTC • TRADINGVIEW + YAHOO FINANCE</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# HEATMAPS TRADINGVIEW
# =============================================
st.markdown("### 🔥 MARKET HEATMAPS - REAL TIME")

# Sélecteur de heatmap
heatmap_tabs = st.tabs(["📊 STOCKS US", "🌍 ETF GLOBAL", "₿ CRYPTO"])

with heatmap_tabs[0]:
    st.markdown("#### 📊 US STOCKS HEATMAP")
    
    # Widget TradingView Stock Heatmap
    stock_heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
      {
        "exchanges": [],
        "dataSource": "SPX500",
        "grouping": "sector",
        "blockSize": "market_cap_basic",
        "blockColor": "change",
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": true,
        "isDataSetEnabled": true,
        "isZoomEnabled": true,
        "hasSymbolTooltip": true,
        "width": "100%",
        "height": "600"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(stock_heatmap, height=650)
    
    st.caption("""
    **📖 Comment lire la heatmap :**
    - 🟢 **Vert** : Performance positive
    - 🔴 **Rouge** : Performance négative
    - **Taille des blocs** : Proportionnelle à la capitalisation boursière
    - **Groupement** : Par secteur (Technology, Healthcare, Finance, etc.)
    """)

with heatmap_tabs[1]:
    st.markdown("#### 🌍 GLOBAL ETF HEATMAP")
    
    # Widget TradingView ETF Heatmap
    etf_heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-etf-heatmap.js" async>
      {
        "dataSource": "AllUSEtf",
        "blockSize": "aum",
        "blockColor": "change",
        "grouping": "asset_class",
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": true,
        "isDataSetEnabled": true,
        "isZoomEnabled": true,
        "hasSymbolTooltip": true,
        "width": "100%",
        "height": "600"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(etf_heatmap, height=650)
    
    st.caption("""
    **📖 Comment lire la heatmap :**
    - 🟢 **Vert** : Performance positive
    - 🔴 **Rouge** : Performance négative
    - **Taille des blocs** : Proportionnelle aux actifs sous gestion (AUM)
    - **Groupement** : Par classe d'actifs (Equity, Bond, Commodity, etc.)
    """)

with heatmap_tabs[2]:
    st.markdown("#### ₿ CRYPTOCURRENCY HEATMAP")
    
    # Widget TradingView Crypto Heatmap
    crypto_heatmap = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js" async>
      {
        "dataSource": "Crypto",
        "blockSize": "market_cap_calc",
        "blockColor": "change",
        "locale": "en",
        "symbolUrl": "",
        "colorTheme": "dark",
        "hasTopBar": true,
        "isDataSetEnabled": true,
        "isZoomEnabled": true,
        "hasSymbolTooltip": true,
        "width": "100%",
        "height": "600"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    
    st.components.v1.html(crypto_heatmap, height=650)
    
    st.caption("""
    **📖 Comment lire la heatmap :**
    - 🟢 **Vert** : Performance positive
    - 🔴 **Rouge** : Performance négative
    - **Taille des blocs** : Proportionnelle à la capitalisation boursière
    - **Principales cryptos** : BTC, ETH, BNB, SOL, XRP, ADA, DOGE, etc.
    """)

st.markdown('<hr style="border-color: #FFAA00; margin: 30px 0;">', unsafe_allow_html=True)

# =============================================
# BASE DE DONNÉES DE TICKERS PAR SECTEUR
# =============================================
STOCK_DATABASE = {
    'Technology': {
        'Large Cap': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'INTU'],
        'Mid Cap': ['SNOW', 'DDOG', 'CRWD', 'PANW', 'FTNT', 'ZS', 'NET', 'MDB', 'PLTR', 'DOCU']
    },
    'Financial Services': {
        'Large Cap': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'BK', 'STT'],
        'Mid Cap': ['ALLY', 'SOFI', 'LC', 'UPST', 'AFRM']
    },
    'Healthcare': {
        'Large Cap': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY', 'AMGN', 'GILD', 'CVS', 'CI'],
        'Mid Cap': ['DXCM', 'ALGN', 'HOLX', 'INCY', 'TECH', 'PODD']
    },
    'Consumer Cyclical': {
        'Large Cap': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MAR', 'GM', 'F'],
        'Mid Cap': ['LULU', 'ULTA', 'DPZ', 'CMG', 'POOL', 'DECK']
    },
    'Consumer Defensive': {
        'Large Cap': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'GIS', 'KMB', 'HSY'],
        'Mid Cap': ['MNST', 'CHD', 'CLX', 'CPB', 'K']
    },
    'Energy': {
        'Large Cap': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
        'Mid Cap': ['FANG', 'DVN', 'MRO', 'HES', 'CTRA']
    },
    'Industrials': {
        'Large Cap': ['UPS', 'HON', 'RTX', 'UNP', 'BA', 'CAT', 'GE', 'LMT', 'DE', 'MMM', 'NOC', 'FDX'],
        'Mid Cap': ['IR', 'FAST', 'EXPD', 'JBHT', 'CHRW']
    },
    'Communication Services': {
        'Large Cap': ['GOOGL', 'META', 'NFLX', 'DIS', 'T', 'VZ', 'CMCSA', 'TMUS', 'CHTR'],
        'Mid Cap': ['MTCH', 'PINS', 'SNAP', 'RBLX', 'PARA']
    },
    'Real Estate': {
        'Large Cap': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'DLR', 'O', 'SBAC'],
        'Mid Cap': ['AVB', 'EQR', 'VTR', 'ARE', 'INVH']
    },
    'Utilities': {
        'Large Cap': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED'],
        'Mid Cap': ['WEC', 'ES', 'AWK', 'DTE', 'PPL']
    }
}

# =============================================
# FONCTIONS
# =============================================
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_info(ticker):
    """Récupère les infos complètes d'une action"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='5d')
        
        if len(hist) < 2:
            return None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        # Calculer le rendement du dividende annualisé
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield:
            dividend_yield = dividend_yield * 100  # Convertir en pourcentage
        
        return {
            'Ticker': ticker,
            'Name': info.get('shortName', ticker)[:30],
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown')[:30],
            'Price': current_price,
            'Change %': change_percent,
            'Volume': hist['Volume'].iloc[-1],
            'Avg Volume': info.get('averageVolume', 0),
            'Market Cap': info.get('marketCap', 0),
            'P/E Ratio': info.get('trailingPE', 0),
            'Forward P/E': info.get('forwardPE', 0),
            'PEG Ratio': info.get('pegRatio', 0),
            'Div Yield %': dividend_yield,
            'Beta': info.get('beta', 0),
            '52W High': info.get('fiftyTwoWeekHigh', 0),
            '52W Low': info.get('fiftyTwoWeekLow', 0),
            'Profit Margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'ROE %': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'Debt/Equity': info.get('debtToEquity', 0),
            'EPS': info.get('trailingEps', 0),
            'Revenue Growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
        }
    except Exception as e:
        return None

# =============================================
# INTERFACE DE FILTRAGE
# =============================================
st.markdown("### 🎯 STOCK SCREENER - CRITÈRES DE RECHERCHE")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 SECTEUR & CAPITALISATION**")
    
    selected_sectors = st.multiselect(
        "Secteurs",
        options=list(STOCK_DATABASE.keys()),
        default=list(STOCK_DATABASE.keys())[:3],
        help="Sélectionnez un ou plusieurs secteurs"
    )
    
    cap_type = st.multiselect(
        "Capitalisation",
        options=['Large Cap', 'Mid Cap'],
        default=['Large Cap', 'Mid Cap']
    )
    
    market_cap_min = st.number_input(
        "Market Cap min (Mds $)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="0 = pas de limite"
    )

with col2:
    st.markdown("**💹 VALORISATION**")
    
    pe_max = st.number_input(
        "P/E Ratio max",
        min_value=0.0,
        value=50.0,
        step=5.0,
        help="0 = pas de limite"
    )
    
    peg_max = st.number_input(
        "PEG Ratio max",
        min_value=0.0,
        value=3.0,
        step=0.5,
        help="PEG < 1 = sous-évalué, 0 = pas de limite"
    )
    
    div_yield_min = st.number_input(
        "Dividend Yield min (%)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Rendement du dividende minimum"
    )

with col3:
    st.markdown("**📈 PERFORMANCE & QUALITÉ**")
    
    roe_min = st.number_input(
        "ROE min (%)",
        min_value=0.0,
        value=0.0,
        step=5.0,
        help="Return on Equity minimum"
    )
    
    profit_margin_min = st.number_input(
        "Profit Margin min (%)",
        min_value=0.0,
        value=0.0,
        step=5.0,
        help="Marge bénéficiaire minimum"
    )
    
    debt_equity_max = st.number_input(
        "Debt/Equity max",
        min_value=0.0,
        value=200.0,
        step=10.0,
        help="Ratio d'endettement maximum, 0 = pas de limite"
    )

st.markdown('<hr>', unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 7])

with col_btn1:
    search_button = st.button("🔍 LANCER LE SCREENING", use_container_width=True)

with col_btn2:
    reset_button = st.button("🔄 RESET", use_container_width=True)

if reset_button:
    st.rerun()

# =============================================
# RECHERCHE
# =============================================
if search_button:
    st.markdown("### 📊 RÉSULTATS DU SCREENING")
    
    # Collecter les tickers à analyser
    tickers_to_analyze = []
    for sector in selected_sectors:
        for cap in cap_type:
            if cap in STOCK_DATABASE[sector]:
                tickers_to_analyze.extend(STOCK_DATABASE[sector][cap])
    
    # Supprimer les doublons
    tickers_to_analyze = list(set(tickers_to_analyze))
    
    if not tickers_to_analyze:
        st.warning("⚠️ Aucun ticker à analyser. Sélectionnez au moins un secteur.")
        st.stop()
    
    st.info(f"🔍 Analyse de {len(tickers_to_analyze)} actions...")
    
    # Récupérer les données
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers_to_analyze):
        status_text.text(f"Analyse de {ticker}... ({idx+1}/{len(tickers_to_analyze)})")
        data = get_stock_info(ticker)
        if data:
            results.append(data)
        progress_bar.progress((idx + 1) / len(tickers_to_analyze))
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("❌ Aucune donnée récupérée. Réessayez.")
        st.stop()
    
    # Créer le DataFrame
    df = pd.DataFrame(results)
    
    # Appliquer les filtres
    initial_count = len(df)
    
    if market_cap_min > 0:
        df = df[df['Market Cap'] >= market_cap_min * 1e9]
    
    if pe_max > 0:
        df = df[(df['P/E Ratio'] > 0) & (df['P/E Ratio'] <= pe_max)]
    
    if peg_max > 0:
        df = df[(df['PEG Ratio'] > 0) & (df['PEG Ratio'] <= peg_max)]
    
    if div_yield_min > 0:
        df = df[df['Div Yield %'] >= div_yield_min]
    
    if roe_min > 0:
        df = df[df['ROE %'] >= roe_min]
    
    if profit_margin_min > 0:
        df = df[df['Profit Margin'] >= profit_margin_min]
    
    if debt_equity_max > 0:
        df = df[(df['Debt/Equity'] >= 0) & (df['Debt/Equity'] <= debt_equity_max)]
    
    filtered_count = len(df)
    
    # Message de résultat
    if filtered_count == 0:
        st.warning(f"⚠️ Aucune action ne correspond aux critères. {initial_count} actions analysées.")
        st.stop()
    
    st.success(f"✅ **{filtered_count} actions trouvées** sur {initial_count} analysées")
    
    # ===== STATISTIQUES =====
    st.markdown("#### 📊 STATISTIQUES DU PORTEFEUILLE")
    
    stat_cols = st.columns(6)
    
    with stat_cols[0]:
        avg_pe = df[df['P/E Ratio'] > 0]['P/E Ratio'].mean()
        st.metric("P/E moyen", f"{avg_pe:.2f}")
    
    with stat_cols[1]:
        avg_div = df['Div Yield %'].mean()
        st.metric("Div. Yield moy.", f"{avg_div:.2f}%")
    
    with stat_cols[2]:
        avg_roe = df['ROE %'].mean()
        st.metric("ROE moyen", f"{avg_roe:.2f}%")
    
    with stat_cols[3]:
        avg_margin = df['Profit Margin'].mean()
        st.metric("Marge moy.", f"{avg_margin:.2f}%")
    
    with stat_cols[4]:
        avg_beta = df['Beta'].mean()
        st.metric("Beta moyen", f"{avg_beta:.2f}")
    
    with stat_cols[5]:
        total_mcap = df['Market Cap'].sum()
        st.metric("Cap. totale", f"${total_mcap/1e12:.2f}T")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== TABLEAU DES RÉSULTATS =====
    st.markdown("#### 📋 LISTE DES ACTIONS")
    
    # Préparer le DataFrame pour l'affichage
    df_display = df.copy()
    
    # Formater les colonnes
    df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}")
    df_display['Change %'] = df_display['Change %'].apply(lambda x: f"{x:+.2f}%")
    df_display['Volume'] = df_display['Volume'].apply(lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.2f}K")
    df_display['Market Cap'] = df_display['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B" if x > 1e9 else f"${x/1e6:.2f}M")
    df_display['P/E Ratio'] = df_display['P/E Ratio'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    df_display['Forward P/E'] = df_display['Forward P/E'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    df_display['PEG Ratio'] = df_display['PEG Ratio'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    df_display['Div Yield %'] = df_display['Div Yield %'].apply(lambda x: f"{x:.2f}%" if x > 0 else "N/A")
    df_display['Beta'] = df_display['Beta'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    df_display['Profit Margin'] = df_display['Profit Margin'].apply(lambda x: f"{x:.2f}%")
    df_display['ROE %'] = df_display['ROE %'].apply(lambda x: f"{x:.2f}%")
    df_display['Debt/Equity'] = df_display['Debt/Equity'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
    df_display['EPS'] = df_display['EPS'].apply(lambda x: f"${x:.2f}")
    df_display['Revenue Growth'] = df_display['Revenue Growth'].apply(lambda x: f"{x:.2f}%")
    
    # Sélectionner les colonnes à afficher
    display_cols = ['Ticker', 'Name', 'Sector', 'Price', 'Change %', 'Market Cap', 
                   'P/E Ratio', 'PEG Ratio', 'Div Yield %', 'ROE %', 'Profit Margin', 
                   'Beta', 'Debt/Equity']
    
    # Options de tri
    col_sort1, col_sort2, col_sort3 = st.columns([2, 2, 6])
    
    with col_sort1:
        sort_by = st.selectbox(
            "Trier par",
            options=['P/E Ratio', 'Div Yield %', 'ROE %', 'Market Cap', 'Change %', 'Profit Margin'],
            index=0
        )
    
    with col_sort2:
        sort_order = st.selectbox("Ordre", options=['Croissant', 'Décroissant'], index=0)
    
    # Trier (utiliser les valeurs originales)
    sort_map = {
        'P/E Ratio': 'P/E Ratio',
        'Div Yield %': 'Div Yield %',
        'ROE %': 'ROE %',
        'Market Cap': 'Market Cap',
        'Change %': 'Change %',
        'Profit Margin': 'Profit Margin'
    }
    
    ascending = (sort_order == 'Croissant')
    df_sorted = df.sort_values(by=sort_map[sort_by], ascending=ascending)
    df_display_sorted = df_display.loc[df_sorted.index]
    
    # Afficher le tableau
    st.dataframe(
        df_display_sorted[display_cols],
        use_container_width=True,
        height=500
    )
    
    # ===== EXPORT CSV =====
    csv = df_display_sorted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 EXPORTER EN CSV",
        data=csv,
        file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # ===== TOP PERFORMERS =====
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("#### 🏆 TOP PERFORMERS")
    
    col_top1, col_top2, col_top3 = st.columns(3)
    
    with col_top1:
        st.markdown("**🎯 MEILLEUR P/E RATIO**")
        top_pe = df[df['P/E Ratio'] > 0].nsmallest(5, 'P/E Ratio')
        for _, row in top_pe.iterrows():
            st.metric(f"{row['Ticker']}", f"P/E: {row['P/E Ratio']:.2f}", f"Price: ${row['Price']:.2f}")
    
    with col_top2:
        st.markdown("**💰 MEILLEURS DIVIDENDES**")
        top_div = df[df['Div Yield %'] > 0].nlargest(5, 'Div Yield %')
        for _, row in top_div.iterrows():
            st.metric(f"{row['Ticker']}", f"Yield: {row['Div Yield %']:.2f}%", f"Price: ${row['Price']:.2f}")
    
    with col_top3:
        st.markdown("**📈 MEILLEUR ROE**")
        top_roe = df[df['ROE %'] > 0].nlargest(5, 'ROE %')
        for _, row in top_roe.iterrows():
            st.metric(f"{row['Ticker']}", f"ROE: {row['ROE %']:.2f}%", f"Price: ${row['Price']:.2f}")

# =============================================
# AIDE
# =============================================
with st.expander("📖 AIDE - COMPRENDRE LES CRITÈRES"):
    st.markdown("""
    **📊 INDICATEURS DE VALORISATION:**
    
    - **P/E Ratio (Price-to-Earnings):** Prix / Bénéfice par action
      - < 15 : Potentiellement sous-évalué
      - 15-25 : Valorisation normale
      - > 25 : Potentiellement surévalué
    
    - **PEG Ratio (Price/Earnings to Growth):** P/E / Croissance des bénéfices
      - < 1 : Sous-évalué par rapport à la croissance
      - 1-2 : Valorisation raisonnable
      - > 2 : Surévalué
    
    - **Dividend Yield:** Dividende annuel / Prix de l'action
      - > 3% : Bon rendement
      - > 5% : Très bon rendement (vérifier la pérennité)
    
    **💪 INDICATEURS DE QUALITÉ:**
    
    - **ROE (Return on Equity):** Rentabilité des capitaux propres
      - > 15% : Bonne performance
      - > 20% : Excellente performance
    
    - **Profit Margin:** Marge bénéficiaire nette
      - > 10% : Bonne marge
      - > 20% : Excellente marge
    
    - **Debt/Equity:** Dette / Capitaux propres
      - < 50 : Faible endettement
      - 50-100 : Endettement modéré
      - > 100 : Endettement élevé
    
    **📈 AUTRES INDICATEURS:**
    
    - **Beta:** Volatilité par rapport au marché
      - < 1 : Moins volatil que le marché
      - = 1 : Volatilité du marché
      - > 1 : Plus volatil que le marché
    
    - **EPS (Earnings Per Share):** Bénéfice par action
    - **Revenue Growth:** Croissance du chiffre d'affaires
    """)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | TRADINGVIEW + YAHOO FINANCE | SYSTÈME OPÉRATIONNEL<br>
    HEATMAPS & SCREENER ACTIFS • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
