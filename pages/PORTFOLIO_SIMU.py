import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from supabase import create_client, Client
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# ============================================================================
# CONFIG STREAMLIT - STYLE BLOOMBERG
# ============================================================================
st.set_page_config(page_title="Stock Chart Analyzer", page_icon="📊", layout="wide")

# CSS personnalisé style Bloomberg
st.markdown("""
<style>
    /* Fond noir Bloomberg */
    .stApp {
        background-color: #000000;
    }
    
    /* Texte en orange Bloomberg */
    .stMarkdown, .stMetric label, p {
        color: #FF8C00 !important;
    }
    
    /* Valeurs des métriques en blanc */
    .stMetric .metric-value {
        color: #FFFFFF !important;
        font-weight: bold;
        font-size: 28px !important;
    }
    
    /* Delta des métriques */
    [data-testid="stMetricDelta"] {
        color: #00FF00 !important;
    }
    
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #FF8C00 !important;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    /* Divider */
    hr {
        border-color: #FF8C00 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #FF8C00 !important;
    }
    
    /* Radio buttons */
    .stRadio > label, .stRadio div {
        color: #FF8C00 !important;
    }
    
    /* Select box */
    .stSelectbox > label, .stSelectbox div {
        color: #FF8C00 !important;
    }
    
    /* Multi-select */
    .stMultiSelect > label, .stMultiSelect div {
        color: #FF8C00 !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #FF8C00 !important;
    }
    
    /* Date input */
    .stDateInput > label {
        color: #FF8C00 !important;
    }
    
    /* Warning box */
    .stAlert {
        background-color: #1a1a1a !important;
        color: #FF8C00 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 STOCK CHART ANALYZER")

# =============================================
# BARRE DE COMMANDE BLOOMBERG
# À ajouter après le header, avant les données de marché
# =============================================

# Style pour la barre de commande
st.markdown("""
<style>
    .command-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 10px 15px;
        margin: 10px 0 20px 0;
    }
    .command-prompt {
        color: #FFAA00;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Dictionnaire des commandes et leurs pages
COMMANDS = {
    "EDGAR": "pages/EDGAR.py",
    "NEWS": "pages/NEWS.py",
    "PRICE": "pages/PRICING.py",
    "CHAT": "pages/CHATBOT.py",
    "BT": "pages/BACKTESTING.py",
    "ANA": "pages/COMPANY_ANALYSIS.py",
    "CRYPTO":"pages/CRYPTO_SCRAPER.py",
    "ECO":"pages/ECONOMICS.py", 
    "EU":"pages/EUROPE.py",
    "SIMU":"pages/PORTFOLIO_SIMU.py",
    "PY":"pages/PYTHON_EDITOR.py",
    "SQL":"pages/SQL_EDITOR.py",
    "BONDS":"pages/BONDS.py",
    "HOME":"pages/HOME.py",
}

# Affichage de la barre de commande
st.markdown('<div class="command-container">', unsafe_allow_html=True)

col_prompt, col_input = st.columns([1, 11])

with col_prompt:
    st.markdown('<span class="command-prompt">BBG&gt;</span>', unsafe_allow_html=True)

with col_input:
    command_input = st.text_input(
        "Command",
        placeholder="Tapez une commande: EDGAR, NEWS, CHATBOT, PRICING, HELP...",
        label_visibility="collapsed",
        key="bloomberg_command"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Traitement de la commande
if command_input:
    cmd = command_input.upper().strip()
    
    if cmd == "HELP" or cmd == "H":
        st.info("""
        **📋 COMMANDES DISPONIBLES:**
        - `EDGAR` → SEC Filings & Documents
        - `NEWS` → Market News Feed
        - `CHAT` → AI Assistant
        - `PRICE` → Options Pricing
        - `HELP` → Afficher cette aide
        - `BT` → Backesting de strategies
        - `ANA` → Analyse financière de sociétés côtées
        - `CRYPTO` → Scrapping et backtest de strategies liées aux cryptos
        - `ECO` → Données économiques
        - `EU` → Données Européennes
        - `SIMU` → Simulation de portefeuille
        - `PY` → Editeur de code python 
        - `SQL` → Editeur de code SQL
        - `BONDS` → Screener d'obligation
        - `HOME` → Menu
        """)
    elif cmd in COMMANDS:
        st.switch_page(COMMANDS[cmd])
    else:
        st.warning(f"⚠️ Commande '{cmd}' non reconnue. Tapez HELP pour voir les commandes disponibles.")


# ============================================================================
# PARAMETRES SUPABASE
# ============================================================================

SUPABASE_URL = "https://gbrefcefeavmqupulzyw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdicmVmY2VmZWF2bXF1cHVsenl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0OTA2NjksImV4cCI6MjA3OTA2NjY2OX0.WsA-3so0J52hAyZTIddVT0qqLuvcxjHYTZ4XkZ5mMio"

# ============================================================================
# RECUPERATION DE LA LISTE DES TABLES
# ============================================================================

@st.cache_data(ttl=3600)
def get_all_tables():
    """Récupère la liste de toutes les tables disponibles"""
    try:
        tables = [
            'a_h4_data', 'aal_h4_data', 'aapl_h4_data', 'abbv_h4_data', 'abnb_h4_data', 
            'abt_h4_data', 'acgl_h4_data', 'acn_h4_data', 'adbe_h4_data', 'adi_h4_data',
            'adm_h4_data', 'adp_h4_data', 'adsk_h4_data', 'aee_h4_data', 'aep_h4_data',
            'aes_h4_data', 'afl_h4_data', 'aig_h4_data', 'aiz_h4_data', 'ajg_h4_data',
            'akam_h4_data', 'alb_h4_data', 'algn_h4_data', 'all_h4_data', 'alle_h4_data',
            'amat_h4_data', 'amcr_h4_data', 'amd_h4_data', 'ame_h4_data', 'amgn_h4_data',
            'amp_h4_data', 'amt_h4_data', 'amzn_h4_data', 'anet_h4_data', 'aon_h4_data',
            'aos_h4_data', 'apa_h4_data', 'apd_h4_data', 'aph_h4_data', 'aptv_h4_data',
            'are_h4_data', 'ato_h4_data', 'avb_h4_data', 'avgo_h4_data', 'avy_h4_data',
            'awk_h4_data', 'axon_h4_data', 'axp_h4_data', 'azo_h4_data', 'ba_h4_data',
            'bac_h4_data', 'ball_h4_data', 'bax_h4_data', 'bbwi_h4_data', 'bby_h4_data',
            'bdx_h4_data', 'ben_h4_data', 'bf_b_h4_data', 'bg_h4_data', 'biib_h4_data',
            'bio_h4_data', 'bk_h4_data', 'bkng_h4_data', 'bkr_h4_data', 'blk_h4_data',
            'bmy_h4_data', 'br_h4_data', 'brk_b_h4_data', 'bro_h4_data', 'bsx_h4_data',
            'bwa_h4_data', 'bx_h4_data', 'bxp_h4_data', 'c_h4_data', 'cag_h4_data',
            'cah_h4_data', 'carr_h4_data', 'cat_h4_data', 'cb_h4_data', 'cboe_h4_data',
            'cbre_h4_data', 'cci_h4_data', 'ccl_h4_data', 'cdns_h4_data', 'cdw_h4_data',
            'meta_h4_data', 'msft_h4_data', 'nvda_h4_data', 'morgan_stanley_h4_data',
            'qqq_h4_data', 'vixm_h4_data'
        ]
        
        return sorted(tables)
    except:
        return ['morgan_stanley_h4_data']

# ============================================================================
# RECUPERATION DES DONNEES
# ============================================================================

@st.cache_data(ttl=600)
def load_data(table_name):
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Récupérer TOUTES les données avec pagination
        all_data = []
        batch_size = 1000
        offset = 0
        
        while True:
            response = supabase.table(table_name).select("*").order("date", desc=False).range(offset, offset + batch_size - 1).execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            # Si on a récupéré moins que batch_size, on a tout récupéré
            if len(response.data) < batch_size:
                break
            
            offset += batch_size
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Supprimer les doublons potentiels et trier
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data for {table_name}: {str(e)}")
        return None

# ============================================================================
# FONCTION POUR NORMALISER LES DONNEES SUR BASE 100
# ============================================================================

def normalize_to_base_100(df, column='close'):
    """Normalise une série de prix sur base 100"""
    if len(df) == 0:
        return pd.Series()
    first_value = df[column].iloc[0]
    return (df[column] / first_value) * 100

# ============================================================================
# SIDEBAR - PARAMETRES
# ============================================================================

st.sidebar.header("⚙️ CHART SETTINGS")

# Onglets pour sélection unique ou multiple
chart_mode = st.sidebar.radio(
    "Chart Mode:",
    ["📊 Single Stock Analysis", "📈 Multi-Stock Comparison", "🚀 Momentum Strategy"],
    index=0
)

available_tables = get_all_tables()
table_names_display = [t.replace('_h4_data', '').upper() for t in available_tables]
table_dict = dict(zip(table_names_display, available_tables))

if chart_mode == "📊 Single Stock Analysis":
    # MODE UNIQUE - Sélection simple
    selected_display = st.sidebar.selectbox(
        "Select Stock:",
        table_names_display,
        index=table_names_display.index('MORGAN_STANLEY') if 'MORGAN_STANLEY' in table_names_display else 0
    )
    
    selected_tables = [table_dict[selected_display]]
    selected_displays = [selected_display]

elif chart_mode == "🚀 Momentum Strategy":
    # MODE MOMENTUM - Sélection de 2 tickers
    st.sidebar.markdown("### Select 2 Assets for Momentum Strategy")
    
    default_ticker1 = 'QQQ' if 'QQQ' in table_names_display else table_names_display[0]
    default_ticker2 = 'VIXM' if 'VIXM' in table_names_display else table_names_display[1]
    
    ticker1 = st.sidebar.selectbox(
        "Primary Asset (Buy & Hold):",
        table_names_display,
        index=table_names_display.index(default_ticker1)
    )
    
    ticker2 = st.sidebar.selectbox(
        "Trading Asset (RSI Strategy):",
        table_names_display,
        index=table_names_display.index(default_ticker2)
    )
    
    selected_displays = [ticker1, ticker2]
    selected_tables = [table_dict[ticker1], table_dict[ticker2]]
    
else:
    # MODE MULTIPLE - Multi-sélection
    default_stocks = ['MORGAN_STANLEY', 'AAPL', 'MSFT', 'NVDA'] if all(s in table_names_display for s in ['MORGAN_STANLEY', 'AAPL', 'MSFT', 'NVDA']) else table_names_display[:4]
    
    selected_displays = st.sidebar.multiselect(
        "Select Stocks to Compare:",
        table_names_display,
        default=default_stocks
    )
    
    if not selected_displays:
        st.warning("⚠️ Please select at least one stock to display")
        st.stop()
    
    selected_tables = [table_dict[display] for display in selected_displays]

# Charger les données pour tous les tickers sélectionnés
data_dict = {}
with st.spinner("📥 Loading data from database..."):
    for table, display in zip(selected_tables, selected_displays):
        df = load_data(table)
        if df is not None:
            data_dict[display] = df

if not data_dict:
    st.error("❌ No data could be loaded!")
    st.stop()

# Utiliser le premier ticker pour les paramètres de période
df_reference = list(data_dict.values())[0]

# ============================================================================
# PARAMETRES DE PERIODE
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Period Selection")

period_options = {
    "Last 50 Candles": 50,
    "Last 100 Candles": 100,
    "Last 200 Candles": 200,
    "Last 500 Candles": 500,
    "Last 1000 Candles": 1000,
    "Custom Number": -1,
    "Custom Date Range": -2
}

period_choice = st.sidebar.selectbox(
    "Choose Period:",
    list(period_options.keys()),
    index=3
)

period_value = period_options[period_choice]

# Filtrage des données
data_filtered = {}

if period_value > 0:
    num_candles = period_value
    for ticker, df in data_dict.items():
        data_filtered[ticker] = df.tail(min(num_candles, len(df)))
        
elif period_value == -1:
    max_candles = min([len(df) for df in data_dict.values()])
    num_candles = st.sidebar.slider(
        "Number of candles:",
        min_value=10,
        max_value=min(max_candles, 5000),
        value=min(500, max_candles),
        step=10
    )
    for ticker, df in data_dict.items():
        data_filtered[ticker] = df.tail(num_candles)
        
else:
    st.sidebar.markdown("**Select Date Range:**")
    
    min_date = min([df.index.min().date() for df in data_dict.values()])
    max_date = max([df.index.max().date() for df in data_dict.values()])
    
    st.sidebar.info(f"📅 Available data: {min_date} to {max_date}")
    
    date_preset = st.sidebar.selectbox(
        "Quick Date Range:",
        ["Custom", "Last Week", "Last Month", "Last 3 Months", "Last 6 Months", "Last Year", "Year to Date", "All Data"]
    )
    
    if date_preset == "Last Week":
        start_date = max_date - timedelta(days=7)
        end_date = max_date
    elif date_preset == "Last Month":
        start_date = max_date - timedelta(days=30)
        end_date = max_date
    elif date_preset == "Last 3 Months":
        start_date = max_date - timedelta(days=90)
        end_date = max_date
    elif date_preset == "Last 6 Months":
        start_date = max_date - timedelta(days=180)
        end_date = max_date
    elif date_preset == "Last Year":
        start_date = max_date - timedelta(days=365)
        end_date = max_date
    elif date_preset == "Year to Date":
        start_date = datetime(max_date.year, 1, 1).date()
        end_date = max_date
    elif date_preset == "All Data":
        start_date = min_date
        end_date = max_date
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "From:",
                value=max_date - timedelta(days=90),
                min_value=min_date,
                max_value=max_date,
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "To:",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date"
            )
    
    if date_preset != "Custom":
        st.sidebar.write(f"**From:** {start_date}")
        st.sidebar.write(f"**To:** {end_date}")
    
    if start_date > end_date:
        st.sidebar.error("⚠️ Start date must be before end date!")
        st.stop()
    
    for ticker, df in data_dict.items():
        data_filtered[ticker] = df.loc[start_date:end_date]
    
    if all(len(df) == 0 for df in data_filtered.values()):
        st.warning(f"⚠️ No data available between {start_date} and {end_date}!")
        st.stop()

# ============================================================================
# AUTRES PARAMETRES
# ============================================================================

st.sidebar.markdown("---")

if chart_mode == "📊 Single Stock Analysis":
    chart_type = st.sidebar.radio(
        "Chart Type:",
        ["📈 Line Chart", "🕯️ Candlestick"],
        index=1
    )
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
elif chart_mode == "🚀 Momentum Strategy":
    # Les paramètres seront dans la page principale
    pass
else:
    show_ma = st.sidebar.checkbox("Show Moving Averages", value=False)

# ============================================================================
# AFFICHAGE MODE SINGLE STOCK
# ============================================================================

if chart_mode == "📊 Single Stock Analysis":
    ticker = selected_displays[0]
    df_filtered = data_filtered[ticker]
    
    # Stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    last_price = df_filtered['close'].iloc[-1]
    prev_price = df_filtered['close'].iloc[-2]
    price_change = last_price - prev_price
    pct_change = (price_change / prev_price) * 100
    
    with col1:
        st.metric("LAST PRICE", f"${last_price:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    with col2:
        st.metric("HIGH", f"${df_filtered['high'].max():.2f}")
    with col3:
        st.metric("LOW", f"${df_filtered['low'].min():.2f}")
    with col4:
        st.metric("VOLUME", f"{df_filtered['volume'].sum()/1e6:.1f}M")
    with col5:
        st.metric("CANDLES", f"{len(df_filtered):,}")
    
    st.markdown(f"**PERIOD:** {df_filtered.index[0].strftime('%Y-%m-%d %H:%M')} to {df_filtered.index[-1].strftime('%Y-%m-%d %H:%M')}")
    
    st.divider()
    
    # Graphique
    if chart_type == "📈 Line Chart":
        st.subheader(f"📈 {ticker} - LINE CHART")
        
        if show_ma:
            df_filtered['SMA50'] = df_filtered['close'].rolling(window=50).mean()
            df_filtered['SMA100'] = df_filtered['close'].rolling(window=100).mean()
            df_filtered['SMA200'] = df_filtered['close'].rolling(window=200).mean()
        
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                            gridspec_kw={'height_ratios': [3, 1]},
                                            facecolor='#000000')
        else:
            fig, ax1 = plt.subplots(figsize=(16, 8), facecolor='#000000')
        
        ax1.set_facecolor('#0a0a0a')
        ax1.plot(df_filtered.index, df_filtered['close'], 
                linewidth=2, color='#00D9FF', label='Close', zorder=5)
        
        if show_ma:
            if len(df_filtered) >= 50:
                ax1.plot(df_filtered.index, df_filtered['SMA50'], 
                        linewidth=1.5, color='#FF1493', label='SMA 50', alpha=0.8)
            if len(df_filtered) >= 100:
                ax1.plot(df_filtered.index, df_filtered['SMA100'], 
                        linewidth=1.5, color='#00FF00', label='SMA 100', alpha=0.8)
            if len(df_filtered) >= 200:
                ax1.plot(df_filtered.index, df_filtered['SMA200'], 
                        linewidth=1.5, color='#FFD700', label='SMA 200', alpha=0.8)
        
        ax1.set_title(f'{ticker} - H4 Chart', 
                     fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
        ax1.set_ylabel('Price ($)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax1.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                  fontsize=10, labelcolor='#FF8C00')
        ax1.tick_params(colors='#FFFFFF', labelsize=10)
        
        if show_volume:
            ax2.set_facecolor('#0a0a0a')
            colors = ['#4169E1' if df_filtered['close'].iloc[i] >= df_filtered['open'].iloc[i] 
                     else '#808080' for i in range(len(df_filtered))]
            ax2.bar(df_filtered.index, df_filtered['volume'], color=colors, alpha=0.6, width=0.8)
            ax2.set_ylabel('Volume', fontsize=12, color='#FF8C00', fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
            ax2.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
            ax2.tick_params(colors='#FFFFFF', labelsize=10)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax1.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        st.pyplot(fig)
    
    else:
        st.subheader(f"🕯️ {ticker} - CANDLESTICK CHART")
        
        mc = mpf.make_marketcolors(
            up='#4169E1',
            down='#808080',
            edge='inherit',
            wick='inherit',
            volume={'up': '#4169E1', 'down': '#808080'},
            alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='#333333',
            facecolor='#0a0a0a',
            figcolor='#000000',
            gridaxis='both',
            y_on_right=False
        )
        
        apds = []
        if show_ma:
            if len(df_filtered) >= 50:
                apds.append(mpf.make_addplot(df_filtered['close'].rolling(50).mean(), 
                                             color='#FF1493', width=1.5))
            if len(df_filtered) >= 100:
                apds.append(mpf.make_addplot(df_filtered['close'].rolling(100).mean(), 
                                             color='#00FF00', width=1.5))
            if len(df_filtered) >= 200:
                apds.append(mpf.make_addplot(df_filtered['close'].rolling(200).mean(), 
                                             color='#FFD700', width=1.5))
        
        kwargs = {
            'type': 'candle',
            'style': s,
            'title': f'{ticker} - H4 Candlestick Chart',
            'ylabel': 'Price ($)',
            'volume': show_volume,
            'figsize': (16, 10) if show_volume else (16, 8),
            'returnfig': True,
            'warn_too_much_data': 10000
        }
        
        if show_volume:
            kwargs['ylabel_lower'] = 'Volume'
        
        if apds:
            kwargs['addplot'] = apds
        
        fig, axes = mpf.plot(df_filtered, **kwargs)
        
        for ax in axes:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='#FFFFFF', labelsize=10)
            ax.spines['bottom'].set_color('#FF8C00')
            ax.spines['top'].set_color('#FF8C00')
            ax.spines['left'].set_color('#FF8C00')
            ax.spines['right'].set_color('#FF8C00')
            ax.yaxis.label.set_color('#FF8C00')
            ax.xaxis.label.set_color('#FF8C00')
            ax.title.set_color('#FF8C00')
        
        st.pyplot(fig)
    
    # Tableau de données
    st.divider()
    st.subheader("📋 RECENT DATA")
    
    display_df = df_filtered.tail(20).sort_index(ascending=False).copy()
    display_df['Change'] = display_df['close'] - display_df['open']
    display_df['Change %'] = (display_df['Change'] / display_df['open'] * 100).round(2)
    
    display_df = display_df[['open', 'high', 'low', 'close', 'volume', 'Change', 'Change %']]
    
    display_df['open'] = display_df['open'].apply(lambda x: f"${x:.2f}")
    display_df['high'] = display_df['high'].apply(lambda x: f"${x:.2f}")
    display_df['low'] = display_df['low'].apply(lambda x: f"${x:.2f}")
    display_df['close'] = display_df['close'].apply(lambda x: f"${x:.2f}")
    display_df['volume'] = display_df['volume'].apply(lambda x: f"{int(x):,}")
    display_df['Change'] = display_df['Change'].apply(lambda x: f"${x:.2f}")
    display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(display_df, use_container_width=True)

# ============================================================================
# AFFICHAGE MODE MULTI-STOCK COMPARISON
# ============================================================================

elif chart_mode == "📈 Multi-Stock Comparison":
    st.subheader("📈 MULTI-STOCK COMPARISON - BASE 100")
    
    # Stats comparatives
    cols = st.columns(len(selected_displays))
    
    for i, ticker in enumerate(selected_displays):
        df = data_filtered[ticker]
        if len(df) > 0:
            last_price = df['close'].iloc[-1]
            first_price = df['close'].iloc[0]
            total_change_pct = ((last_price - first_price) / first_price) * 100
            
            with cols[i]:
                st.metric(
                    ticker,
                    f"${last_price:.2f}",
                    f"{total_change_pct:+.2f}%"
                )
    
    st.divider()
    
    # Graphique de comparaison sur base 100
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#000000')
    ax.set_facecolor('#0a0a0a')
    
    # Palette de couleurs distinctes
    colors = ['#00D9FF', '#FF1493', '#00FF00', '#FFD700', '#FF4500', '#9370DB', '#FF69B4', '#00CED1', '#FFA500', '#7FFF00']
    
    # Tracer chaque ticker normalisé sur base 100
    for i, ticker in enumerate(selected_displays):
        df = data_filtered[ticker]
        if len(df) > 0:
            normalized = normalize_to_base_100(df, 'close')
            ax.plot(df.index, normalized, 
                   linewidth=2.5, 
                   color=colors[i % len(colors)], 
                   label=ticker, 
                   alpha=0.9)
            
            # Optionnel: ajouter moyennes mobiles
            if show_ma and len(df) >= 50:
                ma50 = (df['close'].rolling(window=50).mean() / df['close'].iloc[0]) * 100
                ax.plot(df.index, ma50, 
                       linewidth=1, 
                       color=colors[i % len(colors)], 
                       alpha=0.3, 
                       linestyle='--')
    
    # Ligne de référence à 100
    if len(data_filtered) > 0:
        first_date = min([df.index[0] for df in data_filtered.values()])
        last_date = max([df.index[-1] for df in data_filtered.values()])
        ax.axhline(y=100, color='#FF8C00', linestyle='--', linewidth=1.5, alpha=0.5, label='Base 100')
    
    ax.set_title('Stock Performance Comparison (Base 100)', 
                 fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
    ax.set_ylabel('Performance (Base 100)', fontsize=12, color='#FF8C00', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
    ax.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
    ax.legend(loc='best', facecolor='#1a1a1a', edgecolor='#FF8C00', 
              fontsize=11, labelcolor='#FF8C00', ncol=2)
    ax.tick_params(colors='#FFFFFF', labelsize=10)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Tableau de performance
    st.divider()
    st.subheader("📊 PERFORMANCE SUMMARY")
    
    perf_data = []
    for ticker in selected_displays:
        df = data_filtered[ticker]
        if len(df) > 1:
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            total_change = last_price - first_price
            total_change_pct = (total_change / first_price) * 100
            high = df['high'].max()
            low = df['low'].min()
            avg_volume = df['volume'].mean()
            
            perf_data.append({
                'Ticker': ticker,
                'Start Price': f"${first_price:.2f}",
                'End Price': f"${last_price:.2f}",
                'Change': f"${total_change:+.2f}",
                'Change %': f"{total_change_pct:+.2f}%",
                'High': f"${high:.2f}",
                'Low': f"${low:.2f}",
                'Avg Volume': f"{avg_volume/1e6:.1f}M"
            })
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True)
    
    # ============================================================================
    # ANALYSE DE PORTEFEUILLE
    # ============================================================================
    
    st.divider()
    st.header("💼 PORTFOLIO ANALYSIS")
    
    # Configuration du portefeuille
    st.subheader("⚙️ Portfolio Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Set Portfolio Weights:**")
        weights = {}
        total_weight = 0
        
        # Créer des sliders pour chaque ticker
        for ticker in selected_displays:
            default_weight = 100 / len(selected_displays)  # Répartition égale par défaut
            weight = st.slider(
                f"{ticker} Weight (%):",
                min_value=0.0,
                max_value=100.0,
                value=default_weight,
                step=0.5,
                key=f"weight_{ticker}"
            )
            weights[ticker] = weight
            total_weight += weight
        
        # Vérification des poids
        if abs(total_weight - 100.0) > 0.01:
            st.warning(f"⚠️ Total weight is {total_weight:.2f}% (should be 100%)")
        else:
            st.success(f"✅ Total weight: {total_weight:.2f}%")
    
    with col2:
        st.markdown("**Portfolio Settings:**")
        initial_capital = st.number_input(
            "Initial Capital ($):",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000
        )
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%):",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        ) / 100  # Convertir en décimal
    
    # Bouton pour calculer le portefeuille
    if st.button("📊 Calculate Portfolio Analytics", type="primary"):
        if abs(total_weight - 100.0) < 0.01:
            
            # ========== PRÉPARATION DES DONNÉES ==========
            
            # Créer un DataFrame avec tous les prix de clôture alignés
            portfolio_df = pd.DataFrame()
            
            for ticker in selected_displays:
                df = data_filtered[ticker]
                portfolio_df[ticker] = df['close']
            
            # Supprimer les NaN (pour gérer les dates manquantes)
            portfolio_df = portfolio_df.dropna()
            
            if len(portfolio_df) < 2:
                st.error("❌ Not enough data to calculate portfolio analytics")
            else:
                # ========== CALCULS DU PORTEFEUILLE ==========
                
                # Rendements quotidiens
                returns_df = portfolio_df.pct_change().dropna()
                
                # Poids du portefeuille (normalisés)
                weights_array = np.array([weights[ticker] / 100 for ticker in selected_displays])
                
                # Rendements du portefeuille
                portfolio_returns = (returns_df * weights_array).sum(axis=1)
                
                # Valeur du portefeuille au fil du temps
                portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()
                portfolio_value = pd.concat([pd.Series([initial_capital], index=[portfolio_df.index[0]]), portfolio_value])
                
                # ========== STATISTIQUES DE PERFORMANCE ==========
                
                total_return = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
                total_return_pct = total_return * 100
                
                # Annualiser les rendements (environ 252 jours de trading par an, mais ici H4 donc ~1560 périodes/an)
                # Pour H4: 6 périodes par jour * 252 jours = 1512 périodes/an
                periods_per_year = 1512
                n_periods = len(portfolio_returns)
                years = n_periods / periods_per_year
                
                cagr = (portfolio_value.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else 0
                
                # Volatilité annualisée
                volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
                
                # Sharpe Ratio
                excess_returns = portfolio_returns - risk_free_rate / periods_per_year
                sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
                
                # Sortino Ratio (seulement la volatilité des pertes)
                downside_returns = portfolio_returns[portfolio_returns < 0]
                downside_std = downside_returns.std() * np.sqrt(periods_per_year)
                sortino_ratio = (portfolio_returns.mean() * periods_per_year - risk_free_rate) / downside_std if downside_std > 0 else 0
                
                # Maximum Drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                max_drawdown_pct = max_drawdown * 100
                
                # Calmar Ratio
                calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # Win Rate
                win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100
                
                # Best & Worst periods
                best_return = portfolio_returns.max()
                worst_return = portfolio_returns.min()
                
                # VaR et CVaR (95%)
                var_95 = portfolio_returns.quantile(0.05)
                cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
                
                # ========== AFFICHAGE DES MÉTRIQUES ==========
                
                st.divider()
                st.subheader("📈 Portfolio Performance Metrics")
                
                # Ligne 1: Métriques principales
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Total Return",
                        f"{total_return_pct:+.2f}%",
                        f"${portfolio_value.iloc[-1] - initial_capital:+,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "CAGR",
                        f"{cagr * 100:.2f}%",
                        f"{years:.1f} years"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{sharpe_ratio:.2f}",
                        "Higher is better"
                    )
                
                with col4:
                    st.metric(
                        "Sortino Ratio",
                        f"{sortino_ratio:.2f}",
                        "Higher is better"
                    )
                
                with col5:
                    st.metric(
                        "Max Drawdown",
                        f"{max_drawdown_pct:.2f}%",
                        "Lower is better"
                    )
                
                # Ligne 2: Métriques secondaires
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Volatility (Ann.)", f"{volatility * 100:.2f}%")
                
                with col2:
                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
                
                with col3:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col4:
                    st.metric("VaR (95%)", f"{var_95 * 100:.2f}%")
                
                with col5:
                    st.metric("CVaR (95%)", f"{cvar_95 * 100:.2f}%")
                
                # Ligne 3: Best/Worst
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Period", f"{best_return * 100:+.2f}%")
                
                with col2:
                    st.metric("Worst Period", f"{worst_return * 100:.2f}%")
                
                with col3:
                    st.metric("Final Value", f"${portfolio_value.iloc[-1]:,.2f}")
                
                with col4:
                    profit_loss = portfolio_value.iloc[-1] - initial_capital
                    st.metric("Profit/Loss", f"${profit_loss:+,.2f}")
                
                # ========== GRAPHIQUES ==========
                
                st.divider()
                st.subheader("📊 Portfolio Visualizations")
                
                # Graphique 1: Évolution du portefeuille
                fig1, ax1 = plt.subplots(figsize=(16, 8), facecolor='#000000')
                ax1.set_facecolor('#0a0a0a')
                
                ax1.plot(portfolio_value.index, portfolio_value.values, 
                        linewidth=2.5, color='#00FF00', label='Portfolio Value', zorder=5)
                
                # Ligne de capital initial
                ax1.axhline(y=initial_capital, color='#FF8C00', linestyle='--', 
                           linewidth=1.5, alpha=0.5, label=f'Initial Capital (${initial_capital:,.0f})')
                
                ax1.set_title('Portfolio Value Evolution', 
                             fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
                ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color='#FF8C00', fontweight='bold')
                ax1.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
                ax1.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
                ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                          fontsize=10, labelcolor='#FF8C00')
                ax1.tick_params(colors='#FFFFFF', labelsize=10)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Format des valeurs en K ou M
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
                
                fig1.tight_layout()
                st.pyplot(fig1)
                
                # Graphique 2: Drawdown
                fig2, ax2 = plt.subplots(figsize=(16, 6), facecolor='#000000')
                ax2.set_facecolor('#0a0a0a')
                
                ax2.fill_between(drawdown.index, drawdown.values * 100, 0, 
                                color='#FF4500', alpha=0.6, label='Drawdown')
                ax2.plot(drawdown.index, drawdown.values * 100, 
                        color='#FF0000', linewidth=1.5, alpha=0.8)
                
                ax2.set_title('Portfolio Drawdown', 
                             fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
                ax2.set_ylabel('Drawdown (%)', fontsize=12, color='#FF8C00', fontweight='bold')
                ax2.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
                ax2.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
                ax2.legend(loc='lower left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                          fontsize=10, labelcolor='#FF8C00')
                ax2.tick_params(colors='#FFFFFF', labelsize=10)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                fig2.tight_layout()
                st.pyplot(fig2)
                
                # Graphique 3: Distribution des rendements
                col1, col2 = st.columns(2)
                
                with col1:
                    fig3, ax3 = plt.subplots(figsize=(8, 6), facecolor='#000000')
                    ax3.set_facecolor('#0a0a0a')
                    
                    ax3.hist(portfolio_returns * 100, bins=50, color='#4169E1', 
                            alpha=0.7, edgecolor='#00D9FF', linewidth=1.2)
                    ax3.axvline(x=0, color='#FF8C00', linestyle='--', linewidth=2, alpha=0.7)
                    
                    ax3.set_title('Returns Distribution', 
                                 fontsize=16, fontweight='bold', color='#FF8C00', pad=15)
                    ax3.set_xlabel('Returns (%)', fontsize=11, color='#FF8C00', fontweight='bold')
                    ax3.set_ylabel('Frequency', fontsize=11, color='#FF8C00', fontweight='bold')
                    ax3.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
                    ax3.tick_params(colors='#FFFFFF', labelsize=9)
                    
                    fig3.tight_layout()
                    st.pyplot(fig3)
                
                with col2:
                    fig4, ax4 = plt.subplots(figsize=(8, 6), facecolor='#000000')
                    ax4.set_facecolor('#0a0a0a')
                    
                    # Pie chart des poids
                    colors_pie = ['#00D9FF', '#FF1493', '#00FF00', '#FFD700', '#FF4500', 
                                 '#9370DB', '#FF69B4', '#00CED1', '#FFA500', '#7FFF00']
                    
                    wedges, texts, autotexts = ax4.pie(
                        [weights[ticker] for ticker in selected_displays],
                        labels=selected_displays,
                        autopct='%1.1f%%',
                        colors=colors_pie[:len(selected_displays)],
                        startangle=90,
                        textprops={'fontsize': 10, 'color': '#FFFFFF', 'fontweight': 'bold'}
                    )
                    
                    ax4.set_title('Portfolio Allocation', 
                                 fontsize=16, fontweight='bold', color='#FF8C00', pad=15)
                    
                    fig4.tight_layout()
                    st.pyplot(fig4)
                
                # ========== MATRICE DE CORRÉLATION ==========
                
                st.divider()
                st.subheader("🔗 Correlation Matrix")
                
                correlation_matrix = returns_df.corr()
                
                fig5, ax5 = plt.subplots(figsize=(10, 8), facecolor='#000000')
                ax5.set_facecolor('#0a0a0a')
                
                # Heatmap
                im = ax5.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto', 
                               vmin=-1, vmax=1, interpolation='nearest')
                
                # Configurer les ticks
                ax5.set_xticks(np.arange(len(selected_displays)))
                ax5.set_yticks(np.arange(len(selected_displays)))
                ax5.set_xticklabels(selected_displays, rotation=45, ha='right', color='#FFFFFF')
                ax5.set_yticklabels(selected_displays, color='#FFFFFF')
                
                # Ajouter les valeurs dans les cellules
                for i in range(len(selected_displays)):
                    for j in range(len(selected_displays)):
                        text = ax5.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                       ha='center', va='center', color='#000000', 
                                       fontweight='bold', fontsize=10)
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax5)
                cbar.set_label('Correlation', rotation=270, labelpad=20, 
                              color='#FF8C00', fontweight='bold')
                cbar.ax.tick_params(colors='#FFFFFF')
                
                ax5.set_title('Asset Correlation Matrix', 
                             fontsize=16, fontweight='bold', color='#FF8C00', pad=15)
                
                fig5.tight_layout()
                st.pyplot(fig5)
                
                # ========== TABLEAU RÉCAPITULATIF ==========
                
                st.divider()
                st.subheader("📋 Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    stats_data_1 = {
                        'Metric': [
                            'Initial Capital',
                            'Final Value',
                            'Total Return ($)',
                            'Total Return (%)',
                            'CAGR',
                            'Annualized Volatility',
                            'Sharpe Ratio',
                            'Sortino Ratio',
                            'Calmar Ratio'
                        ],
                        'Value': [
                            f'${initial_capital:,.2f}',
                            f'${portfolio_value.iloc[-1]:,.2f}',
                            f'${portfolio_value.iloc[-1] - initial_capital:+,.2f}',
                            f'{total_return_pct:+.2f}%',
                            f'{cagr * 100:.2f}%',
                            f'{volatility * 100:.2f}%',
                            f'{sharpe_ratio:.3f}',
                            f'{sortino_ratio:.3f}',
                            f'{calmar_ratio:.3f}'
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data_1), use_container_width=True, hide_index=True)
                
                with col2:
                    stats_data_2 = {
                        'Metric': [
                            'Maximum Drawdown',
                            'Win Rate',
                            'Best Period Return',
                            'Worst Period Return',
                            'VaR (95%)',
                            'CVaR (95%)',
                            'Number of Periods',
                            'Time Horizon (Years)',
                            'Risk-Free Rate'
                        ],
                        'Value': [
                            f'{max_drawdown_pct:.2f}%',
                            f'{win_rate:.2f}%',
                            f'{best_return * 100:+.2f}%',
                            f'{worst_return * 100:.2f}%',
                            f'{var_95 * 100:.2f}%',
                            f'{cvar_95 * 100:.2f}%',
                            f'{n_periods:,}',
                            f'{years:.2f}',
                            f'{risk_free_rate * 100:.2f}%'
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data_2), use_container_width=True, hide_index=True)
                
                # ========== STATISTIQUES PAR ACTIF ==========
                
                st.divider()
                st.subheader("📊 Individual Asset Statistics")
                
                asset_stats = []
                for ticker in selected_displays:
                    ticker_returns = returns_df[ticker]
                    ticker_weight = weights[ticker]
                    
                    asset_stats.append({
                        'Ticker': ticker,
                        'Weight': f'{ticker_weight:.1f}%',
                        'Mean Return': f'{ticker_returns.mean() * 100:.3f}%',
                        'Volatility': f'{ticker_returns.std() * 100:.2f}%',
                        'Sharpe': f'{(ticker_returns.mean() / ticker_returns.std() * np.sqrt(periods_per_year)):.2f}' if ticker_returns.std() > 0 else 'N/A',
                        'Best': f'{ticker_returns.max() * 100:+.2f}%',
                        'Worst': f'{ticker_returns.min() * 100:.2f}%',
                        'Win Rate': f'{(ticker_returns > 0).sum() / len(ticker_returns) * 100:.1f}%'
                    })
                
                asset_stats_df = pd.DataFrame(asset_stats)
                st.dataframe(asset_stats_df, use_container_width=True, hide_index=True)
        
        else:
            st.error("⚠️ Please adjust weights to total 100% before calculating portfolio analytics")

# ============================================================================
# AFFICHAGE MODE MOMENTUM STRATEGY
# ============================================================================

elif chart_mode == "🚀 Momentum Strategy":
    
    st.header("🚀 MOMENTUM TRADING STRATEGY")
    
    # ========== PARAMÈTRES DE LA STRATÉGIE (PAGE PRINCIPALE) ==========
    
    st.subheader("⚙️ Strategy Configuration")
    
    # Organisation en colonnes pour les paramètres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💼 Portfolio Allocation")
        weight_ticker1 = st.slider(
            f"% {selected_displays[0]} (Buy & Hold):",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            key="weight_main"
        )
        weight_ticker2 = 100 - weight_ticker1
        st.info(f"**{selected_displays[1]} (RSI Trading):** {weight_ticker2}%")
        
        st.markdown("#### 💰 Capital")
        initial_capital_momentum = st.number_input(
            "Initial Capital ($):",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000,
            key="capital_main"
        )
    
    with col2:
        st.markdown("#### 📊 RSI Parameters")
        rsi_buy = st.slider(
            "RSI Buy Level:",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Buy when RSI falls below this level (oversold)",
            key="rsi_buy_main"
        )
        
        rsi_sell = st.slider(
            "RSI Sell Level:",
            min_value=60,
            max_value=90,
            value=80,
            step=5,
            help="Sell when RSI rises above this level (overbought)",
            key="rsi_sell_main"
        )
        
        rsi_period = st.slider(
            "RSI Period:",
            min_value=7,
            max_value=28,
            value=14,
            step=1,
            help="Number of periods for RSI calculation",
            key="rsi_period_main"
        )
    
    with col3:
        st.markdown("#### 📈 Strategy Summary")
        st.metric("Primary Asset", selected_displays[0], f"{weight_ticker1}% - Hold")
        st.metric("Trading Asset", selected_displays[1], f"{weight_ticker2}% - RSI")
        st.metric("Buy Signal", f"RSI ≤ {rsi_buy}", "Oversold")
        st.metric("Sell Signal", f"RSI ≥ {rsi_sell}", "Overbought")
        
    st.divider()
    
    # Fonction pour calculer le RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Récupérer les données des deux tickers
    ticker1_name = selected_displays[0]
    ticker2_name = selected_displays[1]
    
    df1 = data_filtered[ticker1_name].copy()
    df2 = data_filtered[ticker2_name].copy()
    
    # Fusionner les données sur les dates communes
    merged = pd.merge(
        df1[['close']].reset_index(),
        df2[['close']].reset_index(),
        on='date',
        suffixes=(f'_{ticker1_name}', f'_{ticker2_name}')
    )
    
    merged = merged.set_index('date')
    
    if len(merged) < rsi_period + 2:
        st.error(f"❌ Not enough data to calculate RSI with period {rsi_period}")
        st.stop()
    
    # Calculer les RSI
    merged[f'rsi_{ticker1_name}'] = calculate_rsi(merged[f'close_{ticker1_name}'], rsi_period)
    merged[f'rsi_{ticker2_name}'] = calculate_rsi(merged[f'close_{ticker2_name}'], rsi_period)
    
    # ========== SIMULATION DE LA STRATÉGIE ==========
    
    # Initialisation
    capital_ticker1 = initial_capital_momentum * (weight_ticker1 / 100)
    capital_ticker2_cash = initial_capital_momentum * (weight_ticker2 / 100)
    
    # Acheter ticker1 immédiatement et hold
    prix_achat_ticker1 = merged[f'close_{ticker1_name}'].iloc[0]
    nb_actions_ticker1 = capital_ticker1 / prix_achat_ticker1
    
    # Variables pour ticker2 (stratégie RSI)
    nb_actions_ticker2 = 0
    ticker2_position = False
    prix_achat_ticker2 = 0
    
    # Journal de trading
    journal = []
    
    # Listes pour tracer l'évolution du portefeuille
    valeur_portefeuille = []
    valeur_buy_hold = []
    
    # Simulation
    for i in range(len(merged)):
        date = merged.index[i]
        prix_ticker1 = merged[f'close_{ticker1_name}'].iloc[i]
        prix_ticker2 = merged[f'close_{ticker2_name}'].iloc[i]
        rsi_ticker2 = merged[f'rsi_{ticker2_name}'].iloc[i]
        
        # Valeur actuelle du ticker1 (buy & hold)
        valeur_ticker1 = nb_actions_ticker1 * prix_ticker1
        
        # Gestion de la position ticker2 avec stratégie RSI
        if pd.notna(rsi_ticker2):
            # Signal d'achat: RSI <= rsi_buy et pas de position
            if rsi_ticker2 <= rsi_buy and not ticker2_position:
                nb_actions_ticker2 = capital_ticker2_cash / prix_ticker2
                prix_achat_ticker2 = prix_ticker2
                ticker2_position = True
                journal.append({
                    'Date': date,
                    'Action': f'BUY {ticker2_name}',
                    'Price': prix_ticker2,
                    'Quantity': nb_actions_ticker2,
                    'RSI': rsi_ticker2,
                    'Capital': capital_ticker2_cash
                })
            
            # Signal de vente: RSI >= rsi_sell et position ouverte
            elif rsi_ticker2 >= rsi_sell and ticker2_position:
                valeur_vente = nb_actions_ticker2 * prix_ticker2
                profit = valeur_vente - capital_ticker2_cash
                profit_pct = (profit / capital_ticker2_cash) * 100
                capital_ticker2_cash = valeur_vente
                journal.append({
                    'Date': date,
                    'Action': f'SELL {ticker2_name}',
                    'Price': prix_ticker2,
                    'Quantity': nb_actions_ticker2,
                    'RSI': rsi_ticker2,
                    'Buy Price': prix_achat_ticker2,
                    'Profit': profit,
                    'Profit %': profit_pct,
                    'Capital After': capital_ticker2_cash
                })
                nb_actions_ticker2 = 0
                ticker2_position = False
        
        # Valeur actuelle de la position ticker2
        if ticker2_position:
            valeur_ticker2 = nb_actions_ticker2 * prix_ticker2
        else:
            valeur_ticker2 = capital_ticker2_cash
        
        # Valeur totale du portefeuille avec stratégie
        valeur_totale = valeur_ticker1 + valeur_ticker2
        valeur_portefeuille.append(valeur_totale)
        
        # Valeur du portefeuille buy & hold classique
        valeur_ticker1_bh = (initial_capital_momentum * weight_ticker1 / 100 / merged[f'close_{ticker1_name}'].iloc[0]) * prix_ticker1
        valeur_ticker2_bh = (initial_capital_momentum * weight_ticker2 / 100 / merged[f'close_{ticker2_name}'].iloc[0]) * prix_ticker2
        valeur_buy_hold.append(valeur_ticker1_bh + valeur_ticker2_bh)
    
    # Ajouter les résultats au dataframe
    merged['portfolio_strategy'] = valeur_portefeuille
    merged['portfolio_buy_hold'] = valeur_buy_hold
    merged['pct_strategy'] = (merged['portfolio_strategy'] / initial_capital_momentum) * 100
    merged['pct_buy_hold'] = (merged['portfolio_buy_hold'] / initial_capital_momentum) * 100
    merged[f'pct_{ticker1_name}'] = (merged[f'close_{ticker1_name}'] / merged[f'close_{ticker1_name}'].iloc[0]) * 100
    merged[f'pct_{ticker2_name}'] = (merged[f'close_{ticker2_name}'] / merged[f'close_{ticker2_name}'].iloc[0]) * 100
    
    # ========== AFFICHAGE DES MÉTRIQUES ==========
    
    st.subheader("📊 Strategy Performance")
    
    valeur_finale_bh = merged['portfolio_buy_hold'].iloc[-1]
    perf_bh = ((valeur_finale_bh / initial_capital_momentum) - 1) * 100
    
    valeur_finale_strat = merged['portfolio_strategy'].iloc[-1]
    perf_strat = ((valeur_finale_strat / initial_capital_momentum) - 1) * 100
    
    diff_perf = perf_strat - perf_bh
    diff_valeur = valeur_finale_strat - valeur_finale_bh
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Initial Capital",
            f"${initial_capital_momentum:,.2f}"
        )
    
    with col2:
        st.metric(
            "Buy & Hold",
            f"${valeur_finale_bh:,.2f}",
            f"{perf_bh:+.2f}%"
        )
    
    with col3:
        st.metric(
            "Strategy",
            f"${valeur_finale_strat:,.2f}",
            f"{perf_strat:+.2f}%"
        )
    
    with col4:
        st.metric(
            "Outperformance",
            f"{diff_perf:+.2f}%",
            f"${diff_valeur:+,.2f}"
        )
    
    with col5:
        st.metric(
            "Total Trades",
            f"{len(journal)}"
        )
    
    st.markdown(f"**Strategy:** {ticker1_name} {weight_ticker1}% Hold + {ticker2_name} {weight_ticker2}% RSI({rsi_buy}/{rsi_sell})")
    
    st.divider()
    
    # ========== GRAPHIQUES ==========
    
    st.subheader("📈 Performance Charts")
    
    # Créer une figure avec 6 sous-graphiques
    fig = plt.figure(figsize=(16, 18), facecolor='#000000')
    gs = fig.add_gridspec(6, 1, height_ratios=[2, 1, 2, 1, 2, 2], hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0])
    ax1_rsi = fig.add_subplot(gs[1], sharex=ax1)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2_rsi = fig.add_subplot(gs[3], sharex=ax1)
    ax3 = fig.add_subplot(gs[4], sharex=ax1)
    ax4 = fig.add_subplot(gs[5], sharex=ax1)
    
    # --- Graphique 1: Prix ticker1 ---
    ax1.set_facecolor('#0a0a0a')
    ax1.plot(merged.index, merged[f'close_{ticker1_name}'], label=ticker1_name, color='#00D9FF', linewidth=1.5)
    ax1.set_ylabel(f'Price {ticker1_name} ($)', fontsize=11, color='#00D9FF', fontweight='bold')
    ax1.set_title(f'{ticker1_name} - Price & RSI (Buy & Hold)', fontsize=13, fontweight='bold', color='#FF8C00')
    ax1.grid(True, alpha=0.2, color='#333333')
    ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', labelcolor='#FF8C00')
    ax1.tick_params(axis='x', labelbottom=False, colors='#FFFFFF')
    ax1.tick_params(axis='y', colors='#FFFFFF')
    
    # --- Graphique RSI ticker1 ---
    ax1_rsi.set_facecolor('#0a0a0a')
    ax1_rsi.plot(merged.index, merged[f'rsi_{ticker1_name}'], color='#00D9FF', linewidth=1.5)
    ax1_rsi.axhline(y=70, color='#FF0000', linestyle='--', linewidth=1, alpha=0.5)
    ax1_rsi.axhline(y=30, color='#00FF00', linestyle='--', linewidth=1, alpha=0.5)
    ax1_rsi.axhline(y=50, color='#808080', linestyle=':', linewidth=1, alpha=0.3)
    ax1_rsi.fill_between(merged.index, 70, merged[f'rsi_{ticker1_name}'], 
                          where=(merged[f'rsi_{ticker1_name}'] >= 70),
                          color='#FF0000', alpha=0.2)
    ax1_rsi.fill_between(merged.index, 30, merged[f'rsi_{ticker1_name}'], 
                          where=(merged[f'rsi_{ticker1_name}'] <= 30),
                          color='#00FF00', alpha=0.2)
    ax1_rsi.set_ylabel('RSI', fontsize=10, color='#FF8C00', fontweight='bold')
    ax1_rsi.set_ylim(0, 100)
    ax1_rsi.grid(True, alpha=0.2, color='#333333')
    ax1_rsi.tick_params(axis='x', labelbottom=False, colors='#FFFFFF')
    ax1_rsi.tick_params(axis='y', colors='#FFFFFF')
    
    # --- Graphique 2: Prix ticker2 avec signaux ---
    ax2.set_facecolor('#0a0a0a')
    ax2.plot(merged.index, merged[f'close_{ticker2_name}'], label=ticker2_name, color='#FF1493', linewidth=1.5)
    
    # Ajouter les signaux d'achat et de vente
    for trade in journal:
        if 'BUY' in trade['Action']:
            ax2.scatter(trade['Date'], trade['Price'], color='#00FF00', marker='^', s=100, 
                       label='Buy Signal' if trade == journal[0] else '', zorder=5)
        elif 'SELL' in trade['Action']:
            ax2.scatter(trade['Date'], trade['Price'], color='#FF0000', marker='v', s=100, 
                       label='Sell Signal' if any('BUY' in t['Action'] for t in journal) else '', zorder=5)
    
    ax2.set_ylabel(f'Price {ticker2_name} ($)', fontsize=11, color='#FF1493', fontweight='bold')
    ax2.set_title(f'{ticker2_name} - Price & RSI (Strategy: Buy RSI≤{rsi_buy}, Sell RSI≥{rsi_sell})', 
                 fontsize=13, fontweight='bold', color='#FF8C00')
    ax2.grid(True, alpha=0.2, color='#333333')
    ax2.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', labelcolor='#FF8C00')
    ax2.tick_params(axis='x', labelbottom=False, colors='#FFFFFF')
    ax2.tick_params(axis='y', colors='#FFFFFF')
    
    # --- Graphique RSI ticker2 avec niveaux de trading ---
    ax2_rsi.set_facecolor('#0a0a0a')
    ax2_rsi.plot(merged.index, merged[f'rsi_{ticker2_name}'], color='#FF1493', linewidth=1.5)
    ax2_rsi.axhline(y=rsi_sell, color='#FF0000', linestyle='--', linewidth=2, alpha=0.7, label=f'Sell ({rsi_sell})')
    ax2_rsi.axhline(y=rsi_buy, color='#00FF00', linestyle='--', linewidth=2, alpha=0.7, label=f'Buy ({rsi_buy})')
    ax2_rsi.axhline(y=50, color='#808080', linestyle=':', linewidth=1, alpha=0.3)
    ax2_rsi.fill_between(merged.index, rsi_sell, 100, color='#FF0000', alpha=0.1)
    ax2_rsi.fill_between(merged.index, 0, rsi_buy, color='#00FF00', alpha=0.1)
    ax2_rsi.set_ylabel('RSI', fontsize=10, color='#FF8C00', fontweight='bold')
    ax2_rsi.set_ylim(0, 100)
    ax2_rsi.grid(True, alpha=0.2, color='#333333')
    ax2_rsi.legend(loc='upper left', fontsize=8, facecolor='#1a1a1a', edgecolor='#FF8C00', labelcolor='#FF8C00')
    ax2_rsi.tick_params(axis='x', labelbottom=False, colors='#FFFFFF')
    ax2_rsi.tick_params(axis='y', colors='#FFFFFF')
    
    # --- Graphique 3: Evolution en pourcentage ---
    ax3.set_facecolor('#0a0a0a')
    ax3.plot(merged.index, merged[f'pct_{ticker1_name}'], label=f'{ticker1_name} Only', 
            color='#00D9FF', linewidth=1.5, alpha=0.7)
    ax3.plot(merged.index, merged[f'pct_{ticker2_name}'], label=f'{ticker2_name} Only', 
            color='#FF1493', linewidth=1.5, alpha=0.7)
    ax3.plot(merged.index, merged['pct_buy_hold'], 
            label=f'Buy&Hold ({weight_ticker1}% {ticker1_name} / {weight_ticker2}% {ticker2_name})', 
            color='#FFA500', linewidth=2)
    
    ax3.set_ylabel('Evolution (%)', fontsize=12, color='#FF8C00', fontweight='bold')
    ax3.set_title('Individual Assets & Buy & Hold Performance', fontsize=14, fontweight='bold', color='#FF8C00')
    ax3.axhline(y=100, color='#808080', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(True, alpha=0.2, color='#333333')
    ax3.legend(loc='best', facecolor='#1a1a1a', edgecolor='#FF8C00', labelcolor='#FF8C00')
    ax3.tick_params(axis='x', labelbottom=False, colors='#FFFFFF')
    ax3.tick_params(axis='y', colors='#FFFFFF')
    
    # --- Graphique 4: Comparaison Stratégie vs Buy & Hold ---
    ax4.set_facecolor('#0a0a0a')
    ax4.plot(merged.index, merged['pct_buy_hold'], 
            label=f'Buy&Hold ({weight_ticker1}% {ticker1_name} / {weight_ticker2}% {ticker2_name})', 
            color='#FFA500', linewidth=2, alpha=0.7)
    ax4.plot(merged.index, merged['pct_strategy'], 
            label=f'Strategy ({ticker1_name} hold + {ticker2_name} RSI {rsi_buy}/{rsi_sell})', 
            color='#9370DB', linewidth=2)
    ax4.fill_between(merged.index, merged['pct_buy_hold'], merged['pct_strategy'], 
                     where=(merged['pct_strategy'] >= merged['pct_buy_hold']), 
                     color='#00FF00', alpha=0.2, label='Strategy Outperforms')
    ax4.fill_between(merged.index, merged['pct_buy_hold'], merged['pct_strategy'], 
                     where=(merged['pct_strategy'] < merged['pct_buy_hold']), 
                     color='#FF0000', alpha=0.2, label='Buy&Hold Outperforms')
    
    ax4.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
    ax4.set_ylabel('Evolution (%)', fontsize=12, color='#FF8C00', fontweight='bold')
    ax4.set_title('Strategy vs Buy & Hold Comparison', fontsize=14, fontweight='bold', color='#FF8C00')
    ax4.axhline(y=100, color='#808080', linestyle='--', linewidth=1, alpha=0.5)
    ax4.grid(True, alpha=0.2, color='#333333')
    ax4.legend(loc='best', facecolor='#1a1a1a', edgecolor='#FF8C00', labelcolor='#FF8C00')
    ax4.tick_params(colors='#FFFFFF')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', color='#FFFFFF')
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # ========== JOURNAL DE TRADING ==========
    
    st.divider()
    st.subheader(f"📋 Trading Journal - {ticker2_name} (RSI Strategy)")
    
    if journal:
        journal_df = pd.DataFrame(journal)
        
        # Formater les colonnes
        if 'Price' in journal_df.columns:
            journal_df['Price'] = journal_df['Price'].apply(lambda x: f"${x:.2f}")
        if 'Quantity' in journal_df.columns:
            journal_df['Quantity'] = journal_df['Quantity'].apply(lambda x: f"{x:.4f}")
        if 'RSI' in journal_df.columns:
            journal_df['RSI'] = journal_df['RSI'].apply(lambda x: f"{x:.2f}")
        if 'Capital' in journal_df.columns:
            journal_df['Capital'] = journal_df['Capital'].apply(lambda x: f"${x:.2f}")
        if 'Buy Price' in journal_df.columns:
            journal_df['Buy Price'] = journal_df['Buy Price'].apply(lambda x: f"${x:.2f}")
        if 'Profit' in journal_df.columns:
            journal_df['Profit'] = journal_df['Profit'].apply(lambda x: f"${x:+.2f}")
        if 'Profit %' in journal_df.columns:
            journal_df['Profit %'] = journal_df['Profit %'].apply(lambda x: f"{x:+.2f}%")
        if 'Capital After' in journal_df.columns:
            journal_df['Capital After'] = journal_df['Capital After'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(journal_df, use_container_width=True, hide_index=True)
        
        # Statistiques des trades
        trades_sell = [t for t in journal if 'SELL' in t['Action']]
        if trades_sell:
            st.divider()
            st.subheader("📊 Trade Statistics")
            
            profits = [t['Profit %'] for t in trades_sell]
            nb_winning = sum(1 for p in profits if p > 0)
            nb_losing = sum(1 for p in profits if p < 0)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Completed Trades", f"{len(trades_sell)}")
            
            with col2:
                st.metric("Winning Trades", f"{nb_winning}", f"{nb_winning/len(trades_sell)*100:.1f}%")
            
            with col3:
                st.metric("Losing Trades", f"{nb_losing}", f"{nb_losing/len(trades_sell)*100:.1f}%")
            
            with col4:
                st.metric("Avg Profit/Trade", f"{np.mean(profits):+.2f}%")
            
            with col5:
                st.metric("Best Trade", f"{max(profits):+.2f}%")
    else:
        st.info("No trades executed during this period with current RSI parameters")

# Footer
st.markdown("---")
st.markdown(f"**Data Source:** Supabase | **Mode:** {chart_mode} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
