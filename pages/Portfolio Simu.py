import streamlit as st
import pandas as pd
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
        
        response = supabase.table(table_name).select("*").order("date", desc=False).limit(10000).execute()
        
        if not response.data:
            return None
        
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
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
    ["📊 Single Stock Analysis", "📈 Multi-Stock Comparison"],
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

else:
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

# Footer
st.markdown("---")
st.markdown(f"**Data Source:** Supabase | **Mode:** {chart_mode} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
