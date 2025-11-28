import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from supabase import create_client, Client
from datetime import datetime
import matplotlib.dates as mdates

# ============================================================================
# CONFIG STREAMLIT - STYLE BLOOMBERG
# ============================================================================
st.set_page_config(page_title="Morgan Stanley H4", page_icon="📊", layout="wide")

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
    
    /* Dataframe */
    .stDataFrame {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 MORGAN STANLEY (MS) - H4 CHART")

# ============================================================================
# PARAMETRES SUPABASE
# ============================================================================

SUPABASE_URL = "https://gbrefcefeavmqupulzyw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdicmVmY2VmZWF2bXF1cHVsenl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0OTA2NjksImV4cCI6MjA3OTA2NjY2OX0.WsA-3so0J52hAyZTIddVT0qqLuvcxjHYTZ4XkZ5mMio"

TABLE_NAME = "morgan_stanley_h4_data"

# ============================================================================
# RECUPERATION DES DONNEES
# ============================================================================

@st.cache_data(ttl=3600)
def load_data():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        with st.spinner("📥 Loading data from database..."):
            response = supabase.table(TABLE_NAME).select("*").order("date", desc=False).limit(10000).execute()
        
        if not response.data:
            st.error("❌ No data found in table!")
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
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Charger les données
df = load_data()

if df is not None:
    # ============================================================================
    # SIDEBAR - PARAMETRES
    # ============================================================================
    
    st.sidebar.header("⚙️ CHART SETTINGS")
    
    # Choix du type de graphique
    chart_type = st.sidebar.radio(
        "Chart Type:",
        ["📈 Line Chart", "🕯️ Candlestick"],
        index=1
    )
    
    # Filtrage par nombre de bougies
    max_candles = len(df)
    num_candles = st.sidebar.slider(
        "Number of candles:",
        min_value=50,
        max_value=min(max_candles, 5000),
        value=min(500, max_candles),
        step=50
    )
    
    # Prendre les dernières bougies
    df_filtered = df.tail(num_candles)
    
    # ============================================================================
    # AFFICHAGE DES STATS - STYLE BLOOMBERG
    # ============================================================================
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    last_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
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
        st.metric("CANDLES", f"{len(df):,}")
    
    st.markdown(f"**PERIOD:** {df_filtered.index[0].strftime('%Y-%m-%d')} to {df_filtered.index[-1].strftime('%Y-%m-%d')}")
    
    st.divider()
    
    # ============================================================================
    # GRAPHIQUE STYLE BLOOMBERG
    # ============================================================================
    
    if chart_type == "📈 Line Chart":
        # ========== COURBE STYLE BLOOMBERG ==========
        st.subheader(f"📈 LINE CHART - LAST {num_candles} CANDLES")
        
        # Calculer les moyennes mobiles
        df_filtered['SMA50'] = df_filtered['close'].rolling(window=50).mean()
        df_filtered['SMA100'] = df_filtered['close'].rolling(window=100).mean()
        df_filtered['SMA200'] = df_filtered['close'].rolling(window=200).mean()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        facecolor='#000000')
        
        # Graphique principal - Prix
        ax1.set_facecolor('#0a0a0a')
        
        # Prix de clôture
        ax1.plot(df_filtered.index, df_filtered['close'], 
                linewidth=2, color='#00D9FF', label='Close', zorder=5)
        
        # Moyennes mobiles
        if len(df_filtered) >= 50:
            ax1.plot(df_filtered.index, df_filtered['SMA50'], 
                    linewidth=1.5, color='#FF1493', label='SMA 50', alpha=0.8)
        if len(df_filtered) >= 100:
            ax1.plot(df_filtered.index, df_filtered['SMA100'], 
                    linewidth=1.5, color='#00FF00', label='SMA 100', alpha=0.8)
        if len(df_filtered) >= 200:
            ax1.plot(df_filtered.index, df_filtered['SMA200'], 
                    linewidth=1.5, color='#FFD700', label='SMA 200', alpha=0.8)
        
        ax1.set_title('MORGAN STANLEY (MS) - H4 Chart', 
                     fontsize=18, fontweight='bold', color='#FF8C00', pad=20)
        ax1.set_ylabel('Price ($)', fontsize=12, color='#FF8C00', fontweight='bold')
        ax1.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#FF8C00', 
                  fontsize=10, labelcolor='#FF8C00')
        ax1.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Graphique du volume
        ax2.set_facecolor('#0a0a0a')
        colors = ['#00FF00' if df_filtered['close'].iloc[i] >= df_filtered['open'].iloc[i] 
                 else '#FF0000' for i in range(len(df_filtered))]
        ax2.bar(df_filtered.index, df_filtered['volume'], color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume', fontsize=12, color='#FF8C00', fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, color='#FF8C00', fontweight='bold')
        ax2.grid(True, alpha=0.2, color='#333333', linestyle='-', linewidth=0.5)
        ax2.tick_params(colors='#FFFFFF', labelsize=10)
        
        # Format des dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        st.pyplot(fig)
    
    else:
        # ========== BOUGIES STYLE BLOOMBERG ==========
        st.subheader(f"🕯️ CANDLESTICK CHART - LAST {num_candles} CANDLES")
        
        # Style Bloomberg pour les bougies
        mc = mpf.make_marketcolors(
            up='#00FF00',        # Vert fluo
            down='#FF0000',      # Rouge vif
            edge='inherit',
            wick='inherit',
            volume={
                'up': '#00FF00',
                'down': '#FF0000'
            },
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
        
        # Ajouter les moyennes mobiles
        apds = []
        if len(df_filtered) >= 50:
            apds.append(mpf.make_addplot(df_filtered['close'].rolling(50).mean(), 
                                         color='#FF1493', width=1.5, label='SMA 50'))
        if len(df_filtered) >= 100:
            apds.append(mpf.make_addplot(df_filtered['close'].rolling(100).mean(), 
                                         color='#00FF00', width=1.5, label='SMA 100'))
        if len(df_filtered) >= 200:
            apds.append(mpf.make_addplot(df_filtered['close'].rolling(200).mean(), 
                                         color='#FFD700', width=1.5, label='SMA 200'))
        
        # Créer le graphique
        fig, axes = mpf.plot(
            df_filtered,
            type='candle',
            style=s,
            title='MORGAN STANLEY (MS) - H4 Candlestick Chart',
            ylabel='Price ($)',
            volume=True,
            ylabel_lower='Volume',
            figsize=(16, 10),
            addplot=apds if apds else None,
            returnfig=True,
            warn_too_much_data=10000
        )
        
        # Personnaliser les couleurs des axes
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
    
    # ============================================================================
    # TABLEAU DE DONNEES - STYLE BLOOMBERG
    # ============================================================================
    
    st.divider()
    st.subheader("📋 RECENT DATA")
    
    # Préparer le dataframe pour l'affichage
    display_df = df_filtered.tail(20).sort_index(ascending=False).copy()
    display_df['Change'] = display_df['close'] - display_df['open']
    display_df['Change %'] = (display_df['Change'] / display_df['open'] * 100).round(2)
    
    # Sélectionner les colonnes
    display_df = display_df[['open', 'high', 'low', 'close', 'volume', 'Change', 'Change %']]
    
    # Formatter
    display_df['open'] = display_df['open'].apply(lambda x: f"${x:.2f}")
    display_df['high'] = display_df['high'].apply(lambda x: f"${x:.2f}")
    display_df['low'] = display_df['low'].apply(lambda x: f"${x:.2f}")
    display_df['close'] = display_df['close'].apply(lambda x: f"${x:.2f}")
    display_df['volume'] = display_df['volume'].apply(lambda x: f"{int(x):,}")
    display_df['Change'] = display_df['Change'].apply(lambda x: f"${x:.2f}")
    display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(display_df, width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Data Source:** Supabase | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.warning("⚠️ Unable to load data. Check your Supabase connection.")
