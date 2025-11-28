import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from supabase import create_client, Client

# ============================================================================
# CONFIG STREAMLIT
# ============================================================================
st.set_page_config(page_title="Morgan Stanley H4", page_icon="📊", layout="wide")

st.title("📊 Morgan Stanley (MS) - Données H4")

# ============================================================================
# PARAMETRES SUPABASE
# ============================================================================

SUPABASE_URL = "https://gbrefcefeavmqupulzyw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdicmVmY2VmZWF2bXF1cHVsenl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0OTA2NjksImV4cCI6MjA3OTA2NjY2OX0.WsA-3so0J52hAyZTIddVT0qqLuvcxjHYTZ4XkZ5mMio"

TABLE_NAME = "morgan_stanley_h4_data"

# ============================================================================
# RECUPERATION DES DONNEES
# ============================================================================

@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def load_data():
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        with st.spinner("📥 Chargement des données..."):
            response = supabase.table(TABLE_NAME).select("*").order("date", desc=False).limit(10000).execute()
        
        if not response.data:
            st.error("❌ Aucune donnée trouvée dans la table!")
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
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

# Charger les données
df = load_data()

if df is not None:
    # ============================================================================
    # SIDEBAR - PARAMETRES
    # ============================================================================
    
    st.sidebar.header("⚙️ Paramètres")
    
    # Choix du type de graphique
    chart_type = st.sidebar.radio(
        "Type de graphique:",
        ["📈 Courbe", "🕯️ Bougies japonaises"],
        index=1
    )
    
    # Filtrage par nombre de bougies
    max_candles = len(df)
    num_candles = st.sidebar.slider(
        "Nombre de bougies à afficher:",
        min_value=50,
        max_value=min(max_candles, 5000),
        value=min(500, max_candles),
        step=50
    )
    
    # Prendre les dernières bougies
    df_filtered = df.tail(num_candles)
    
    # ============================================================================
    # AFFICHAGE DES STATS
    # ============================================================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total bougies", f"{len(df):,}")
    with col2:
        st.metric("💰 Dernier prix", f"${df['close'].iloc[-1]:.2f}")
    with col3:
        prix_change = df['close'].iloc[-1] - df['close'].iloc[-2]
        pct_change = (prix_change / df['close'].iloc[-2]) * 100
        st.metric("📈 Variation", f"${prix_change:.2f}", f"{pct_change:.2f}%")
    with col4:
        st.metric("📅 Période", f"{df.index[0].date()} à {df.index[-1].date()}")
    
    st.divider()
    
    # ============================================================================
    # GRAPHIQUE
    # ============================================================================
    
    if chart_type == "📈 Courbe":
        # COURBE
        st.subheader(f"📈 Courbe de prix - {num_candles} dernières bougies")
        
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df_filtered.index, df_filtered['close'], linewidth=1.5, color='#2962FF')
        ax.set_title('Morgan Stanley (MS) - Prix de clôture H4', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Prix ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        st.pyplot(fig)
    
    else:
        # BOUGIES JAPONAISES
        st.subheader(f"🕯️ Bougies japonaises - {num_candles} dernières bougies")
        
        mc = mpf.make_marketcolors(
            up='#26a69a',
            down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='lightgray',
            facecolor='white',
            figcolor='white'
        )
        
        fig, axes = mpf.plot(
            df_filtered,
            type='candle',
            style=s,
            title='Morgan Stanley (MS) - Bougies H4',
            ylabel='Prix ($)',
            volume=True,
            ylabel_lower='Volume',
            figsize=(15, 8),
            returnfig=True
        )
        
        st.pyplot(fig)
    
    # ============================================================================
    # TABLEAU DE DONNEES
    # ============================================================================
    
    st.divider()
    st.subheader("📋 Aperçu des données")
    
    # Afficher les dernières lignes
    st.dataframe(
        df_filtered.tail(20).sort_index(ascending=False),
        width='stretch'
    )

else:
    st.warning("⚠️ Impossible de charger les données. Vérifie ta connexion Supabase.")
