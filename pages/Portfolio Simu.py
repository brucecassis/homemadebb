import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
import numpy as np

# =============================================
# CONFIGURATION SUPABASE
# =============================================
SUPABASE_URL = "https://gbrefcefeavmqupulzyw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdicmVmY2VmZWF2bXF1cHVsenl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM0OTA2NjksImV4cCI6MjA3OTA2NjY2OX0.WsA-3so0J52hAyZTIddVT0qqLuvcxjHYTZ4XkZ5mMio"

@st.cache_resource
def init_supabase():
    """Initialise la connexion Supabase"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Stock Charts",
    page_icon="📈",
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
        transition: none !important;
    }
    
    .main {
        transition: none !important;
        animation: none !important;
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
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

    /* Style pour les inputs */
    .stSelectbox, .stMultiSelect, .stNumberInput, .stDateInput {
        color: #FFAA00 !important;
    }
    
    input, select, textarea {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    /* Style pour les sliders */
    .stSlider > div > div > div {
        background-color: #FFAA00 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>📈 BLOOMBERG ENS® TERMINAL - STOCK CHARTS</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">PORTFOLIO</a>
    </div>
    <div>{datetime.now().strftime("%H:%M:%S")} UTC</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS UTILITAIRES
# =============================================

@st.cache_data(ttl=3600)
def get_available_stocks():
    """Récupère la liste des actions disponibles"""
    known_tables = [
        'aapl_h4_data', 'msft_h4_data', 'googl_h4_data', 'goog_h4_data', 'amzn_h4_data',
        'nvda_h4_data', 'meta_h4_data', 'tsla_h4_data', 'brk_b_h4_data', 'unh_h4_data',
        'jnj_h4_data', 'jpm_h4_data', 'v_h4_data', 'pg_h4_data', 'xom_h4_data',
        'hd_h4_data', 'cvx_h4_data', 'ma_h4_data', 'abbv_h4_data', 'pfe_h4_data',
        'avgo_h4_data', 'cost_h4_data', 'dis_h4_data', 'ko_h4_data', 'adbe_h4_data',
        'pep_h4_data', 'csco_h4_data', 'tmo_h4_data', 'nflx_h4_data', 'wmt_h4_data',
        'mcd_h4_data', 'abt_h4_data', 'crm_h4_data', 'lin_h4_data', 'dhp_h4_data',
        'acn_h4_data', 'nke_h4_data', 'txt_h4_data', 'orcl_h4_data', 'intc_h4_data',
        'vz_h4_data', 'cmcsa_h4_data', 'mrk_h4_data', 'amd_h4_data', 'qcom_h4_data',
        'ibm_h4_data', 'ba_h4_data', 'cat_h4_data', 'ge_h4_data', 'spg_h4_data'
    ]
    
    stocks = []
    for table in known_tables:
        if table.endswith('_h4_data'):
            ticker = table.replace('_h4_data', '').upper()
            stocks.append(ticker)
    
    return sorted(stocks)

def get_date_range_for_stock(ticker):
    """Récupère la plage de dates disponible pour une action"""
    try:
        table_name = f"{ticker.lower()}_h4_data"
        
        # Récupérer la première et dernière date
        response = supabase.table(table_name)\
            .select("date")\
            .order('date', desc=False)\
            .limit(1)\
            .execute()
        
        first_date = None
        if response.data and len(response.data) > 0:
            first_date = pd.to_datetime(response.data[0]['date']).date()
        
        response = supabase.table(table_name)\
            .select("date")\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        
        last_date = None
        if response.data and len(response.data) > 0:
            last_date = pd.to_datetime(response.data[0]['date']).date()
        
        return first_date, last_date
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération des dates: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_stock_data(ticker, start_date, end_date):
    """Récupère les données d'une action depuis Supabase avec diagnostic amélioré"""
    try:
        table_name = f"{ticker.lower()}_h4_data"
        
        # Convertir les dates en datetime si ce sont des objets date
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        # Format simplifié sans timezone - juste la date
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        st.info(f"🔍 Recherche de données pour {ticker} du {start_str} au {end_str}")
        
        # D'abord, vérifier combien de données existent dans la table
        count_response = supabase.table(table_name)\
            .select("date", count="exact")\
            .execute()
        
        total_count = count_response.count if hasattr(count_response, 'count') else 0
        st.info(f"📊 Total d'entrées dans {table_name}: {total_count}")
        
        # Récupérer un échantillon pour voir le format des dates
        sample_response = supabase.table(table_name)\
            .select("date")\
            .limit(5)\
            .execute()
        
        if sample_response.data:
            st.info(f"📅 Échantillon de dates dans la table:")
            for item in sample_response.data[:3]:
                st.write(f"  • {item['date']}")
        
        # Essayer plusieurs formats de requête
        response = None
        
        # Méthode 1: Avec timezone
        try:
            start_str_tz = f"{start_date.strftime('%Y-%m-%d')}T00:00:00+00:00"
            end_str_tz = f"{end_date.strftime('%Y-%m-%d')}T23:59:59+00:00"
            
            response = supabase.table(table_name)\
                .select("date, open, high, low, close, volume")\
                .gte('date', start_str_tz)\
                .lte('date', end_str_tz)\
                .order('date', desc=False)\
                .execute()
            
            if response.data and len(response.data) > 0:
                st.success(f"✅ Méthode 1 (avec TZ): {len(response.data)} entrées trouvées")
        except Exception as e:
            st.warning(f"⚠️ Méthode 1 échouée: {str(e)}")
        
        # Méthode 2: Sans timezone, juste la date
        if not response or not response.data:
            try:
                response = supabase.table(table_name)\
                    .select("date, open, high, low, close, volume")\
                    .gte('date', start_str)\
                    .lte('date', end_str)\
                    .order('date', desc=False)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    st.success(f"✅ Méthode 2 (sans TZ): {len(response.data)} entrées trouvées")
            except Exception as e:
                st.warning(f"⚠️ Méthode 2 échouée: {str(e)}")
        
        # Méthode 3: Récupérer toutes les données et filtrer après
        if not response or not response.data:
            try:
                st.info("🔄 Tentative de récupération de toutes les données...")
                response = supabase.table(table_name)\
                    .select("date, open, high, low, close, volume")\
                    .order('date', desc=False)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    st.success(f"✅ Méthode 3 (tout): {len(response.data)} entrées totales")
                    # Filtrer en Python
                    df_temp = pd.DataFrame(response.data)
                    df_temp['date'] = pd.to_datetime(df_temp['date'])
                    df_temp = df_temp[(df_temp['date'].dt.date >= start_date) & 
                                      (df_temp['date'].dt.date <= end_date)]
                    if len(df_temp) > 0:
                        st.success(f"✅ Après filtrage: {len(df_temp)} entrées dans la période demandée")
                        response.data = df_temp.to_dict('records')
            except Exception as e:
                st.error(f"❌ Méthode 3 échouée: {str(e)}")
        
        if response and response.data and len(response.data) > 0:
            df = pd.DataFrame(response.data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Afficher les statistiques
            st.info(f"📊 Période des données: {df['date'].min()} à {df['date'].max()}")
            
            df = df.set_index('date')
            
            # Resampler par jour
            daily_df = pd.DataFrame({
                'open': df['open'].resample('D').first(),
                'high': df['high'].resample('D').max(),
                'low': df['low'].resample('D').min(),
                'close': df['close'].resample('D').last(),
                'volume': df['volume'].resample('D').sum()
            }).dropna()
            
            if len(daily_df) > 0:
                st.success(f"✅ {ticker}: {len(response.data)} entrées → {len(daily_df)} jours")
                return daily_df
            else:
                st.warning(f"⚠️ {ticker}: Données vides après regroupement")
                return None
        else:
            st.error(f"❌ Aucune donnée trouvée pour {ticker}")
            return None
            
    except Exception as e:
        st.error(f"❌ Erreur pour {ticker}: {str(e)}")
        import traceback
        st.error(f"Détails: {traceback.format_exc()}")
        return None

def calculate_technical_indicators(df):
    """Calcule les indicateurs techniques"""
    # SMA 20 et 50 jours
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['BB_upper'] = sma_20 + (std_20 * 2)
    df['BB_lower'] = sma_20 - (std_20 * 2)
    
    # RSI (14 jours)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# =============================================
# INTERFACE PRINCIPALE
# =============================================

st.markdown("### 📊 STOCK CHART VIEWER")

# Récupérer la liste des actions disponibles
available_stocks = get_available_stocks()

if not available_stocks:
    st.error("❌ Aucune action disponible")
    st.stop()

# ===== TEST DE CONNEXION =====
with st.expander("🔍 DIAGNOSTIC DE CONNEXION"):
    if st.button("Tester la connexion Supabase"):
        try:
            test = supabase.table("aapl_h4_data").select("date, close").limit(5).execute()
            if test.data:
                st.success("✅ Connexion Supabase OK!")
                st.json(test.data)
            else:
                st.error("❌ Table accessible mais vide")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# ===== SÉLECTION DE L'ACTION =====
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_stock = st.selectbox(
        "📈 Sélectionnez une action",
        options=available_stocks,
        index=0,
        help="Choisissez l'action à analyser"
    )

with col2:
    chart_type = st.selectbox(
        "Type de graphique",
        options=['Candlestick', 'Line', 'OHLC'],
        index=0
    )

with col3:
    show_volume = st.checkbox("Afficher volume", value=True)

# Afficher la plage de dates disponible
if st.checkbox("🔍 Voir les dates disponibles", value=True):
    with st.spinner(f"Récupération des dates pour {selected_stock}..."):
        first_date, last_date = get_date_range_for_stock(selected_stock)
        if first_date and last_date:
            st.success(f"📅 Données disponibles de **{first_date}** à **{last_date}**")
        else:
            st.warning("⚠️ Impossible de récupérer les dates disponibles")

st.markdown('<hr>', unsafe_allow_html=True)

# ===== PÉRIODE D'ANALYSE =====
st.markdown("#### 📅 TIME PERIOD")

col_date1, col_date2, col_date3 = st.columns([2, 2, 2])

# Période prédéfinie d'abord
with col_date3:
    period_preset = st.selectbox(
        "Période prédéfinie",
        options=['Personnalisée', '1 Semaine', '1 Mois', '3 Mois', '6 Mois', '1 An', '2 Ans', '5 Ans'],
        index=3  # 3 Mois par défaut
    )

# Définir les dates par défaut
end_date = datetime.now().date()
if period_preset == '1 Semaine':
    start_date = (datetime.now() - timedelta(days=7)).date()
elif period_preset == '1 Mois':
    start_date = (datetime.now() - timedelta(days=30)).date()
elif period_preset == '3 Mois':
    start_date = (datetime.now() - timedelta(days=90)).date()
elif period_preset == '6 Mois':
    start_date = (datetime.now() - timedelta(days=180)).date()
elif period_preset == '1 An':
    start_date = (datetime.now() - timedelta(days=365)).date()
elif period_preset == '2 Ans':
    start_date = (datetime.now() - timedelta(days=730)).date()
elif period_preset == '5 Ans':
    start_date = (datetime.now() - timedelta(days=1825)).date()
else:
    start_date = (datetime.now() - timedelta(days=90)).date()

with col_date1:
    start_date = st.date_input(
        "Date de début",
        value=start_date,
        max_value=datetime.now()
    )

with col_date2:
    end_date = st.date_input(
        "Date de fin",
        value=end_date,
        max_value=datetime.now()
    )

if start_date >= end_date:
    st.error("❌ La date de début doit être antérieure à la date de fin")
    st.stop()

# Afficher la période sélectionnée
period_days = (end_date - start_date).days
st.info(f"📊 Période sélectionnée: **{period_days} jours** (du {start_date} au {end_date})")

st.markdown('<hr>', unsafe_allow_html=True)

# ===== OPTIONS D'ANALYSE =====
st.markdown("#### 🔧 TECHNICAL ANALYSIS")

col_tech1, col_tech2 = st.columns(2)

with col_tech1:
    show_sma = st.checkbox("Afficher SMA (20, 50)", value=True)
    show_bollinger = st.checkbox("Afficher Bollinger Bands", value=False)

with col_tech2:
    show_rsi = st.checkbox("Afficher RSI", value=False)

st.markdown('<hr>', unsafe_allow_html=True)

# ===== CHARGER ET AFFICHER LES DONNÉES =====
if st.button("📊 AFFICHER LE GRAPHIQUE", use_container_width=True):
    
    with st.spinner(f"Chargement des données pour {selected_stock}..."):
        df = get_stock_data(selected_stock, start_date, end_date)
    
    if df is None or len(df) == 0:
        st.error(f"❌ Aucune donnée disponible pour {selected_stock} sur cette période")
        st.info("💡 Vérifiez que:")
        st.info("1. La table existe dans Supabase")
        st.info("2. Les données couvrent la période sélectionnée")
        st.info("3. Utilisez le bouton 'Voir les dates disponibles' ci-dessus")
        st.stop()
    
    st.success(f"✅ {len(df)} jours de données chargés pour {selected_stock}")
    
    # Calculer les indicateurs techniques
    df = calculate_technical_indicators(df)
    
    # ===== MÉTRIQUES PRINCIPALES =====
    st.markdown("#### 📊 KEY METRICS")
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    period_start_price = df['close'].iloc[0]
    period_return = ((current_price - period_start_price) / period_start_price) * 100
    
    metric_cols = st.columns(6)
    
    with metric_cols[0]:
        st.metric("Prix actuel", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
    
    with metric_cols[1]:
        st.metric("Plus haut", f"${df['high'].max():.2f}")
    
    with metric_cols[2]:
        st.metric("Plus bas", f"${df['low'].min():.2f}")
    
    with metric_cols[3]:
        st.metric("Volume moyen", f"{df['volume'].mean():.0f}")
    
    with metric_cols[4]:
        volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Volatilité (ann.)", f"{volatility:.2f}%")
    
    with metric_cols[5]:
        st.metric("Rendement période", f"{period_return:+.2f}%")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== GRAPHIQUE PRINCIPAL =====
    st.markdown(f"#### 📈 {selected_stock} - PRICE CHART")
    
    # Créer le graphique principal
    fig = go.Figure()
    
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=selected_stock,
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF0000'
        ))
    elif chart_type == 'OHLC':
        fig.add_trace(go.Ohlc(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=selected_stock,
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF0000'
        ))
    else:  # Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name=selected_stock,
            line=dict(color='#FFAA00', width=2)
        ))
    
    # Ajouter les indicateurs techniques
    if show_sma:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#00FFFF', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='#FF00FF', width=1, dash='dot')
        ))
    
    if show_bollinger:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='#888', width=1, dash='dash'),
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='#888', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(136, 136, 136, 0.1)',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"{selected_stock} - {start_date} to {end_date}",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        xaxis=dict(
            gridcolor='#333',
            showgrid=True,
            title="Date",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            gridcolor='#333',
            showgrid=True,
            title="Price (USD)"
        ),
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== GRAPHIQUE DU VOLUME =====
    if show_volume:
        st.markdown("#### 📊 VOLUME")
        
        fig_volume = go.Figure()
        
        colors = ['#00FF00' if df['close'].iloc[i] >= df['open'].iloc[i] else '#FF0000' 
                  for i in range(len(df))]
        
        fig_volume.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ))
        
        fig_volume.update_layout(
            title="Trading Volume",
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
                title="Volume"
            ),
            height=250
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # ===== GRAPHIQUE RSI =====
    if show_rsi:
        st.markdown("#### 📉 RSI (Relative Strength Index)")
        
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#FFAA00', width=2)
        ))
        
        # Lignes de référence
        fig_rsi.add_shape(
            type="line",
            x0=df.index[0],
            x1=df.index[-1],
            y0=70,
            y1=70,
            line=dict(color="#FF0000", width=1, dash="dash")
        )
        fig_rsi.add_shape(
            type="line",
            x0=df.index[0],
            x1=df.index[-1],
            y0=30,
            y1=30,
            line=dict(color="#00FF00", width=1, dash="dash")
        )
        
        fig_rsi.update_layout(
            title="RSI (14 periods)",
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
                title="RSI",
                range=[0, 100]
            ),
            height=250
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Interprétation RSI
        current_rsi = df['RSI'].iloc[-1]
        if current_rsi > 70:
            st.warning(f"⚠️ RSI = {current_rsi:.2f} - Action potentiellement SURACHETÉ")
        elif current_rsi < 30:
            st.warning(f"⚠️ RSI = {current_rsi:.2f} - Action potentiellement SURVENDU")
        else:
            st.info(f"✅ RSI = {current_rsi:.2f} - Zone NEUTRE")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== STATISTIQUES DÉTAILLÉES =====
    st.markdown("#### 📊 DETAILED STATISTICS")
    
    stats_cols = st.columns(4)
    
    with stats_cols[0]:
        st.markdown("**Prix**")
        st.write(f"• Ouverture: ${df['open'].iloc[0]:.2f}")
        st.write(f"• Clôture: ${df['close'].iloc[-1]:.2f}")
        st.write(f"• Plus haut: ${df['high'].max():.2f}")
        st.write(f"• Plus bas: ${df['low'].min():.2f}")
    
    with stats_cols[1]:
        st.markdown("**Rendements**")
        daily_returns = df['close'].pct_change()
        st.write(f"• Rendement total: {period_return:+.2f}%")
        st.write(f"• Rdt moyen jour: {daily_returns.mean()*100:+.3f}%")
        st.write(f"• Meilleur jour: {daily_returns.max()*100:+.2f}%")
        st.write(f"• Pire jour: {daily_returns.min()*100:+.2f}%")
    
    with stats_cols[2]:
        st.markdown("**Volatilité**")
        st.write(f"• Écart-type: {daily_returns.std()*100:.2f}%")
        st.write(f"• Vol. annualisée: {volatility:.2f}%")
        st.write(f"• Amplitude moy.: ${(df['high'] - df['low']).mean():.2f}")
    
    with stats_cols[3]:
        st.markdown("**Volume**")
        st.write(f"• Volume total: {df['volume'].sum():,.0f}")
        st.write(f"• Volume moyen: {df['volume'].mean():,.0f}")
        st.write(f"• Volume max: {df['volume'].max():,.0f}")
        st.write(f"• Volume min: {df['volume'].min():,.0f}")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== TABLEAU DES DONNÉES =====
    with st.expander("📋 VOIR LES DONNÉES BRUTES"):
        display_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        display_df = display_df.round(2)
        display_df['volume'] = display_df['volume'].astype(int)
        
        st.dataframe(
            display_df.tail(50),
            use_container_width=True,
            height=400
        )

# ===== FOOTER =====
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | STOCK CHART VIEWER | DONNÉES HISTORIQUES SUPABASE<br>
    ANALYSE TECHNIQUE COMPLÈTE • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
