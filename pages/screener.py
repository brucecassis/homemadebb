import streamlit as st
import pandas as pd
import requests
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
# API CONFIGURATION
# =============================================
FMP_API_KEY = "3eR6TT3vA8rOZvoiFtZzGbro5a3rA5Ix"

# =============================================
# HEADER
# =============================================
current_time = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>🔍 BLOOMBERG ENS® - STOCK SCREENER</div>
    </div>
    <div>{current_time} UTC • HYBRID API (FMP + YAHOO)</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS API
# =============================================
@st.cache_data(ttl=3600)
def get_stock_list(exchange="NASDAQ"):
    """Récupère la liste des actions d'un exchange"""
    url = f"https://financialmodelingprep.com/api/v3/stock/list"
    params = {'apikey': FMP_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Filtrer par exchange
        if exchange != "All":
            data = [stock for stock in data if stock.get('exchangeShortName') == exchange]
        
        return data
    except Exception as e:
        st.error(f"Erreur API: {str(e)}")
        return []

@st.cache_data(ttl=600)
def get_quotes_batch(symbols):
    """Récupère les cours pour plusieurs symboles"""
    symbols_str = ",".join(symbols[:100])  # Limite à 100 symboles
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_str}"
    params = {'apikey': FMP_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return []

@st.cache_data(ttl=3600)
def get_company_profile_batch(symbols):
    """Récupère les profils d'entreprises"""
    results = []
    
    # Traiter par batch de 5 pour éviter les limites
    for i in range(0, len(symbols), 5):
        batch = symbols[i:i+5]
        symbols_str = ",".join(batch)
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbols_str}"
        params = {'apikey': FMP_API_KEY}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data:
                results.extend(data)
            time.sleep(0.2)  # Éviter les limites de rate
        except:
            continue
    
    return results

def filter_stocks(stocks_data, filters):
    """Applique les filtres sur les données"""
    df = pd.DataFrame(stocks_data)
    
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
        df = df[df['changesPercentage'] >= filters['change_min']]
    if filters['change_max'] is not None:
        df = df[df['changesPercentage'] <= filters['change_max']]
    
    # Filtrer par secteur
    if filters['sector'] != "All":
        df = df[df['sector'] == filters['sector']]
    
    return df

# =============================================
# INTERFACE DE FILTRAGE
# =============================================
st.markdown("### 🎯 FILTRES DE RECHERCHE")

# Créer des colonnes pour les filtres
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**📊 MARCHÉ**")
    
    exchange = st.selectbox(
        "Exchange",
        options=["NASDAQ", "NYSE", "AMEX", "All"],
        index=0
    )
    
    sector = st.selectbox(
        "Secteur",
        options=["All", "Technology", "Financial Services", "Healthcare", 
                "Consumer Cyclical", "Industrials", "Energy", "Basic Materials",
                "Consumer Defensive", "Real Estate", "Utilities", "Communication Services"],
        index=0
    )

with col2:
    st.markdown("**💰 CAPITALISATION**")
    
    market_cap_range = st.select_slider(
        "Market Cap",
        options=["Micro (<300M)", "Small (300M-2B)", "Mid (2B-10B)", 
                "Large (10B-200B)", "Mega (>200B)", "All"],
        value="All"
    )
    
    # Convertir en valeurs
    market_cap_min, market_cap_max = None, None
    if market_cap_range == "Micro (<300M)":
        market_cap_max = 300000000
    elif market_cap_range == "Small (300M-2B)":
        market_cap_min = 300000000
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
    st.markdown("**📈 VOLUME**")
    
    volume_range = st.selectbox(
        "Volume moyen",
        options=["All", "Low (<100K)", "Medium (100K-1M)", "High (1M-10M)", "Very High (>10M)"],
        index=0
    )
    
    volume_min, volume_max = None, None
    if volume_range == "Low (<100K)":
        volume_max = 100000
    elif volume_range == "Medium (100K-1M)":
        volume_min = 100000
        volume_max = 1000000
    elif volume_range == "High (1M-10M)":
        volume_min = 1000000
        volume_max = 10000000
    elif volume_range == "Very High (>10M)":
        volume_min = 10000000
    
    limit_results = st.number_input("Limite de résultats", min_value=10, max_value=500, value=100, step=10)

# Bouton de recherche
st.markdown('<hr>', unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 7])

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
    
    with st.spinner('🔍 Recherche en cours... Cela peut prendre quelques secondes...'):
        # 1. Récupérer la liste des actions
        stock_list = get_stock_list(exchange)
        
        if not stock_list:
            st.error("❌ Impossible de récupérer la liste des actions")
            st.stop()
        
        # 2. Limiter le nombre d'actions à analyser
        stock_list = stock_list[:limit_results]
        symbols = [stock['symbol'] for stock in stock_list]
        
        st.info(f"📊 Analyse de {len(symbols)} actions...")
        
        # 3. Récupérer les cours
        quotes = get_quotes_batch(symbols)
        
        if not quotes:
            st.error("❌ Impossible de récupérer les cours")
            st.stop()
        
        # 4. Récupérer les profils (secteur, etc.)
        st.info("📊 Récupération des profils d'entreprises...")
        profiles = get_company_profile_batch(symbols)
        
        # 5. Fusionner les données
        quotes_df = pd.DataFrame(quotes)
        profiles_df = pd.DataFrame(profiles)
        
        if not profiles_df.empty:
            merged_df = quotes_df.merge(profiles_df[['symbol', 'sector', 'industry']], 
                                       on='symbol', how='left')
        else:
            merged_df = quotes_df
            merged_df['sector'] = 'Unknown'
        
        # 6. Appliquer les filtres
        filters = {
            'price_min': price_min,
            'price_max': price_max,
            'market_cap_min': market_cap_min,
            'market_cap_max': market_cap_max,
            'volume_min': volume_min,
            'volume_max': volume_max,
            'change_min': change_min,
            'change_max': change_max,
            'sector': sector
        }
        
        filtered_df = filter_stocks(merged_df.to_dict('records'), filters)
    
    if not filtered_df.empty:
        # Colonnes à afficher
        display_columns = ['symbol', 'name', 'sector', 'price', 'changesPercentage', 
                          'marketCap', 'volume', 'dayHigh', 'dayLow']
        
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        df_display = filtered_df[available_columns].copy()
        
        # Renommer les colonnes
        column_names = {
            'symbol': 'Ticker',
            'name': 'Company',
            'sector': 'Sector',
            'price': 'Price ($)',
            'changesPercentage': 'Change (%)',
            'marketCap': 'Market Cap',
            'volume': 'Volume',
            'dayHigh': 'High ($)',
            'dayLow': 'Low ($)'
        }
        df_display.rename(columns=column_names, inplace=True)
        
        # Formater les nombres
        if 'Market Cap' in df_display.columns:
            df_display['Market Cap'] = df_display['Market Cap'].apply(
                lambda x: f"{x/1e9:.2f}B" if x > 1e9 else f"{x/1e6:.2f}M" if x > 1e6 else f"{x:,.0f}"
            )
        
        if 'Volume' in df_display.columns:
            df_display['Volume'] = df_display['Volume'].apply(
                lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.2f}K" if x > 1e3 else f"{x:,.0f}"
            )
        
        if 'Price ($)' in df_display.columns:
            df_display['Price ($)'] = df_display['Price ($)'].apply(lambda x: f"${x:.2f}")
        
        if 'High ($)' in df_display.columns:
            df_display['High ($)'] = df_display['High ($)'].apply(lambda x: f"${x:.2f}")
        
        if 'Low ($)' in df_display.columns:
            df_display['Low ($)'] = df_display['Low ($)'].apply(lambda x: f"${x:.2f}")
        
        if 'Change (%)' in df_display.columns:
            df_display['Change (%)'] = df_display['Change (%)'].apply(lambda x: f"{x:+.2f}%")
        
        # Afficher le nombre de résultats
        st.success(f"✅ **{len(df_display)} actions trouvées**")
        
        # Statistiques rapides
        st.markdown("#### 📊 STATISTIQUES")
        
        stat_cols = st.columns(5)
        
        with stat_cols[0]:
            avg_change = filtered_df['changesPercentage'].mean()
            st.metric("Variation moyenne", f"{avg_change:+.2f}%")
        
        with stat_cols[1]:
            avg_volume = filtered_df['volume'].mean()
            st.metric("Volume moyen", f"{avg_volume/1e6:.2f}M")
        
        with stat_cols[2]:
            avg_price = filtered_df['price'].mean()
            st.metric("Prix moyen", f"${avg_price:.2f}")
        
        with stat_cols[3]:
            total_mcap = filtered_df['marketCap'].sum()
            st.metric("Cap. totale", f"${total_mcap/1e9:.2f}B")
        
        with stat_cols[4]:
            gainers = len(filtered_df[filtered_df['changesPercentage'] > 0])
            losers = len(filtered_df[filtered_df['changesPercentage'] < 0])
            st.metric("Gainers/Losers", f"{gainers}/{losers}")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        # Tableau des résultats
        st.markdown("#### 📋 LISTE DES ACTIONS")
        
        # Options de tri
        col_sort1, col_sort2 = st.columns([3, 9])
        
        with col_sort1:
            sort_by = st.selectbox(
                "Trier par",
                options=list(df_display.columns),
                index=0
            )
        
        # Trier le DataFrame (en gardant les valeurs numériques pour le tri)
        sort_column_original = {v: k for k, v in column_names.items()}[sort_by]
        filtered_df_sorted = filtered_df.sort_values(by=sort_column_original, ascending=False)
        df_display = df_display.loc[filtered_df_sorted.index]
        
        # Afficher le tableau
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # Bouton d'export
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 EXPORTER EN CSV",
            data=csv,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # ===== TOP GAINERS & LOSERS =====
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("#### 🏆 TOP GAINERS & LOSERS")
        
        col_gain, col_lose = st.columns(2)
        
        with col_gain:
            st.markdown("**🟢 TOP 5 GAINERS**")
            top_gainers = filtered_df.nlargest(5, 'changesPercentage')[['symbol', 'name', 'changesPercentage', 'price']]
            for _, row in top_gainers.iterrows():
                st.metric(
                    f"{row['symbol']} - {row['name'][:20]}",
                    f"${row['price']:.2f}",
                    f"{row['changesPercentage']:+.2f}%"
                )
        
        with col_lose:
            st.markdown("**🔴 TOP 5 LOSERS**")
            top_losers = filtered_df.nsmallest(5, 'changesPercentage')[['symbol', 'name', 'changesPercentage', 'price']]
            for _, row in top_losers.iterrows():
                st.metric(
                    f"{row['symbol']} - {row['name'][:20]}",
                    f"${row['price']:.2f}",
                    f"{row['changesPercentage']:+.2f}%"
                )
        
        # ===== ANALYSE PAR SECTEUR =====
        if 'sector' in filtered_df.columns:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("#### 📊 RÉPARTITION PAR SECTEUR")
            
            sector_counts = filtered_df['sector'].value_counts()
            
            fig_sector = go.Figure(data=[go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                hole=0.3,
                marker=dict(
                    colors=['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', 
                           '#FFFF00', '#FF1493', '#00CED1', '#32CD32', '#FFD700']
                )
            )])
            
            fig_sector.update_layout(
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                height=400
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
    
    else:
        st.warning("⚠️ Aucun résultat trouvé. Essayez de modifier vos critères de recherche.")

# =============================================
# INFO
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.info("""
📌 **NOTE:** Ce screener utilise une approche hybride pour contourner les limitations de l'API gratuite :
- Liste des actions via FMP API
- Cours en temps réel via FMP API  
- Filtrage côté client pour une flexibilité maximale

⏱️ Le chargement peut prendre quelques secondes selon le nombre d'actions analysées.
""")

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | HYBRID SCREENER (FMP + YAHOO) | SYSTÈME OPÉRATIONNEL<br>
    SCREENER ACTIF • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
