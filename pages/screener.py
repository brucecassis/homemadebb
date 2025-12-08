import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
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
    <div>{current_time} UTC • FINANCIAL MODELING PREP API</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS API
# =============================================
@st.cache_data(ttl=600)
def get_stock_screener(market_cap_min=None, market_cap_max=None, 
                       price_min=None, price_max=None,
                       beta_min=None, beta_max=None,
                       volume_min=None, volume_max=None,
                       dividend_min=None, dividend_max=None,
                       sector=None, exchange=None, limit=1000):
    """
    Screener d'actions via FMP API
    """
    url = f"https://financialmodelingprep.com/api/v3/stock-screener"
    
    params = {
        'apikey': FMP_API_KEY,
        'limit': limit
    }
    
    if market_cap_min:
        params['marketCapMoreThan'] = market_cap_min
    if market_cap_max:
        params['marketCapLowerThan'] = market_cap_max
    if price_min:
        params['priceMoreThan'] = price_min
    if price_max:
        params['priceLowerThan'] = price_max
    if beta_min:
        params['betaMoreThan'] = beta_min
    if beta_max:
        params['betaLowerThan'] = beta_max
    if volume_min:
        params['volumeMoreThan'] = volume_min
    if volume_max:
        params['volumeLowerThan'] = volume_max
    if dividend_min:
        params['dividendMoreThan'] = dividend_min
    if dividend_max:
        params['dividendLowerThan'] = dividend_max
    if sector and sector != "All":
        params['sector'] = sector
    if exchange and exchange != "All":
        params['exchange'] = exchange
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur API: {str(e)}")
        return []

@st.cache_data(ttl=300)
def get_quote(symbol):
    """Récupère le cours en temps réel"""
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
    params = {'apikey': FMP_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None
    except:
        return None

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
        options=["All", "NASDAQ", "NYSE", "AMEX", "EURONEXT", "TSX", "LSE"],
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
        "Market Cap (Mds $)",
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

with col4:
    st.markdown("**📈 DIVIDENDES & VOLUME**")
    
    dividend_min = st.number_input("Dividend Yield min (%)", min_value=0.0, value=0.0, step=0.5)
    
    volume_range = st.selectbox(
        "Volume moyen",
        options=["All", "Low (<100K)", "Medium (100K-1M)", "High (>1M)"],
        index=0
    )
    
    volume_min, volume_max = None, None
    if volume_range == "Low (<100K)":
        volume_max = 100000
    elif volume_range == "Medium (100K-1M)":
        volume_min = 100000
        volume_max = 1000000
    elif volume_range == "High (>1M)":
        volume_min = 1000000

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
    
    with st.spinner('🔍 Recherche en cours...'):
        results = get_stock_screener(
            market_cap_min=market_cap_min,
            market_cap_max=market_cap_max,
            price_min=price_min if price_min > 0 else None,
            price_max=price_max if price_max < 10000 else None,
            beta_min=beta_min,
            beta_max=beta_max,
            volume_min=volume_min,
            volume_max=volume_max,
            dividend_min=dividend_min if dividend_min > 0 else None,
            sector=sector if sector != "All" else None,
            exchange=exchange if exchange != "All" else None,
            limit=500
        )
    
    if results:
        # Créer un DataFrame
        df = pd.DataFrame(results)
        
        # Colonnes à afficher
        display_columns = ['symbol', 'companyName', 'sector', 'exchange', 
                          'price', 'changesPercentage', 'marketCap', 
                          'volume', 'beta', 'lastDiv']
        
        # Filtrer les colonnes disponibles
        available_columns = [col for col in display_columns if col in df.columns]
        df_display = df[available_columns].copy()
        
        # Renommer les colonnes
        column_names = {
            'symbol': 'Ticker',
            'companyName': 'Company',
            'sector': 'Sector',
            'exchange': 'Exchange',
            'price': 'Price ($)',
            'changesPercentage': 'Change (%)',
            'marketCap': 'Market Cap ($)',
            'volume': 'Volume',
            'beta': 'Beta',
            'lastDiv': 'Dividend ($)'
        }
        df_display.rename(columns=column_names, inplace=True)
        
        # Formater les nombres
        if 'Market Cap ($)' in df_display.columns:
            df_display['Market Cap ($)'] = df_display['Market Cap ($)'].apply(
                lambda x: f"{x/1e9:.2f}B" if x > 1e9 else f"{x/1e6:.2f}M" if x > 1e6 else f"{x:,.0f}"
            )
        
        if 'Volume' in df_display.columns:
            df_display['Volume'] = df_display['Volume'].apply(
                lambda x: f"{x/1e6:.2f}M" if x > 1e6 else f"{x/1e3:.2f}K" if x > 1e3 else f"{x:,.0f}"
            )
        
        if 'Price ($)' in df_display.columns:
            df_display['Price ($)'] = df_display['Price ($)'].apply(lambda x: f"${x:.2f}")
        
        if 'Change (%)' in df_display.columns:
            df_display['Change (%)'] = df_display['Change (%)'].apply(lambda x: f"{x:+.2f}%")
        
        if 'Beta' in df_display.columns:
            df_display['Beta'] = df_display['Beta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        
        if 'Dividend ($)' in df_display.columns:
            df_display['Dividend ($)'] = df_display['Dividend ($)'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) and x > 0 else "N/A"
            )
        
        # Afficher le nombre de résultats
        st.success(f"✅ **{len(df_display)} actions trouvées**")
        
        # Statistiques rapides
        st.markdown("#### 📊 STATISTIQUES")
        
        stat_cols = st.columns(5)
        
        with stat_cols[0]:
            avg_change = df['changesPercentage'].mean() if 'changesPercentage' in df.columns else 0
            st.metric("Variation moyenne", f"{avg_change:+.2f}%")
        
        with stat_cols[1]:
            avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
            st.metric("Volume moyen", f"{avg_volume/1e6:.2f}M")
        
        with stat_cols[2]:
            avg_beta = df['beta'].mean() if 'beta' in df.columns else 0
            st.metric("Beta moyen", f"{avg_beta:.2f}")
        
        with stat_cols[3]:
            total_mcap = df['marketCap'].sum() if 'marketCap' in df.columns else 0
            st.metric("Cap. totale", f"${total_mcap/1e9:.2f}B")
        
        with stat_cols[4]:
            avg_div = df['lastDiv'].mean() if 'lastDiv' in df.columns else 0
            st.metric("Div. moyen", f"${avg_div:.2f}")
        
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
        
        # Trier le DataFrame
        df_display = df_display.sort_values(by=sort_by, ascending=False)
        
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
        
        # ===== ANALYSE PAR SECTEUR =====
        if 'Sector' in df_display.columns:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("#### 📊 RÉPARTITION PAR SECTEUR")
            
            sector_counts = df['sector'].value_counts()
            
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
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | FINANCIAL MODELING PREP API | SYSTÈME OPÉRATIONNEL<br>
    SCREENER ACTIF • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
