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
    page_title="Bloomberg Terminal - Portfolio Simulator",
    page_icon="💼",
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
        <div>💼 BLOOMBERG ENS® TERMINAL - PORTFOLIO SIMULATOR</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">MARKETS</a>
    </div>
    <div>{datetime.now().strftime("%H:%M:%S")} UTC</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# FONCTIONS UTILITAIRES
# =============================================

@st.cache_data(ttl=3600)
def get_available_stocks():
    """Récupère la liste des actions disponibles (basé sur vos fichiers CSV)"""
    # Liste complète des actions disponibles dans votre base Supabase
    stocks = [
        'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 
        'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 
        'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 
        'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 
        'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 
        'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF_B', 'BG', 'BIIB', 
        'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BMY', 'BR', 'BRK_B', 'BRO', 'BSX', 
        'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 
        'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 
        'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 
        'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 
        'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CVS', 'CVX', 'D', 
        'DAL', 'DAN', 'DAR', 'DD', 'DE', 'DG', 'DGX', 'DHI', 'DIS', 'DLR', 
        'DLTR', 'DOV', 'DOW', 'DPZ', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 
        'EBAY', 'ECL', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 
        'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 
        'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 
        'FDS', 'FDX', 'FE', 'FFIV', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX', 'FOXA', 
        'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 
        'GIS', 'GL', 'GM', 'GNRC', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 
        'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 
        'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 
        'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 
        'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 
        'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 
        'KLAC', 'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 
        'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 
        'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 
        'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 
        'MMC', 'MNST', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI', 
        'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 
        'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 
        'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 
        'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYC', 'PAYX', 'PCAR', 'PCG', 
        'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 
        'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 
        'PTC', 'PWR', 'PYPL', 'QCOM', 'QQQ', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 
        'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 
        'RVTY', 'SBAC', 'SBUX', 'SEDG', 'SEE', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 
        'SO', 'SPG', 'SPGI', 'SRE', 'VIXM'
    ]
    
    return sorted(stocks)

@st.cache_data(ttl=300)
def get_stock_data(ticker, start_date, end_date):
    """Récupère les données d'une action depuis Supabase"""
    try:
        table_name = f"{ticker.lower()}_h4_data"
        
        # Convertir les dates en string ISO
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Requête Supabase
        response = supabase.table(table_name)\
            .select("date, open, high, low, close, volume")\
            .gte('date', start_str)\
            .lte('date', end_str)\
            .order('date', desc=False)\
            .execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Erreur pour {ticker}: {str(e)}")
        return None

def calculate_portfolio_metrics(portfolio_df, weights):
    """Calcule les métriques du portefeuille"""
    # Rendements quotidiens
    returns = portfolio_df.pct_change().dropna()
    
    # Rendement du portefeuille
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Valeur cumulée du portefeuille
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Métriques
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    # Volatilité annualisée
    volatility = portfolio_returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (simplifié, sans taux sans risque)
    sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0
    
    # Max Drawdown
    cumulative = cumulative_returns
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cumulative_returns,
        'portfolio_returns': portfolio_returns
    }

# =============================================
# INTERFACE PRINCIPALE
# =============================================

st.markdown("### 💼 PORTFOLIO CONFIGURATION")

# Récupérer la liste des actions disponibles
available_stocks = get_available_stocks()

if not available_stocks:
    st.error("❌ Aucune action disponible dans la base de données")
    st.stop()

# ===== SÉLECTION DES ACTIONS =====
st.markdown("#### 📊 SELECT STOCKS")

col1, col2 = st.columns([3, 1])

with col1:
    selected_stocks = st.multiselect(
        "Choisissez les actions pour votre portefeuille",
        options=available_stocks,
        default=['AAPL', 'MSFT', 'GOOGL'] if all(s in available_stocks for s in ['AAPL', 'MSFT', 'GOOGL']) else available_stocks[:3],
        help="Sélectionnez jusqu'à 20 actions"
    )

with col2:
    st.metric("Actions sélectionnées", len(selected_stocks))

if not selected_stocks:
    st.warning("⚠️ Veuillez sélectionner au moins une action")
    st.stop()

# Limiter à 20 actions
if len(selected_stocks) > 20:
    st.warning("⚠️ Maximum 20 actions. Les actions supplémentaires seront ignorées.")
    selected_stocks = selected_stocks[:20]

st.markdown('<hr>', unsafe_allow_html=True)

# ===== PÉRIODE D'ANALYSE =====
st.markdown("#### 📅 TIME PERIOD")

col_date1, col_date2, col_date3 = st.columns([2, 2, 2])

with col_date1:
    start_date = st.date_input(
        "Date de début",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )

with col_date2:
    end_date = st.date_input(
        "Date de fin",
        value=datetime.now(),
        max_value=datetime.now()
    )

with col_date3:
    # Raccourcis de période
    period_preset = st.selectbox(
        "Période prédéfinie",
        options=['Personnalisée', '1 Mois', '3 Mois', '6 Mois', '1 An', '2 Ans', '5 Ans', 'Max'],
        index=0
    )
    
    if period_preset != 'Personnalisée':
        end_date = datetime.now().date()
        if period_preset == '1 Mois':
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
        elif period_preset == 'Max':
            start_date = (datetime.now() - timedelta(days=3650)).date()

# Vérifier que start_date < end_date
if start_date >= end_date:
    st.error("❌ La date de début doit être antérieure à la date de fin")
    st.stop()

st.markdown('<hr>', unsafe_allow_html=True)

# ===== ALLOCATION DU PORTEFEUILLE =====
st.markdown("#### 💰 PORTFOLIO ALLOCATION")

# Choix du mode d'allocation
allocation_mode = st.radio(
    "Mode d'allocation",
    options=['Équipondéré', 'Personnalisé', 'Value Weighted'],
    horizontal=True,
    help="Équipondéré: poids égaux | Personnalisé: définir manuellement | Value Weighted: pondération par capitalisation"
)

weights = {}

if allocation_mode == 'Équipondéré':
    # Répartition égale
    equal_weight = 100.0 / len(selected_stocks)
    for stock in selected_stocks:
        weights[stock] = equal_weight
    
    st.info(f"✅ Allocation équipondérée: {equal_weight:.2f}% par action")

elif allocation_mode == 'Personnalisé':
    # Sliders pour chaque action
    st.markdown("**Définissez le poids de chaque action (en %) :**")
    
    # Créer des colonnes pour les sliders
    n_cols = 4
    cols = st.columns(n_cols)
    
    for idx, stock in enumerate(selected_stocks):
        with cols[idx % n_cols]:
            weights[stock] = st.slider(
                f"{stock}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / len(selected_stocks),
                step=0.5,
                key=f"weight_{stock}"
            )
    
    # Vérifier que la somme fait 100%
    total_weight = sum(weights.values())
    if abs(total_weight - 100.0) > 0.1:
        st.warning(f"⚠️ Total: {total_weight:.2f}% (devrait être 100%)")
    else:
        st.success(f"✅ Portefeuille équilibré: {total_weight:.2f}%")

else:  # Value Weighted
    st.info("🔄 Chargement des capitalisations boursières...")
    # Pour l'instant, on fait une répartition équipondérée
    # TODO: Implémenter la récupération de la capitalisation boursière
    equal_weight = 100.0 / len(selected_stocks)
    for stock in selected_stocks:
        weights[stock] = equal_weight
    
    st.warning("⚠️ Mode Value Weighted pas encore implémenté. Allocation équipondérée utilisée.")

st.markdown('<hr>', unsafe_allow_html=True)

# ===== BOUTON DE SIMULATION =====
if st.button("🚀 LANCER LA SIMULATION", use_container_width=True):
    
    st.markdown("### 📊 SIMULATION RESULTS")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Charger les données pour chaque action
    portfolio_data = pd.DataFrame()
    weight_array = []
    
    for idx, stock in enumerate(selected_stocks):
        status_text.text(f"Chargement de {stock}... ({idx+1}/{len(selected_stocks)})")
        progress_bar.progress((idx + 1) / len(selected_stocks))
        
        df = get_stock_data(stock, start_date, end_date)
        
        if df is not None and len(df) > 0:
            portfolio_data[stock] = df['close']
            weight_array.append(weights[stock] / 100.0)
        else:
            st.warning(f"⚠️ Pas de données pour {stock} sur cette période")
    
    progress_bar.empty()
    status_text.empty()
    
    if portfolio_data.empty:
        st.error("❌ Aucune donnée disponible pour les actions sélectionnées sur cette période")
        st.stop()
    
    # Normaliser les poids
    weight_array = np.array(weight_array)
    weight_array = weight_array / weight_array.sum()
    
    # Supprimer les NaN
    portfolio_data = portfolio_data.fillna(method='ffill').fillna(method='bfill')
    
    # Calculer les métriques
    metrics = calculate_portfolio_metrics(portfolio_data, weight_array)
    
    # ===== AFFICHAGE DES MÉTRIQUES =====
    st.markdown("#### 📈 PERFORMANCE METRICS")
    
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.metric("Total Return", f"{metrics['total_return']:+.2f}%")
    
    with metric_cols[1]:
        st.metric("Volatility (ann.)", f"{metrics['volatility']:.2f}%")
    
    with metric_cols[2]:
        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    
    with metric_cols[3]:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    
    with metric_cols[4]:
        period_days = (end_date - start_date).days
        st.metric("Period (days)", f"{period_days}")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== GRAPHIQUE DE PERFORMANCE =====
    st.markdown("#### 📊 PORTFOLIO VALUE EVOLUTION")
    
    fig = go.Figure()
    
    # Courbe du portefeuille
    portfolio_value = metrics['cumulative_returns'] * 100
    
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#FFAA00', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,170,0,0.1)',
        hovertemplate='<b>Portfolio</b><br>Value: %{y:.2f}<br>Date: %{x}<extra></extra>'
    ))
    
    # Ligne de référence à 100
    fig.add_shape(
        type="line",
        x0=portfolio_value.index[0],
        x1=portfolio_value.index[-1],
        y0=100,
        y1=100,
        line=dict(color="#666", width=1, dash="dash")
    )
    
    fig.update_layout(
        title=f"Portfolio Performance ({start_date} to {end_date})",
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
            title="Portfolio Value (Base 100)"
        ),
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== GRAPHIQUE DES ACTIONS INDIVIDUELLES =====
    st.markdown("#### 📈 INDIVIDUAL STOCKS PERFORMANCE")
    
    fig_stocks = go.Figure()
    
    # Couleurs
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00',
              '#FF1493', '#00CED1', '#32CD32', '#FFD700']
    
    for idx, stock in enumerate(portfolio_data.columns):
        normalized = (portfolio_data[stock] / portfolio_data[stock].iloc[0]) * 100
        
        fig_stocks.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            mode='lines',
            name=f"{stock} ({weights[stock]:.1f}%)",
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate=f'<b>{stock}</b><br>%{{y:.2f}}<br>%{{x}}<extra></extra>'
        ))
    
    # Ajouter le portefeuille
    fig_stocks.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode='lines',
        name='PORTFOLIO',
        line=dict(color='#FFAA00', width=4, dash='dot'),
        hovertemplate='<b>PORTFOLIO</b><br>%{y:.2f}<br>%{x}<extra></extra>'
    ))
    
    fig_stocks.update_layout(
        title="Individual Stocks vs Portfolio (Base 100)",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        xaxis=dict(gridcolor='#333', showgrid=True, title="Date"),
        yaxis=dict(gridcolor='#333', showgrid=True, title="Normalized Value (Base 100)"),
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_stocks, use_container_width=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== COMPOSITION DU PORTEFEUILLE =====
    st.markdown("#### 💼 PORTFOLIO COMPOSITION")
    
    # Graphique en camembert
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.4,
        marker=dict(
            colors=colors[:len(weights)],
            line=dict(color='#000', width=2)
        ),
        textfont=dict(size=12, color='#000'),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
    )])
    
    fig_pie.update_layout(
        title="Portfolio Allocation",
        paper_bgcolor='#000',
        font=dict(color='#FFAA00', size=10),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        )
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== TABLEAU DE PERFORMANCE PAR ACTION =====
    st.markdown("#### 📊 INDIVIDUAL STOCK PERFORMANCE")
    
    perf_data = []
    
    for stock in portfolio_data.columns:
        start_price = portfolio_data[stock].iloc[0]
        end_price = portfolio_data[stock].iloc[-1]
        stock_return = ((end_price - start_price) / start_price) * 100
        
        stock_returns = portfolio_data[stock].pct_change().dropna()
        stock_vol = stock_returns.std() * np.sqrt(252) * 100
        
        perf_data.append({
            'Stock': stock,
            'Weight (%)': f"{weights[stock]:.2f}",
            'Start Price': f"${start_price:.2f}",
            'End Price': f"${end_price:.2f}",
            'Return (%)': f"{stock_return:+.2f}",
            'Volatility (%)': f"{stock_vol:.2f}"
        })
    
    perf_df = pd.DataFrame(perf_data)
    
    # Afficher le tableau avec un style personnalisé
    st.dataframe(
        perf_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== ANALYSE DES RISQUES =====
    st.markdown("#### ⚠️ RISK ANALYSIS")
    
    # Matrice de corrélation
    corr_matrix = portfolio_data.pct_change().corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, '#FF0000'],
            [0.5, '#000000'],
            [1, '#00FF00']
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
        title="Correlation Matrix",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        xaxis=dict(tickfont=dict(color='#FFAA00', size=9)),
        yaxis=dict(tickfont=dict(color='#FFAA00', size=9)),
        height=400
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.caption("""
    **📖 Interprétation de la corrélation :**
    - 🟢 **Vert (1.0)** : Corrélation parfaite - les actions évoluent ensemble
    - ⚫ **Noir (0)** : Aucune corrélation - mouvements indépendants
    - 🔴 **Rouge (-1.0)** : Corrélation inverse - les actions évoluent en sens opposé
    """)

# ===== FOOTER =====
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | PORTFOLIO SIMULATOR | DONNÉES HISTORIQUES SUPABASE<br>
    SIMULATION COMPLÈTE • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
