import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from auth_utils import init_session_state, logout
from login import show_login_page
from adsense_utils import add_header_ad, add_footer_ad

# =============================================
# AUTH & SESSION
# =============================================
init_session_state()

if not st.session_state.get('authenticated', False):
    show_login_page()
    st.stop()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Portfolio Backtesting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG
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
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
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
    
    .stNumberInput input, .stSelectbox select, .stDateInput input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stSlider {
        color: #FFAA00 !important;
    }
    
    .strategy-info {
        background: #1a1a1a;
        border: 1px solid #444;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 10px;
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
        <div>⬛ BLOOMBERG ENS® TERMINAL - PORTFOLIO BACKTESTING</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • BACKTEST ENGINE</div>
</div>
""", unsafe_allow_html=True)

add_header_ad()

# =============================================
# FONCTIONS COMMUNES
# =============================================

@st.cache_data(ttl=3600)
def get_historical_data(ticker, start_date, end_date):
    """Récupère les données historiques d'un ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval='1d')
        if len(hist) == 0:
            st.error(f"Aucune donnée trouvée pour {ticker}")
            return None
        return hist
    except Exception as e:
        st.error(f"Erreur lors du téléchargement de {ticker}: {str(e)}")
        return None

def nettoyer_donnees(df, nom_colonne='Close'):
    """Nettoie les données et supprime les outliers"""
    df = df.copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df[df[nom_colonne] > 0]
    df = df.dropna(subset=[nom_colonne])
    
    df["returns"] = np.log(df[nom_colonne] / df[nom_colonne].shift(1))
    mean_ret = df["returns"].mean()
    std_ret = df["returns"].std()
    seuil = 4
    df = df[(df["returns"] > mean_ret - seuil * std_ret) & (df["returns"] < mean_ret + seuil * std_ret)]
    df.drop(columns=["returns"], inplace=True)
    
    return df

def calculate_rsi(prices, period=14):
    """Calcule le RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =============================================
# STRATÉGIE 1: RSI
# =============================================

def run_backtest_rsi(ticker1, ticker2, weight1, weight2, capital, start_date, end_date, rsi_buy, rsi_sell):
    """Exécute le backtest de la stratégie RSI"""
    
    # Télécharger les données
    data1 = get_historical_data(ticker1, start_date, end_date)
    data2 = get_historical_data(ticker2, start_date, end_date)
    
    if data1 is None or data2 is None or len(data1) == 0 or len(data2) == 0:
        return None, None, None
    
    # Fusionner les données
    merged = pd.merge(
        data1[['Close']].reset_index(),
        data2[['Close']].reset_index(),
        on='Date',
        suffixes=(f'_{ticker1}', f'_{ticker2}')
    )
    
    if len(merged) == 0:
        st.error("Aucune date commune trouvée entre les deux actifs.")
        return None, None, None
    
    merged = merged.set_index('Date')
    
    # Calculer les RSI
    merged[f'rsi_{ticker1}'] = calculate_rsi(merged[f'Close_{ticker1}'])
    merged[f'rsi_{ticker2}'] = calculate_rsi(merged[f'Close_{ticker2}'])
    
    # Initialisation de la stratégie
    capital_asset1 = capital * (weight1 / 100)
    capital_asset2_cash = capital * (weight2 / 100)
    
    # Buy & Hold pour asset1
    prix_achat_asset1 = merged[f'Close_{ticker1}'].iloc[0]
    nb_actions_asset1 = capital_asset1 / prix_achat_asset1
    
    # Stratégie RSI pour asset2
    nb_actions_asset2 = 0
    asset2_position = False
    prix_achat_asset2 = 0
    
    journal = []
    valeur_portefeuille = []
    valeur_buy_hold = []
    
    # Simulation
    for i in range(len(merged)):
        date = merged.index[i]
        prix_asset1 = merged[f'Close_{ticker1}'].iloc[i]
        prix_asset2 = merged[f'Close_{ticker2}'].iloc[i]
        rsi_asset2 = merged[f'rsi_{ticker2}'].iloc[i]
        
        # Valeur asset1 (buy & hold)
        valeur_asset1 = nb_actions_asset1 * prix_asset1
        
        # Gestion de la position asset2 avec RSI
        if pd.notna(rsi_asset2):
            # Signal d'achat
            if rsi_asset2 <= rsi_buy and not asset2_position:
                nb_actions_asset2 = capital_asset2_cash / prix_asset2
                prix_achat_asset2 = prix_asset2
                asset2_position = True
                journal.append({
                    'Date': date,
                    'Action': f'ACHAT {ticker2}',
                    'Prix': prix_asset2,
                    'Quantité': nb_actions_asset2,
                    'RSI': rsi_asset2,
                    'Capital investi': capital_asset2_cash
                })
            
            # Signal de vente
            elif rsi_asset2 >= rsi_sell and asset2_position:
                valeur_vente = nb_actions_asset2 * prix_asset2
                profit = valeur_vente - capital_asset2_cash
                profit_pct = (profit / capital_asset2_cash) * 100
                capital_asset2_cash = valeur_vente
                journal.append({
                    'Date': date,
                    'Action': f'VENTE {ticker2}',
                    'Prix': prix_asset2,
                    'Quantité': nb_actions_asset2,
                    'RSI': rsi_asset2,
                    'Prix achat': prix_achat_asset2,
                    'Profit': profit,
                    'Profit %': profit_pct,
                    'Capital après vente': capital_asset2_cash
                })
                nb_actions_asset2 = 0
                asset2_position = False
        
        # Valeur actuelle asset2
        if asset2_position:
            valeur_asset2 = nb_actions_asset2 * prix_asset2
        else:
            valeur_asset2 = capital_asset2_cash
        
        # Valeur totale portefeuille stratégie
        valeur_totale = valeur_asset1 + valeur_asset2
        valeur_portefeuille.append(valeur_totale)
        
        # Valeur buy & hold
        valeur_a1_bh = (capital * weight1 / 100 / merged[f'Close_{ticker1}'].iloc[0]) * prix_asset1
        valeur_a2_bh = (capital * weight2 / 100 / merged[f'Close_{ticker2}'].iloc[0]) * prix_asset2
        valeur_buy_hold.append(valeur_a1_bh + valeur_a2_bh)
    
    # Ajouter au dataframe
    merged['valeur_strategie'] = valeur_portefeuille
    merged['valeur_buy_hold'] = valeur_buy_hold
    merged['pct_strategie'] = (merged['valeur_strategie'] / capital) * 100
    merged['pct_buy_hold'] = (merged['valeur_buy_hold'] / capital) * 100
    merged[f'pct_{ticker1}'] = (merged[f'Close_{ticker1}'] / merged[f'Close_{ticker1}'].iloc[0]) * 100
    merged[f'pct_{ticker2}'] = (merged[f'Close_{ticker2}'] / merged[f'Close_{ticker2}'].iloc[0]) * 100
    
    return merged, journal, (ticker1, ticker2)

# =============================================
# STRATÉGIE 2: COINTÉGRATION
# =============================================

def test_stationnarite(serie):
    """Test ADF de stationnarité"""
    result = adfuller(serie.dropna(), maxlag=1, regression='c')
    return result[1]  # p-value

def test_integration(serie):
    """Teste si la série est I(1)"""
    # Test niveau
    p_niveau = test_stationnarite(serie)
    if p_niveau < 0.05:
        return 0  # Stationnaire en niveau (I(0))
    
    # Test première différence
    diff_serie = serie.diff().dropna()
    p_diff = test_stationnarite(diff_serie)
    if p_diff < 0.05:
        return 1  # I(1)
    
    return -1  # Ni I(0) ni I(1)

def run_backtest_cointegration(ticker1, ticker2, capital, start_date, end_date, seuil_residus):
    """Exécute le backtest de la stratégie de cointégration"""
    
    # Télécharger les données
    data1 = get_historical_data(ticker1, start_date, end_date)
    data2 = get_historical_data(ticker2, start_date, end_date)
    
    if data1 is None or data2 is None or len(data1) == 0 or len(data2) == 0:
        return None, None, None, None
    
    # Nettoyer et fusionner
    df1 = nettoyer_donnees(data1[['Close']])
    df2 = nettoyer_donnees(data2[['Close']])
    
    df1.columns = [ticker1]
    df2.columns = [ticker2]
    
    df = pd.merge(df1, df2, left_index=True, right_index=True, how="inner")
    
    if len(df) < 100:
        st.error(f"Pas assez de données communes ({len(df)} observations)")
        return None, None, None, None
    
    # Test d'intégration
    ordre1 = test_integration(df[ticker1])
    ordre2 = test_integration(df[ticker2])
    
    test_results = {
        'ordre1': ordre1,
        'ordre2': ordre2,
        'cointegre': False,
        'adf_residus': None,
        'p_value_residus': None
    }
    
    if ordre1 != 1 or ordre2 != 1:
        st.warning(f"⚠️ Les séries ne sont pas I(1) ({ticker1}: I({ordre1}), {ticker2}: I({ordre2}))")
        return None, None, None, test_results
    
    # Régression de cointégration
    X = sm.add_constant(df[ticker1])
    model = sm.OLS(df[ticker2], X).fit()
    df["residuals"] = model.resid
    
    # Test de cointégration
    adf_res = adfuller(df["residuals"])
    test_results['adf_residus'] = adf_res[0]
    test_results['p_value_residus'] = adf_res[1]
    test_results['cointegre'] = adf_res[1] < 0.05
    
    if not test_results['cointegre']:
        st.warning(f"⚠️ Les actifs ne sont pas cointégrés (p-value={adf_res[1]:.4f})")
    
    # Signaux de trading
    df["signal"] = 0
    df.loc[df["residuals"] > seuil_residus, "signal"] = -1  # Short Y, Long X
    df.loc[df["residuals"] < -seuil_residus, "signal"] = 1   # Long Y, Short X
    
    # Backtest
    journal = []
    capital_evolution = []
    position = 0
    entry_price_x = entry_price_y = None
    
    for i in range(1, len(df)):
        res = df["residuals"].iloc[i]
        date = df.index[i]
        px_x = df[ticker1].iloc[i]
        px_y = df[ticker2].iloc[i]
        
        # Ouvrir position
        if position == 0:
            if df["signal"].iloc[i] == 1:  # Long Y, Short X
                entry_price_y = px_y
                entry_price_x = px_x
                qty_y = (capital / 2) / entry_price_y
                qty_x = (capital / 2) / entry_price_x
                position = 1
                entry_date = date
            elif df["signal"].iloc[i] == -1:  # Short Y, Long X
                entry_price_y = px_y
                entry_price_x = px_x
                qty_y = (capital / 2) / entry_price_y
                qty_x = (capital / 2) / entry_price_x
                position = -1
                entry_date = date
        
        # Fermer position Long Y, Short X
        elif position == 1:
            if res >= 0:
                pnl_y = (px_y - entry_price_y) * qty_y
                pnl_x = (entry_price_x - px_x) * qty_x
                total_pnl = pnl_y + pnl_x
                capital += total_pnl
                duration = (date - entry_date).days
                journal.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Entry X': entry_price_x,
                    'Exit X': px_x,
                    'Entry Y': entry_price_y,
                    'Exit Y': px_y,
                    'PnL': total_pnl,
                    'Duration (days)': duration,
                    'Type': 'Long Y / Short X'
                })
                position = 0
        
        # Fermer position Short Y, Long X
        elif position == -1:
            if res <= 0:
                pnl_y = (entry_price_y - px_y) * qty_y
                pnl_x = (px_x - entry_price_x) * qty_x
                total_pnl = pnl_y + pnl_x
                capital += total_pnl
                duration = (date - entry_date).days
                journal.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Entry X': entry_price_x,
                    'Exit X': px_x,
                    'Entry Y': entry_price_y,
                    'Exit Y': px_y,
                    'PnL': total_pnl,
                    'Duration (days)': duration,
                    'Type': 'Short Y / Long X'
                })
                position = 0
        
        capital_evolution.append(capital)
    
    # Créer la série de capital
    df['capital'] = [capital] * len(df)
    if len(capital_evolution) > 0:
        df.iloc[-len(capital_evolution):, df.columns.get_loc('capital')] = capital_evolution
    
    return df, journal, test_results, (ticker1, ticker2)

# =============================================
# INTERFACE
# =============================================

st.markdown("### 🎯 SÉLECTION DE LA STRATÉGIE")

strategy = st.radio(
    "Choisissez votre stratégie de trading:",
    options=["RSI (Momentum)", "Cointégration (Pairs Trading)"],
    horizontal=True
)

# Afficher la description de la stratégie
if strategy == "RSI (Momentum)":
    st.markdown("""
    <div class="strategy-info">
    <b>📈 STRATÉGIE RSI (MOMENTUM)</b><br>
    • Asset 1 : Buy & Hold (position longue maintenue)<br>
    • Asset 2 : Trading actif basé sur le RSI<br>
    • Achat quand RSI ≤ seuil bas (survente)<br>
    • Vente quand RSI ≥ seuil haut (surachat)<br>
    • Idéal pour: Actifs volatils avec tendances claires
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="strategy-info">
    <b>🔄 STRATÉGIE COINTÉGRATION (PAIRS TRADING)</b><br>
    • Exploite la relation statistique entre 2 actifs cointégrés<br>
    • Long/Short basé sur les résidus de régression<br>
    • Achat quand résidus < -seuil (sous-valorisé)<br>
    • Vente quand résidus > +seuil (survalorié)<br>
    • Idéal pour: Actifs du même secteur (ex: banques, pétrole)
    </div>
    """, unsafe_allow_html=True)

st.markdown("### ⚙️ CONFIGURATION")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ACTIFS")
    ticker1 = st.text_input("Ticker 1", value="QQQ" if strategy == "RSI (Momentum)" else "MS", help="Symbole Yahoo Finance")
    ticker2 = st.text_input("Ticker 2", value="VIXM" if strategy == "RSI (Momentum)" else "BAC", help="Symbole Yahoo Finance")

with col2:
    if strategy == "RSI (Momentum)":
        st.markdown("#### ALLOCATION")
        weight1 = st.slider("Poids Ticker 1 (%)", 0, 100, 60, 5)
        weight2 = 100 - weight1
        st.metric("Poids Ticker 2 (%)", f"{weight2}%")
    else:
        st.markdown("#### ALLOCATION")
        st.info("Cointégration: 50/50 automatique")
        weight1 = weight2 = 50
    
    capital = st.number_input("Capital Initial ($)", min_value=1000, value=10000, step=1000)

with col3:
    if strategy == "RSI (Momentum)":
        st.markdown("#### PARAMÈTRES RSI")
        rsi_buy = st.slider("RSI Achat", 0, 50, 20, 1)
        rsi_sell = st.slider("RSI Vente", 50, 100, 80, 1)
        rsi_period = st.number_input("Période RSI", min_value=5, max_value=30, value=14, step=1)
    else:
        st.markdown("#### PARAMÈTRES COINTÉGRATION")
        seuil_residus = st.slider("Seuil Résidus (±)", 0.5, 10.0, 5.0, 0.5, 
                                  help="Écart-type pour déclencher les signaux")
        st.caption("Plus le seuil est élevé, moins il y a de trades")

st.markdown("#### PÉRIODE")
col_date1, col_date2 = st.columns(2)

with col_date1:
    start_date = st.date_input(
        "Date de début",
        value=datetime.now() - timedelta(days=365 if strategy == "Cointégration (Pairs Trading)" else 3*365),
        max_value=datetime.now()
    )

with col_date2:
    end_date = st.date_input(
        "Date de fin",
        value=datetime.now(),
        max_value=datetime.now()
    )

st.markdown('<hr>', unsafe_allow_html=True)

# Bouton de lancement
if st.button("🚀 LANCER LE BACKTEST", use_container_width=True):
    
    if start_date >= end_date:
        st.error("⚠️ La date de début doit être avant la date de fin")
        st.stop()
    
    # =============================================
    # EXÉCUTION STRATÉGIE RSI
    # =============================================
    if strategy == "RSI (Momentum)":
        with st.spinner(f"📊 Téléchargement et analyse de {ticker1} et {ticker2}..."):
            merged, journal, tickers = run_backtest_rsi(
                ticker1, ticker2, weight1, weight2, capital,
                start_date, end_date, rsi_buy, rsi_sell
            )
        
        if merged is None:
            st.error("Erreur lors du backtest. Vérifiez les tickers et les dates.")
            st.stop()
        
        st.success(f"✅ Backtest RSI terminé ! Période: {merged.index[0].strftime('%Y-%m-%d')} → {merged.index[-1].strftime('%Y-%m-%d')}")
        
        # STATISTIQUES
        st.markdown("### 📊 STATISTIQUES DE PERFORMANCE")
        
        valeur_finale_bh = merged['valeur_buy_hold'].iloc[-1]
        perf_bh = ((valeur_finale_bh / capital) - 1) * 100
        
        valeur_finale_strat = merged['valeur_strategie'].iloc[-1]
        perf_strat = ((valeur_finale_strat / capital) - 1) * 100
        
        diff_perf = perf_strat - perf_bh
        diff_valeur = valeur_finale_strat - valeur_finale_bh
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Buy & Hold", f"${valeur_finale_bh:,.2f}", f"{perf_bh:+.2f}%")
        
        with col_stat2:
            st.metric("Stratégie RSI", f"${valeur_finale_strat:,.2f}", f"{perf_strat:+.2f}%")
        
        with col_stat3:
            st.metric("Différence", f"${diff_valeur:+,.2f}", f"{diff_perf:+.2f}%")
        
        with col_stat4:
            nb_trades = len([j for j in journal if 'VENTE' in j['Action']])
            st.metric("Trades Complétés", f"{nb_trades}")
        
        # GRAPHIQUES RSI
        st.markdown("### 📈 GRAPHIQUES D'ANALYSE")
        
        # Graph 1: Ticker 1
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.7, 0.3], subplot_titles=(f'{ticker1} - Prix', 'RSI'))
        
        fig1.add_trace(go.Scatter(x=merged.index, y=merged[f'Close_{ticker1}'], 
                                 name=ticker1, line=dict(color='blue', width=2)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=merged.index, y=merged[f'rsi_{ticker1}'], 
                                 name='RSI', line=dict(color='blue', width=2)), row=2, col=1)
        fig1.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig1.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        fig1.update_layout(height=500, paper_bgcolor='#000', plot_bgcolor='#111',
                          font=dict(color='#FFAA00', size=10), hovermode='x unified')
        fig1.update_xaxes(gridcolor='#333', showgrid=True)
        fig1.update_yaxes(gridcolor='#333', showgrid=True)
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graph 2: Ticker 2 avec signaux
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.7, 0.3], subplot_titles=(f'{ticker2} - Prix et Signaux', 'RSI'))
        
        fig2.add_trace(go.Scatter(x=merged.index, y=merged[f'Close_{ticker2}'], 
                                 name=ticker2, line=dict(color='red', width=2)), row=1, col=1)
        
        achats = [j for j in journal if 'ACHAT' in j['Action']]
        ventes = [j for j in journal if 'VENTE' in j['Action']]
        
        if achats:
            fig2.add_trace(go.Scatter(x=[j['Date'] for j in achats], y=[j['Prix'] for j in achats],
                                     mode='markers', name='Achat', 
                                     marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        
        if ventes:
            fig2.add_trace(go.Scatter(x=[j['Date'] for j in ventes], y=[j['Prix'] for j in ventes],
                                     mode='markers', name='Vente', 
                                     marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
        
        fig2.add_trace(go.Scatter(x=merged.index, y=merged[f'rsi_{ticker2}'], 
                                 name='RSI', line=dict(color='red', width=2)), row=2, col=1)
        fig2.add_hline(y=rsi_sell, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig2.add_hline(y=rsi_buy, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
        
        fig2.update_layout(height=500, paper_bgcolor='#000', plot_bgcolor='#111',
                          font=dict(color='#FFAA00', size=10), hovermode='x unified')
        fig2.update_xaxes(gridcolor='#333', showgrid=True)
        fig2.update_yaxes(gridcolor='#333', showgrid=True)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Graph 3: Comparaison
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=merged.index, y=merged['pct_buy_hold'],
                                 name='Buy&Hold', line=dict(color='orange', width=2)))
        fig3.add_trace(go.Scatter(x=merged.index, y=merged['pct_strategie'],
                                 name='Stratégie RSI', line=dict(color='purple', width=3)))
        fig3.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig3.update_layout(title="Comparaison Performance", height=400,
                          paper_bgcolor='#000', plot_bgcolor='#111',
                          font=dict(color='#FFAA00', size=10), hovermode='x unified',
                          xaxis=dict(gridcolor='#333', showgrid=True),
                          yaxis=dict(gridcolor='#333', showgrid=True, title='Evolution (%)'))
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Journal
        if journal:
            st.markdown("### 📋 JOURNAL DE TRADING")
            trades_vente = [t for t in journal if 'VENTE' in t['Action']]
            if trades_vente:
                profits = [t['Profit %'] for t in trades_vente]
                nb_gagnants = sum(1 for p in profits if p > 0)
                nb_perdants = sum(1 for p in profits if p < 0)
                
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                with col_t1:
                    st.metric("Trades Gagnants", f"{nb_gagnants}", f"{nb_gagnants/len(trades_vente)*100:.1f}%")
                with col_t2:
                    st.metric("Trades Perdants", f"{nb_perdants}", f"{nb_perdants/len(trades_vente)*100:.1f}%")
                with col_t3:
                    st.metric("Profit Moyen", f"{np.mean(profits):+.2f}%")
                with col_t4:
                    st.metric("Meilleur Trade", f"{max(profits):+.2f}%")
            
            journal_df = pd.DataFrame(journal)
            journal_df['Date'] = journal_df['Date'].dt.strftime('%Y-%m-%d')
            for col in ['Prix', 'Quantité', 'RSI', 'Capital investi', 'Prix achat', 'Profit', 'Profit %', 'Capital après vente']:
                if col in journal_df.columns:
                    journal_df[col] = journal_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
            
            st.dataframe(journal_df, use_container_width=True, height=300)
    
    # =============================================
    # EXÉCUTION STRATÉGIE COINTÉGRATION
    # =============================================
    else:
        with st.spinner(f"📊 Test de cointégration entre {ticker1} et {ticker2}..."):
            df, journal, test_results, tickers = run_backtest_cointegration(
                ticker1, ticker2, capital, start_date, end_date, seuil_residus
            )
        
        if df is None:
            st.error("Erreur lors du backtest de cointégration.")
            st.stop()
        
        # Résultats des tests
        st.markdown("### 🔬 TESTS STATISTIQUES")
        
        col_test1, col_test2, col_test3 = st.columns(3)
        
        with col_test1:
            st.metric(f"{ticker1} Ordre d'intégration", 
                     f"I({test_results['ordre1']})" if test_results['ordre1'] >= 0 else "Non I(0) ni I(1)",
                     "✅" if test_results['ordre1'] == 1 else "❌")
        
        with col_test2:
            st.metric(f"{ticker2} Ordre d'intégration", 
                     f"I({test_results['ordre2']})" if test_results['ordre2'] >= 0 else "Non I(0) ni I(1)",
                     "✅" if test_results['ordre2'] == 1 else "❌")
        
        with col_test3:
            if test_results['cointegre'] is not None:
                st.metric("Cointégration", 
                         "✅ Cointégrés" if test_results['cointegre'] else "❌ Non cointégrés",
                         f"p={test_results['p_value_residus']:.4f}" if test_results['p_value_residus'] else "")
        
        if not test_results['cointegre']:
            st.warning("⚠️ Les actifs ne sont pas statistiquement cointégrés. Les résultats du backtest peuvent être peu fiables.")
        
        # STATISTIQUES
        st.markdown("### 📊 STATISTIQUES DE PERFORMANCE")
        
        valeur_finale = df['capital'].iloc[-1]
        perf = ((valeur_finale / capital) - 1) * 100
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Capital Initial", f"${capital:,.2f}")
        
        with col_stat2:
            st.metric("Capital Final", f"${valeur_finale:,.2f}", f"{perf:+.2f}%")
        
        with col_stat3:
            st.metric("Nombre de Trades", f"{len(journal)}")
        
        # GRAPHIQUES COINTÉGRATION
        st.markdown("### 📈 GRAPHIQUES D'ANALYSE")
        
        # Graph 1: Prix normalisés
        fig1 = go.Figure()
        df_pct = df[[ticker1, ticker2]] / df[[ticker1, ticker2]].iloc[0] * 100
        fig1.add_trace(go.Scatter(x=df_pct.index, y=df_pct[ticker1], 
                                 name=ticker1, line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=df_pct.index, y=df_pct[ticker2], 
                                 name=ticker2, line=dict(color='orange', width=2)))
        
        fig1.update_layout(title="Évolution des prix normalisés (%)", height=400,
                          paper_bgcolor='#000', plot_bgcolor='#111',
                          font=dict(color='#FFAA00', size=10), hovermode='x unified',
                          xaxis=dict(gridcolor='#333', showgrid=True),
                          yaxis=dict(gridcolor='#333', showgrid=True))
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graph 2: Résidus avec signaux
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['residuals'], 
                                 name='Résidus', line=dict(color='blue', width=2)))
        fig2.add_hline(y=0, line_dash="solid", line_color="red", opacity=0.5)
        fig2.add_hline(y=seuil_residus, line_dash="dash", line_color="green", opacity=0.7)
        fig2.add_hline(y=-seuil_residus, line_dash="dash", line_color="green", opacity=0.7)
        
        # Ajouter les zones de signaux
        fig2.add_hrect(y0=seuil_residus, y1=df['residuals'].max(), fillcolor="red", opacity=0.1)
        fig2.add_hrect(y0=df['residuals'].min(), y1=-seuil_residus, fillcolor="green", opacity=0.1)
        
        fig2.update_layout(title="Résidus de la régression de cointégration", height=400,
                          paper_bgcolor='#000', plot_bgcolor='#111',
                          font=dict(color='#FFAA00', size=10), hovermode='x unified',
                          xaxis=dict(gridcolor='#333', showgrid=True),
                          yaxis=dict(gridcolor='#333', showgrid=True, title='Résidus'))
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Graph 3: Évolution du capital
        if len(journal) > 0:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df.index, y=df['capital'], 
                                     name='Capital', line=dict(color='purple', width=3)))
            fig3.add_hline(y=capital, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig3.update_layout(title="Évolution du capital", height=400,
                              paper_bgcolor='#000', plot_bgcolor='#111',
                              font=dict(color='#FFAA00', size=10), hovermode='x unified',
                              xaxis=dict(gridcolor='#333', showgrid=True),
                              yaxis=dict(gridcolor='#333', showgrid=True, title='Capital ($)'))
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Journal de trading
        if journal:
            st.markdown("### 📋 JOURNAL DE TRADING")
            
            # Stats des trades
            pnls = [t['PnL'] for t in journal]
            nb_gagnants = sum(1 for p in pnls if p > 0)
            nb_perdants = sum(1 for p in pnls if p < 0)
            
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            
            with col_t1:
                st.metric("Trades Gagnants", f"{nb_gagnants}", f"{nb_gagnants/len(journal)*100:.1f}%")
            
            with col_t2:
                st.metric("Trades Perdants", f"{nb_perdants}", f"{nb_perdants/len(journal)*100:.1f}%")
            
            with col_t3:
                st.metric("PnL Moyen", f"${np.mean(pnls):,.2f}")
            
            with col_t4:
                st.metric("Meilleur Trade", f"${max(pnls):,.2f}")
            
            # Tableau
            journal_df = pd.DataFrame(journal)
            journal_df['Entry Date'] = pd.to_datetime(journal_df['Entry Date']).dt.strftime('%Y-%m-%d')
            journal_df['Exit Date'] = pd.to_datetime(journal_df['Exit Date']).dt.strftime('%Y-%m-%d')
            
            for col in ['Entry X', 'Exit X', 'Entry Y', 'Exit Y', 'PnL']:
                journal_df[col] = journal_df[col].apply(lambda x: f"{x:,.2f}")
            
            st.dataframe(journal_df, use_container_width=True, height=300)
        else:
            st.info("Aucun trade effectué avec les paramètres actuels. Essayez de réduire le seuil des résidus.")

# =============================================
# FOOTER
# =============================================
add_footer_ad()

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | BACKTEST ENGINE v2.0 | RSI + COINTEGRATION STRATEGIES<br>
    SYSTÈME OPÉRATIONNEL • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
