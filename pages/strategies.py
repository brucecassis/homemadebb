import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

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
# FONCTIONS
# =============================================

@st.cache_data(ttl=3600)
def get_historical_data(ticker, start_date, end_date):
    """Récupère les données historiques d'un ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval='1d')
        return hist
    except Exception as e:
        st.error(f"Erreur lors du téléchargement de {ticker}: {str(e)}")
        return None

def calculate_rsi(prices, period=14):
    """Calcule le RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_backtest(ticker1, ticker2, weight1, weight2, capital, start_date, end_date, rsi_buy, rsi_sell):
    """Exécute le backtest de la stratégie"""
    
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
# INTERFACE DE CONFIGURATION
# =============================================

st.markdown("### ⚙️ CONFIGURATION DE LA STRATÉGIE")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ACTIFS")
    ticker1 = st.text_input("Ticker 1 (Buy & Hold)", value="QQQ", help="Symbole Yahoo Finance")
    ticker2 = st.text_input("Ticker 2 (Stratégie RSI)", value="VIXM", help="Symbole Yahoo Finance")

with col2:
    st.markdown("#### ALLOCATION")
    weight1 = st.slider("Poids Ticker 1 (%)", 0, 100, 60, 5)
    weight2 = 100 - weight1
    st.metric("Poids Ticker 2 (%)", f"{weight2}%")
    
    capital = st.number_input("Capital Initial ($)", min_value=1000, value=10000, step=1000)

with col3:
    st.markdown("#### STRATÉGIE RSI")
    rsi_buy = st.slider("RSI Achat", 0, 50, 20, 1, help="Acheter quand RSI <= cette valeur")
    rsi_sell = st.slider("RSI Vente", 50, 100, 80, 1, help="Vendre quand RSI >= cette valeur")
    
    rsi_period = st.number_input("Période RSI", min_value=5, max_value=30, value=14, step=1)

st.markdown("#### PÉRIODE")
col_date1, col_date2 = st.columns(2)

with col_date1:
    start_date = st.date_input(
        "Date de début",
        value=datetime.now() - timedelta(days=3*365),
        max_value=datetime.now()
    )

with col_date2:
    end_date = st.date_input(
        "Date de fin",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Bouton de lancement
st.markdown('<hr>', unsafe_allow_html=True)

if st.button("🚀 LANCER LE BACKTEST", use_container_width=True):
    
    if weight1 + weight2 != 100:
        st.error("⚠️ La somme des poids doit faire 100%")
        st.stop()
    
    if start_date >= end_date:
        st.error("⚠️ La date de début doit être avant la date de fin")
        st.stop()
    
    with st.spinner(f"📊 Téléchargement et analyse de {ticker1} et {ticker2}..."):
        merged, journal, tickers = run_backtest(
            ticker1, ticker2, weight1, weight2, capital,
            start_date, end_date, rsi_buy, rsi_sell
        )
    
    if merged is None:
        st.error("Erreur lors du backtest. Vérifiez les tickers et les dates.")
        st.stop()
    
    # =============================================
    # AFFICHAGE DES RÉSULTATS
    # =============================================
    
    st.success(f"✅ Backtest terminé ! Période: {merged.index[0].strftime('%Y-%m-%d')} → {merged.index[-1].strftime('%Y-%m-%d')}")
    
    # STATISTIQUES PRINCIPALES
    st.markdown("### 📊 STATISTIQUES DE PERFORMANCE")
    
    valeur_finale_bh = merged['valeur_buy_hold'].iloc[-1]
    perf_bh = ((valeur_finale_bh / capital) - 1) * 100
    
    valeur_finale_strat = merged['valeur_strategie'].iloc[-1]
    perf_strat = ((valeur_finale_strat / capital) - 1) * 100
    
    diff_perf = perf_strat - perf_bh
    diff_valeur = valeur_finale_strat - valeur_finale_bh
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric(
            "Buy & Hold",
            f"${valeur_finale_bh:,.2f}",
            f"{perf_bh:+.2f}%"
        )
    
    with col_stat2:
        st.metric(
            "Stratégie RSI",
            f"${valeur_finale_strat:,.2f}",
            f"{perf_strat:+.2f}%"
        )
    
    with col_stat3:
        st.metric(
            "Différence",
            f"${diff_valeur:+,.2f}",
            f"{diff_perf:+.2f}%"
        )
    
    with col_stat4:
        nb_trades = len([j for j in journal if 'VENTE' in j['Action']])
        st.metric(
            "Trades Complétés",
            f"{nb_trades}"
        )
    
    # GRAPHIQUES
    st.markdown("### 📈 GRAPHIQUES D'ANALYSE")
    
    # Graphique 1: Prix et RSI Ticker 1
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker1} - Prix (Buy & Hold)', 'RSI')
    )
    
    fig1.add_trace(
        go.Scatter(x=merged.index, y=merged[f'Close_{ticker1}'], 
                   name=ticker1, line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(x=merged.index, y=merged[f'rsi_{ticker1}'], 
                   name='RSI', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    fig1.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig1.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    fig1.update_layout(
        height=500,
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        showlegend=True,
        hovermode='x unified'
    )
    
    fig1.update_xaxes(gridcolor='#333', showgrid=True)
    fig1.update_yaxes(gridcolor='#333', showgrid=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Graphique 2: Prix et RSI Ticker 2 avec signaux
    fig2 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker2} - Prix et Signaux (Stratégie RSI)', 'RSI')
    )
    
    fig2.add_trace(
        go.Scatter(x=merged.index, y=merged[f'Close_{ticker2}'], 
                   name=ticker2, line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Ajouter les signaux d'achat/vente
    achats = [j for j in journal if 'ACHAT' in j['Action']]
    ventes = [j for j in journal if 'VENTE' in j['Action']]
    
    if achats:
        dates_achat = [j['Date'] for j in achats]
        prix_achat = [j['Prix'] for j in achats]
        fig2.add_trace(
            go.Scatter(x=dates_achat, y=prix_achat, mode='markers',
                       name='Achat', marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
    
    if ventes:
        dates_vente = [j['Date'] for j in ventes]
        prix_vente = [j['Prix'] for j in ventes]
        fig2.add_trace(
            go.Scatter(x=dates_vente, y=prix_vente, mode='markers',
                       name='Vente', marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )
    
    fig2.add_trace(
        go.Scatter(x=merged.index, y=merged[f'rsi_{ticker2}'], 
                   name='RSI', line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    fig2.add_hline(y=rsi_sell, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig2.add_hline(y=rsi_buy, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    
    fig2.update_layout(
        height=500,
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        showlegend=True,
        hovermode='x unified'
    )
    
    fig2.update_xaxes(gridcolor='#333', showgrid=True)
    fig2.update_yaxes(gridcolor='#333', showgrid=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Graphique 3: Évolution en pourcentage
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=merged.index, y=merged[f'pct_{ticker1}'],
        name=f'{ticker1} seul', line=dict(color='blue', width=1.5, dash='dot'), opacity=0.7
    ))
    
    fig3.add_trace(go.Scatter(
        x=merged.index, y=merged[f'pct_{ticker2}'],
        name=f'{ticker2} seul', line=dict(color='red', width=1.5, dash='dot'), opacity=0.7
    ))
    
    fig3.add_trace(go.Scatter(
        x=merged.index, y=merged['pct_buy_hold'],
        name=f'Buy&Hold ({weight1}% {ticker1} / {weight2}% {ticker2})',
        line=dict(color='orange', width=2)
    ))
    
    fig3.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig3.update_layout(
        title="Évolution en % - Actifs individuels et Buy & Hold",
        height=400,
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(gridcolor='#333', showgrid=True, title='Date'),
        yaxis=dict(gridcolor='#333', showgrid=True, title='Évolution (%)')
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Graphique 4: Comparaison Stratégie vs Buy & Hold
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatter(
        x=merged.index, y=merged['pct_buy_hold'],
        name=f'Buy&Hold ({weight1}% {ticker1} / {weight2}% {ticker2})',
        line=dict(color='orange', width=2), opacity=0.7
    ))
    
    fig4.add_trace(go.Scatter(
        x=merged.index, y=merged['pct_strategie'],
        name=f'Stratégie ({ticker1} hold + {ticker2} RSI {rsi_buy}/{rsi_sell})',
        line=dict(color='purple', width=3)
    ))
    
    # Zone de surperformance
    fig4.add_trace(go.Scatter(
        x=merged.index, y=merged['pct_strategie'],
        fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    fig4.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig4.update_layout(
        title="Comparaison: Stratégie de Trading vs Buy & Hold",
        height=400,
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', size=10),
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(gridcolor='#333', showgrid=True, title='Date'),
        yaxis=dict(gridcolor='#333', showgrid=True, title='Évolution (%)')
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # JOURNAL DE TRADING
    if journal:
        st.markdown("### 📋 JOURNAL DE TRADING")
        
        # Statistiques des trades
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
        
        # Tableau des trades
        journal_df = pd.DataFrame(journal)
        journal_df['Date'] = journal_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Formater les colonnes numériques
        for col in ['Prix', 'Quantité', 'RSI', 'Capital investi', 'Prix achat', 'Profit', 'Profit %', 'Capital après vente']:
            if col in journal_df.columns:
                journal_df[col] = journal_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        
        st.dataframe(
            journal_df,
            use_container_width=True,
            height=300
        )

# =============================================
# FOOTER
# =============================================
add_footer_ad()

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | BACKTEST ENGINE v1.0 | YAHOO FINANCE DATA<br>
    SYSTÈME OPÉRATIONNEL • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
