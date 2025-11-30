import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time
import numpy as np

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Fixed Income",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# STYLE BLOOMBERG TERMINAL (COPIE EXACTE)
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
        min-height: 100vh !important;
    }
    
    h1, h2, h3, h4 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 14px !important;
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
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
    
    hr {
        border-color: #333333;
        margin: 8px 0;
    }
    
    p, div, span, label, input {
        font-family: 'Courier New', monospace !important;
        color: #FFAA00 !important;
    }

    /* Style spécifique pour les inputs du calculateur */
    .stNumberInput input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
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
# BARRE DE COMMANDE (Navigation)
# =============================================
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
<div class="command-container">
    <span class="command-prompt">BBG-RATES&gt;</span>
</div>
""", unsafe_allow_html=True)

# =============================================
# FONCTION DONNÉES TAUX
# =============================================
@st.cache_data(ttl=60)
def get_bond_data(ticker):
    """
    Récupère les données obligataires. 
    Pour les indices de taux Yahoo (^TNX, etc.), la valeur est le rendement x10.
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period='5d')
        
        if len(hist) < 2:
            return None, None
            
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        
        # Calcul de la variation en points de base (bps)
        change_bps = (current - prev)
        
        return current, change_bps
    except:
        return None, None

# =============================================
# HEADER
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® - FIXED INCOME DESK</div>
        <a href="accueil.html" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">MARKETS</a>
    </div>
    <div>{current_time} UTC • YIELD ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# 1. SOVEREIGN YIELDS (Dettes d'État)
# =============================================
st.markdown("### 🏛️ GOVERNMENT BOND YIELDS (TREASURIES & SOVEREIGNS)")

# Tickers Yahoo pour les taux US (Treasury Yields)
# Note: Yahoo donne ^TNX (10 ans) à 42.50 pour 4.25%
treasuries = {
    'US 3M BILL': '^IRX',
    'US 5Y NOTE': '^FVX',
    'US 10Y NOTE': '^TNX',
    'US 30Y BOND': '^TYX',
}

cols_bonds = st.columns(4)

yield_curve_data = {}

for idx, (name, ticker) in enumerate(treasuries.items()):
    with cols_bonds[idx]:
        val, change = get_bond_data(ticker)
        if val is not None:
            # Correction: Yahoo finance donne souvent les index de taux multipliés par 10 ou tel quel
            # Pour ^TNX, 42.50 = 4.25%
            # On assume ici que la donnée brute est le taux directement affichable en % pour l'affichage, 
            # mais attention aux échelles selon la source.
            
            yield_display = f"{val:.2f}%"
            yield_curve_data[name] = val
            
            # Couleur Delta: Rouge si les taux montent (mauvais pour prix), Vert si baissent
            st.metric(
                label=name,
                value=yield_display,
                delta=f"{change:+.2f} bps"
            )
        else:
            st.metric(label=name, value="N/A", delta="0")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# 2. COURBE DES TAUX (YIELD CURVE)
# =============================================
col_curve, col_spreads = st.columns([2, 1])

with col_curve:
    st.markdown("### 📈 US TREASURY YIELD CURVE")
    
    if len(yield_curve_data) > 0:
        fig_curve = go.Figure()
        
        # Données X et Y pour la courbe
        maturities = ['3M', '5Y', '10Y', '30Y']
        yields = [yield_curve_data.get('US 3M BILL', 0), 
                  yield_curve_data.get('US 5Y NOTE', 0), 
                  yield_curve_data.get('US 10Y NOTE', 0), 
                  yield_curve_data.get('US 30Y BOND', 0)]
        
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=yields,
            mode='lines+markers',
            name='US Yield Curve',
            line=dict(color='#FFAA00', width=3),
            marker=dict(size=10, color='#FFAA00')
        ))
        
        fig_curve.update_layout(
            title="Term Structure of Interest Rates",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', family='Courier New'),
            xaxis=dict(gridcolor='#333', title="Maturity"),
            yaxis=dict(gridcolor='#333', title="Yield (%)"),
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

with col_spreads:
    st.markdown("### ↔️ KEY SPREADS")
    
    # Calcul du 2s10s (Indicateur de récession)
    # Note: On utilise 3M ici faute de 2Y fiable dans la liste simple, ou on récupère 10Y - 3M
    if 'US 10Y NOTE' in yield_curve_data and 'US 3M BILL' in yield_curve_data:
        slope = yield_curve_data['US 10Y NOTE'] - yield_curve_data['US 3M BILL']
        
        st.markdown(f"""
        <div class="section-box" style="text-align:center;">
            <div style="font-size:12px;color:#888;">SLOPE (10Y - 3M)</div>
            <div style="font-size:24px;font-weight:bold;color:{'#00FF00' if slope > 0 else '#FF0000'};">
                {slope:.2f} bps
            </div>
            <div style="font-size:10px;margin-top:5px;">
                {'✅ NORMAL CURVE' if slope > 0 else '⚠️ INVERTED CURVE (RECESSION SIGNAL)'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Indicateur Volatilité taux (MOVE Index proxy via TLT volatility logic would be complex, simplified here)
    st.info("ℹ️ INFO: Une courbe inversée (rouge) précède souvent une récession économique.")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# 3. CORPORATE BONDS (ETFs PROXIES)
# =============================================
st.markdown("### 🏢 CORPORATE & CREDIT MARKETS (ETFs PROXIES)")

# On utilise des ETF liquides pour représenter le marché Corporate
credit_tickers = {
    'IG CORP (LQD)': 'LQD',   # Investment Grade
    'HIGH YIELD (HYG)': 'HYG', # Junk Bonds
    'EMERGING (EMB)': 'EMB',   # Emerging Markets
    'TOTAL BOND (BND)': 'BND', # Total Market
    'TIPS (TIP)': 'TIP'        # Inflation Protected
}

cols_credit = st.columns(5)

for idx, (name, ticker) in enumerate(credit_tickers.items()):
    with cols_credit[idx]:
        price, change_pct = get_bond_data(ticker) # Réutilise la fonction générique mais interprète comme prix
        
        # Pour les ETF, get_bond_data renvoie le prix, change est en valeur absolue, on veut %
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='2d')
            if len(hist) > 1:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                pct = ((curr - prev) / prev) * 100
                
                st.metric(
                    label=name,
                    value=f"${curr:.2f}",
                    delta=f"{pct:+.2f}%"
                )
            else:
                st.metric(label=name, value="LOAD...", delta="0%")
        except:
             st.metric(label=name, value="ERR", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# 4. CALCULATEUR D'OBLIGATION (Interactive)
# =============================================
st.markdown("### 🧮 BOND PRICING CALCULATOR (YIELD TO PRICE)")

col_calc_input, col_calc_res = st.columns([1, 1])

with col_calc_input:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    face_value = st.number_input("Face Value (Nominal)", value=1000.0, step=100.0)
    coupon_rate = st.number_input("Coupon Rate (%)", value=5.0, step=0.1)
    years = st.number_input("Years to Maturity", value=10.0, step=0.5)
    ytm = st.number_input("Yield to Maturity (YTM %)", value=4.5, step=0.1)
    frequency = st.selectbox("Frequency", options=[1, 2], format_func=lambda x: "Annual" if x==1 else "Semi-Annual")
    st.markdown('</div>', unsafe_allow_html=True)

def calculate_bond_price(face, coupon_pct, years, ytm_pct, freq):
    """Calcule le prix d'une obligation"""
    c = (coupon_pct / 100) * face / freq
    y = (ytm_pct / 100) / freq
    n = years * freq
    
    # PV of Coupons
    if y == 0:
        pv_coupons = c * n
        pv_face = face
    else:
        pv_coupons = c * ((1 - (1 + y) ** -n) / y)
        pv_face = face / ((1 + y) ** n)
        
    price = pv_coupons + pv_face
    
    duration = 0 # Simplified Duration calculation could go here
    return price

with col_calc_res:
    price = calculate_bond_price(face_value, coupon_rate, years, ytm, frequency)
    
    # Déterminer si Premium ou Discount
    status = "PAR"
    color = "#888"
    if price < face_value:
        status = "DISCOUNT (Price < Face)"
        color = "#00FF00" # Bon deal ? (dépend du point de vue)
    elif price > face_value:
        status = "PREMIUM (Price > Face)"
        color = "#FF0000"

    st.markdown(f"""
    <div style="padding: 20px; text-align: right;">
        <div style="font-size: 14px; color: #888; margin-bottom: 10px;">ESTIMATED PRICE</div>
        <div style="font-size: 48px; font-weight: bold; color: #FFAA00; font-family: 'Courier New';">
            ${price:,.2f}
        </div>
        <div style="font-size: 14px; font-weight: bold; color: {color}; margin-top: 10px; border-top: 1px solid #333; padding-top:10px;">
            {status}
        </div>
        <div style="font-size: 12px; color: #666; margin-top: 5px;">
            (Excl. Accrued Interest)
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® - FIXED INCOME | DATA DELAYED 15MIN | FOR EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
