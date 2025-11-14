import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="Bloomberg - Options Pricing",
    page_icon="📊",
    layout="wide"
)

# CSS Style Bloomberg
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background-color: #000000;
        color: #FFAA00;
        font-family: 'Courier New', monospace;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    .bloomberg-header {
        background: linear-gradient(90deg, #FFAA00 0%, #FF8C00 100%);
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 16px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    h1, h2, h3 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
        font-size: 13px !important;
        font-weight: bold !important;
        letter-spacing: 1px !important;
    }
    
    .parameter-group {
        background: #111;
        border: 1px solid #333;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .group-title {
        color: #FFAA00;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 10px;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    
    label {
        color: #FFAA00 !important;
        font-size: 11px !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stNumberInput input, .stSlider {
        background-color: #000 !important;
        color: #FFAA00 !important;
        border: 1px solid #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stSelectbox select {
        background-color: #000 !important;
        color: #FFAA00 !important;
        border: 1px solid #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        border: 2px solid #FFAA00;
        font-weight: bold;
        text-transform: uppercase;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        padding: 10px 20px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    .greek-card {
        background: #111;
        border-left: 4px solid #FFAA00;
        padding: 12px;
        margin: 8px 0;
    }
    
    .greek-label {
        color: #FFAA00;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .greek-value {
        color: #FFF;
        font-size: 20px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    
    .block-container {
        padding-top: 0rem;
    }
    
    /* Horloge */
    .live-clock {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
</style>

<script>
    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        const timeString = hours + ':' + minutes + ':' + seconds + ' PARIS';
        
        const clockElements = document.querySelectorAll('.live-clock');
        clockElements.forEach(el => {
            el.textContent = timeString;
        });
    }
    
    setInterval(updateClock, 1000);
    updateClock();
</script>
""", unsafe_allow_html=True)

# Fonctions Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type='Call'):
    """Calcul du prix d'une option selon Black-Scholes"""
    if T <= 0:
        T = 0.001
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_greeks(S, K, T, r, sigma, option_type='Call'):
    """Calcul des Greeks"""
    if T <= 0:
        T = 0.001
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'Call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    # Theta
    if option_type == 'Call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    # Rho
    if option_type == 'Call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    # Prix
    price = black_scholes(S, K, T, r, sigma, option_type)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega / 100,  # Vega par 1% de volatilité
        'theta': theta / 365,  # Theta par jour
        'rho': rho / 100  # Rho par 1% de taux
    }

# Header
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® | OPTIONS PRICING</div>
    <div class="live-clock">{current_time.strftime("%H:%M:%S")} PARIS</div>
</div>
''', unsafe_allow_html=True)

# Bouton retour
if st.button("← RETOUR AU TERMINAL", key="back_btn"):
    st.switch_page("streamlit_app.py")

st.markdown("---")

# Layout principal
col_params, col_results = st.columns([1, 2])

with col_params:
    st.markdown("### ⚙️ OPTIONS PARAMETERS")
    
    # Market Data
    st.markdown('<div class="parameter-group">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">Market Data</div>', unsafe_allow_html=True)
    
    S = st.number_input("Underlying Price (S)", min_value=10.0, max_value=500.0, value=100.0, step=1.0)
    S_slider = st.slider("", min_value=10.0, max_value=500.0, value=S, step=1.0, key="S_slider", label_visibility="collapsed")
    
    K = st.number_input("Strike Price (K)", min_value=10.0, max_value=500.0, value=100.0, step=1.0)
    K_slider = st.slider("", min_value=10.0, max_value=500.0, value=K, step=1.0, key="K_slider", label_visibility="collapsed")
    
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
    T_slider = st.slider("", min_value=0.01, max_value=5.0, value=T, step=0.1, key="T_slider", label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Parameters
    st.markdown('<div class="parameter-group">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">Risk Parameters</div>', unsafe_allow_html=True)
    
    r = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    r_slider = st.slider("", min_value=0.0, max_value=20.0, value=r, step=0.1, key="r_slider", label_visibility="collapsed")
    
    sigma = st.number_input("Volatility σ (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    sigma_slider = st.slider("", min_value=1.0, max_value=100.0, value=sigma, step=1.0, key="sigma_slider", label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Option Type
    st.markdown('<div class="parameter-group">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">Option Type</div>', unsafe_allow_html=True)
    
    option_type = st.selectbox("Call / Put", ['Call', 'Put'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calcul
    if st.button("📊 CALCULATE GREEKS", key="calc_btn"):
        st.session_state.calculate = True

with col_results:
    st.markdown("### 📈 RESULTS & GREEKS")
    
    # Calcul automatique
    S_val = S_slider if S_slider != S else S
    K_val = K_slider if K_slider != K else K
    T_val = T_slider if T_slider != T else T
    r_val = (r_slider if r_slider != r else r) / 100
    sigma_val = (sigma_slider if sigma_slider != sigma else sigma) / 100
    
    greeks = calculate_greeks(S_val, K_val, T_val, r_val, sigma_val, option_type)
    
    # Affichage des Greeks
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-label">Option Price</div>
            <div class="greek-value">${greeks['price']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-label">Delta (Δ)</div>
            <div class="greek-value">{greeks['delta']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-label">Gamma (Γ)</div>
            <div class="greek-value">{greeks['gamma']:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-label">Vega (ν)</div>
            <div class="greek-value">{greeks['vega']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-label">Theta (Θ)</div>
            <div class="greek-value">{greeks['theta']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="greek-card">
            <div class="greek-label">Rho (ρ)</div>
            <div class="greek-value">{greeks['rho']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques
    st.markdown("### 📊 SENSITIVITY ANALYSIS")
    
    # Delta vs Underlying Price
    S_range = np.linspace(max(10, S_val * 0.5), S_val * 1.5, 100)
    deltas = [calculate_greeks(s, K_val, T_val, r_val, sigma_val, option_type)['delta'] for s in S_range]
    
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(
        x=S_range, y=deltas,
        mode='lines',
        line=dict(color='#00FFFF', width=2),
        name='Delta'
    ))
    fig_delta.add_trace(go.Scatter(
        x=[S_val], y=[greeks['delta']],
        mode='markers',
        marker=dict(color='#FF0000', size=10),
        name='Current'
    ))
    fig_delta.update_layout(
        title="Delta vs Underlying Price",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', family='Courier New', size=10),
        xaxis=dict(gridcolor='#333', title="Underlying Price"),
        yaxis=dict(gridcolor='#333', title="Delta"),
        height=300
    )
    st.plotly_chart(fig_delta, use_container_width=True)
    
    # Gamma vs Underlying Price
    gammas = [calculate_greeks(s, K_val, T_val, r_val, sigma_val, option_type)['gamma'] for s in S_range]
    
    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(
        x=S_range, y=gammas,
        mode='lines',
        line=dict(color='#FF00FF', width=2),
        name='Gamma'
    ))
    fig_gamma.add_trace(go.Scatter(
        x=[S_val], y=[greeks['gamma']],
        mode='markers',
        marker=dict(color='#FF0000', size=10),
        name='Current'
    ))
    fig_gamma.update_layout(
        title="Gamma vs Underlying Price",
        paper_bgcolor='#000',
        plot_bgcolor='#111',
        font=dict(color='#FFAA00', family='Courier New', size=10),
        xaxis=dict(gridcolor='#333', title="Underlying Price"),
        yaxis=dict(gridcolor='#333', title="Gamma"),
        height=300
    )
    st.plotly_chart(fig_gamma, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 10px; font-family: "Courier New", monospace;'>
    © 2025 BLOOMBERG ENS® | BLACK-SCHOLES MODEL
</div>
""", unsafe_allow_html=True)
