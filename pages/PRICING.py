import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Options Pricing",
    page_icon="📊",
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
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
        font-size: 12px;
    }
    
    .main {
        background-color: #000000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 0rem 1rem !important;
    }
    
    .stApp {
        background-color: #000000;
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
    
    .stNumberInput > div > div > input {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stSelectbox > div > div {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
    
    .stSlider > div > div > div {
        background-color: #333 !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #FFAA00 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - OPTIONS PRICING</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">ACCUEIL</a>
    </div>
    <div>{current_time} UTC • BINOMIAL MODEL</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# BARRE DE COMMANDE BLOOMBERG
# À ajouter après le header, avant les données de marché
# =============================================

# Style pour la barre de commande
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
""", unsafe_allow_html=True)

# Dictionnaire des commandes et leurs pages
COMMANDS = {
    "EDGAR": "pages/EDGAR.py",
    "NEWS": "pages/NEWS.py",
    "PRICE": "pages/PRICING.py",
    "CHAT": "pages/CHATBOT.py",
    "BT": "pages/BACKTESTING.py",
    "ANA": "pages/COMPANY_ANALYSIS.py",
    "CRYPTO":"pages/CRYPTO_SCRAPER.py",
    "ECO":"pages/ECONOMICS.py", 
    "EU":"pages/EUROPE.py",
    "SIMU":"pages/PORTFOLIO_SIMU.py",
    "PY":"pages/PYTHON_EDITOR.py",
    "SQL":"pages/SQL_EDITOR.py",
    "BONDS":"pages/BONDS.py",
    "HOME":"pages/HOME.py",
}

# Affichage de la barre de commande
st.markdown('<div class="command-container">', unsafe_allow_html=True)

col_prompt, col_input = st.columns([1, 11])

with col_prompt:
    st.markdown('<span class="command-prompt">BBG&gt;</span>', unsafe_allow_html=True)

with col_input:
    command_input = st.text_input(
        "Command",
        placeholder="Tapez une commande: EDGAR, NEWS, CHATBOT, PRICING, HELP...",
        label_visibility="collapsed",
        key="bloomberg_command"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Traitement de la commande
if command_input:
    cmd = command_input.upper().strip()
    
    if cmd == "HELP" or cmd == "H":
        st.info("""
        **📋 COMMANDES DISPONIBLES:**
        - `EDGAR` → SEC Filings & Documents
        - `NEWS` → Market News Feed
        - `CHAT` → AI Assistant
        - `PRICE` → Options Pricing
        - `HELP` → Afficher cette aide
        - `BT` → Backesting de strategies
        - `ANA` → Analyse financière de sociétés côtées
        - `CRYPTO` → Scrapping et backtest de strategies liées aux cryptos
        - `ECO` → Données économiques
        - `EU` → Données Européennes
        - `SIMU` → Simulation de portefeuille
        - `PY` → Editeur de code python 
        - `SQL` → Editeur de code SQL
        - `BONDS` → Screener d'obligation
        - `HOME` → Menu
        """)
    elif cmd in COMMANDS:
        st.switch_page(COMMANDS[cmd])
    else:
        st.warning(f"⚠️ Commande '{cmd}' non reconnue. Tapez HELP pour voir les commandes disponibles.")


# =============================================
# FONCTIONS DE PRICING
# =============================================

@st.cache_data
def binomial_tree_american_option(S, K, T_days, r, sigma, N, option_type="put"):
    """
    Calcule le prix d'une option américaine par arbre binomial (CRR)
    Retourne: (prix, delta, gamma, theta, stock_tree, option_tree)
    """
    if T_days <= 0:
        intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
        return intrinsic, 0, 0, 0, None, None
    
    T = T_days / 365
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Construction de l'arbre des prix du sous-jacent
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    # Construction de l'arbre des prix de l'option
    option_tree = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option_tree[:, N] = np.maximum(0, stock_tree[:, N] - K)
    else:
        option_tree[:, N] = np.maximum(0, K - stock_tree[:, N])

    # Backward induction avec possibilité d'exercice anticipé
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            exercise = (stock_tree[j, i] - K) if option_type == "call" else (K - stock_tree[j, i])
            option_tree[j, i] = max(hold, exercise)

    # Calcul des Greeks à partir de l'arbre
    prix = option_tree[0, 0]
    
    # Delta : dérivée première par rapport à S
    if N >= 1:
        dS = stock_tree[0, 1] - stock_tree[1, 1]
        if abs(dS) > 1e-10:
            delta = (option_tree[0, 1] - option_tree[1, 1]) / dS
        else:
            delta = 0
    else:
        delta = 0
    
    # Gamma : dérivée seconde par rapport à S
    if N >= 2:
        S_up = stock_tree[0, 2]
        S_mid = stock_tree[1, 2]
        S_down = stock_tree[2, 2]
        
        delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (S_up - S_mid)
        delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (S_mid - S_down)
        
        dS_gamma = (S_up - S_down) / 2
        if abs(dS_gamma) > 1e-10:
            gamma = (delta_up - delta_down) / dS_gamma
        else:
            gamma = 0
    else:
        gamma = 0
    
    theta = 0  # Sera calculé séparément
    
    return prix, delta, gamma, theta, stock_tree, option_tree


def compute_greeks_optimized(S, K, T_days, r, sigma, N, option_type="put"):
    """
    Calcule tous les Greeks de manière optimisée
    """
    # Prix et Greeks de base
    prix, delta, gamma, _, _, _ = binomial_tree_american_option(S, K, T_days, r, sigma, N, option_type)
    
    # Theta : sensibilité au temps (1 jour de différence)
    if T_days > 1:
        prix_demain, _, _, _, _, _ = binomial_tree_american_option(S, K, T_days - 1, r, sigma, N, option_type)
        theta = prix_demain - prix  # Perte de valeur par jour
    else:
        theta = 0
    
    # Vega : sensibilité à la volatilité
    h_sigma = 0.01
    prix_up, _, _, _, _, _ = binomial_tree_american_option(S, K, T_days, r, sigma + h_sigma, N, option_type)
    prix_down, _, _, _, _, _ = binomial_tree_american_option(S, K, T_days, r, sigma - h_sigma, N, option_type)
    vega = (prix_up - prix_down) / (2 * h_sigma)
    
    # Rho : sensibilité au taux d'intérêt
    h_r = 0.001
    prix_up, _, _, _, _, _ = binomial_tree_american_option(S, K, T_days, r + h_r, sigma, N, option_type)
    prix_down, _, _, _, _, _ = binomial_tree_american_option(S, K, T_days, r - h_r, sigma, N, option_type)
    rho = (prix_up - prix_down) / (2 * h_r)
    
    return {
        'prix': prix,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# =============================================
# INTERFACE DE SAISIE
# =============================================

st.markdown("### ⚙️ PARAMETRES DE L'OPTION")

col_params1, col_params2, col_params3, col_params4 = st.columns(4)

with col_params1:
    S = st.number_input("SPOT (S)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)
    K = st.number_input("STRIKE (K)", min_value=1.0, max_value=10000.0, value=100.0, step=1.0)

with col_params2:
    T_days = st.number_input("MATURITÉ (JOURS)", min_value=1, max_value=365, value=30, step=1)
    r = st.number_input("TAUX (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f")

with col_params3:
    sigma = st.number_input("VOLATILITÉ (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01, format="%.3f")
    N = st.slider("PAS BINOMIAL (N)", min_value=20, max_value=500, value=100, step=20)

with col_params4:
    option_type = st.selectbox("TYPE D'OPTION", options=["call", "put"], index=1)
    calculate_button = st.button("🚀 CALCULER LE PRIX", use_container_width=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# CALCUL ET AFFICHAGE DES RÉSULTATS
# =============================================

if calculate_button or 'greeks' not in st.session_state:
    with st.spinner('💫 CALCUL EN COURS...'):
        greeks = compute_greeks_optimized(S, K, T_days, r, sigma, N, option_type)
        st.session_state.greeks = greeks
        st.session_state.params = {'S': S, 'K': K, 'T_days': T_days, 'r': r, 'sigma': sigma, 'N': N, 'option_type': option_type}

if 'greeks' in st.session_state:
    greeks = st.session_state.greeks
    
    # ===== AFFICHAGE DES RÉSULTATS PRINCIPAUX =====
    st.markdown("### 💰 RÉSULTATS DU PRICING")
    
    col_res1, col_res2, col_res3, col_res4, col_res5, col_res6 = st.columns(6)
    
    with col_res1:
        st.metric(
            label="PRIX",
            value=f"${greeks['prix']:.4f}"
        )
    
    with col_res2:
        st.metric(
            label="DELTA (∂V/∂S)",
            value=f"{greeks['delta']:.4f}"
        )
    
    with col_res3:
        st.metric(
            label="GAMMA (∂²V/∂S²)",
            value=f"{greeks['gamma']:.6f}"
        )
    
    with col_res4:
        st.metric(
            label="THETA (∂V/∂t)",
            value=f"{greeks['theta']:.4f}"
        )
    
    with col_res5:
        st.metric(
            label="VEGA (∂V/∂σ)",
            value=f"{greeks['vega']:.4f}"
        )
    
    with col_res6:
        st.metric(
            label="RHO (∂V/∂r)",
            value=f"{greeks['rho']:.4f}"
        )
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== GRAPHIQUES 2D DES GREEKS =====
    st.markdown("### 📈 ANALYSE DES GREEKS (2D)")
    
    with st.spinner('📊 GÉNÉRATION DES GRAPHIQUES...'):
        params = st.session_state.params
        
        # Ranges pour les calculs
        S_range = np.linspace(params['S'] * 0.7, params['S'] * 1.3, 30)
        sigma_range = np.linspace(0.05, 0.6, 30)
        T_range = np.linspace(max(1, params['T_days'] * 0.1), params['T_days'], 30)
        r_range = np.linspace(0.001, 0.15, 30)
        
        # Calculs pour les graphiques
        prix_vals_S, delta_vals_S, gamma_vals_S = [], [], []
        for s in S_range:
            g = compute_greeks_optimized(s, params['K'], params['T_days'], params['r'], params['sigma'], params['N'], params['option_type'])
            prix_vals_S.append(g['prix'])
            delta_vals_S.append(g['delta'])
            gamma_vals_S.append(g['gamma'])
        
        prix_vals_sigma, vega_vals = [], []
        for sig in sigma_range:
            g = compute_greeks_optimized(params['S'], params['K'], params['T_days'], params['r'], sig, params['N'], params['option_type'])
            prix_vals_sigma.append(g['prix'])
            vega_vals.append(g['vega'])
        
        prix_vals_T, theta_vals = [], []
        for t in T_range:
            g = compute_greeks_optimized(params['S'], params['K'], t, params['r'], params['sigma'], params['N'], params['option_type'])
            prix_vals_T.append(g['prix'])
            theta_vals.append(g['theta'])
        
        prix_vals_r, rho_vals = [], []
        for rate in r_range:
            g = compute_greeks_optimized(params['S'], params['K'], params['T_days'], rate, params['sigma'], params['N'], params['option_type'])
            prix_vals_r.append(g['prix'])
            rho_vals.append(g['rho'])
        
        # Création des subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'PRIX vs SPOT (S)', 'DELTA vs SPOT (S)', 'GAMMA vs SPOT (S)',
                'PRIX vs VOLATILITÉ (σ)', 'VEGA vs VOLATILITÉ (σ)', 'PRIX vs MATURITÉ (T)',
                'THETA vs MATURITÉ (T)', 'PRIX vs TAUX (r)', 'RHO vs TAUX (r)'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Ligne 1
        fig.add_trace(go.Scatter(x=S_range, y=prix_vals_S, mode='lines', line=dict(color='#00FFFF', width=2), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=S_range, y=delta_vals_S, mode='lines', line=dict(color='#00FF00', width=2), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=S_range, y=gamma_vals_S, mode='lines', line=dict(color='#FF00FF', width=2), showlegend=False), row=1, col=3)
        
        # Ligne 2
        fig.add_trace(go.Scatter(x=sigma_range, y=prix_vals_sigma, mode='lines', line=dict(color='#FFA500', width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=sigma_range, y=vega_vals, mode='lines', line=dict(color='#FFD700', width=2), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=T_range, y=prix_vals_T, mode='lines', line=dict(color='#00CED1', width=2), showlegend=False), row=2, col=3)
        
        # Ligne 3
        fig.add_trace(go.Scatter(x=T_range, y=theta_vals, mode='lines', line=dict(color='#FF0000', width=2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=r_range, y=prix_vals_r, mode='lines', line=dict(color='#32CD32', width=2), showlegend=False), row=3, col=2)
        fig.add_trace(go.Scatter(x=r_range, y=rho_vals, mode='lines', line=dict(color='#FF1493', width=2), showlegend=False), row=3, col=3)
        
        # Lignes verticales pour valeurs actuelles
        for i in range(1, 4):
            fig.add_vline(x=params['S'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=1, col=i)
        fig.add_vline(x=params['sigma'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=2, col=1)
        fig.add_vline(x=params['sigma'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=2, col=2)
        fig.add_vline(x=params['T_days'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=2, col=3)
        fig.add_vline(x=params['T_days'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=3, col=1)
        fig.add_vline(x=params['r'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=3, col=2)
        fig.add_vline(x=params['r'], line_dash="dash", line_color="#FFAA00", opacity=0.5, row=3, col=3)
        
        # Mise en forme
        fig.update_layout(
            height=900,
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=9),
            showlegend=False
        )
        
        fig.update_xaxes(gridcolor='#333', showgrid=True)
        fig.update_yaxes(gridcolor='#333', showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # ===== GRAPHIQUES 3D DES GREEKS =====
    st.markdown("### 📐 SURFACES 3D DES GREEKS")
    
    with st.spinner('🎨 GÉNÉRATION DES SURFACES 3D...'):
        # Grilles de calcul
        S_range_3d = np.linspace(params['S'] * 0.7, params['S'] * 1.3, 25)
        sigma_range_3d = np.linspace(0.05, 0.6, 25)
        T_range_3d = np.linspace(max(1, params['T_days'] * 0.1), params['T_days'], 25)
        
        S_grid, Sigma_grid = np.meshgrid(S_range_3d, sigma_range_3d)
        S_grid_T, T_grid = np.meshgrid(S_range_3d, T_range_3d)
        
        # Surface 1: Prix(S, σ)
        Prix_surface = np.zeros_like(S_grid)
        for i in range(len(sigma_range_3d)):
            for j in range(len(S_range_3d)):
                g = compute_greeks_optimized(S_range_3d[j], params['K'], params['T_days'], params['r'], sigma_range_3d[i], params['N'], params['option_type'])
                Prix_surface[i, j] = g['prix']
        
        # Surface 2: Delta(S, T)
        Delta_surface = np.zeros_like(S_grid_T)
        for i in range(len(T_range_3d)):
            for j in range(len(S_range_3d)):
                g = compute_greeks_optimized(S_range_3d[j], params['K'], T_range_3d[i], params['r'], params['sigma'], params['N'], params['option_type'])
                Delta_surface[i, j] = g['delta']
        
        # Surface 3: Gamma(S, σ)
        Gamma_surface = np.zeros_like(S_grid)
        for i in range(len(sigma_range_3d)):
            for j in range(len(S_range_3d)):
                g = compute_greeks_optimized(S_range_3d[j], params['K'], params['T_days'], params['r'], sigma_range_3d[i], params['N'], params['option_type'])
                Gamma_surface[i, j] = g['gamma']
        
        # Surface 4: Vega(S, σ)
        Vega_surface = np.zeros_like(S_grid)
        for i in range(len(sigma_range_3d)):
            for j in range(len(S_range_3d)):
                g = compute_greeks_optimized(S_range_3d[j], params['K'], params['T_days'], params['r'], sigma_range_3d[i], params['N'], params['option_type'])
                Vega_surface[i, j] = g['vega']
        
        # Création des 4 surfaces 3D
        fig_3d = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('PRIX (S, σ)', 'DELTA (S, T)', 'GAMMA (S, σ)', 'VEGA (S, σ)'),
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Surface Prix
        fig_3d.add_trace(
            go.Surface(x=S_grid, y=Sigma_grid, z=Prix_surface, colorscale='Viridis', showscale=False),
            row=1, col=1
        )
        
        # Surface Delta
        fig_3d.add_trace(
            go.Surface(x=S_grid_T, y=T_grid, z=Delta_surface, colorscale='Plasma', showscale=False),
            row=1, col=2
        )
        
        # Surface Gamma
        fig_3d.add_trace(
            go.Surface(x=S_grid, y=Sigma_grid, z=Gamma_surface, colorscale='Hot', showscale=False),
            row=2, col=1
        )
        
        # Surface Vega
        fig_3d.add_trace(
            go.Surface(x=S_grid, y=Sigma_grid, z=Vega_surface, colorscale='Electric', showscale=False),
            row=2, col=2
        )
        
        # Mise en forme
        fig_3d.update_layout(
            height=800,
            paper_bgcolor='#000',
            font=dict(color='#FFAA00', size=9),
            showlegend=False
        )
        
        # Update axes pour chaque subplot
        fig_3d.update_scenes(
            xaxis=dict(backgroundcolor='#111', gridcolor='#333', title_font=dict(color='#FFAA00')),
            yaxis=dict(backgroundcolor='#111', gridcolor='#333', title_font=dict(color='#FFAA00')),
            zaxis=dict(backgroundcolor='#111', gridcolor='#333', title_font=dict(color='#FFAA00'))
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# INFO SYSTÈME
# =============================================
col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 MODÈLE BINOMIAL CRR • OPTIONS AMÉRICAINES<br>
        🔢 CALCUL HAUTE PRÉCISION • ARBRE RECOMBINANT
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 SYSTÈME OPÉRATIONNEL • PARIS, FRANCE
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | OPTIONS PRICING ENGINE | BINOMIAL MODEL (COX-ROSS-RUBINSTEIN)<br>
    AMERICAN OPTIONS • EARLY EXERCISE FEATURE • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
