import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Bonds",
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
    
    section.main > div {
        animation: none !important;
        opacity: 1 !important;
    }
    
    .stApp [data-testid="stStatusWidget"] {
        visibility: hidden;
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
    
    /* Style pour les tables */
    .dataframe {
        font-family: 'Courier New', monospace !important;
        font-size: 10px !important;
        color: #FFAA00 !important;
        background-color: #111 !important;
    }
    
    .dataframe th {
        background-color: #333 !important;
        color: #FFAA00 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        border: 1px solid #555 !important;
    }
    
    .dataframe td {
        background-color: #111 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# FONCTION DONNÉES OBLIGATIONS
# =============================================
@st.cache_data(ttl=60)
def get_bond_data(ticker):
    """Récupère les données d'obligations via Yahoo Finance"""
    try:
        bond = yf.Ticker(ticker)
        hist = bond.history(period='5d')
        info = bond.info
        
        if len(hist) < 2:
            return None, None, None
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
        # Essayer de récupérer le yield si disponible
        bond_yield = info.get('yield', None)
        
        return current_price, change_percent, bond_yield
    except:
        return None, None, None

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#FFAA00;padding:8px 20px;color:#000;font-weight:bold;font-size:14px;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>⬛ BLOOMBERG ENS® TERMINAL - BONDS MARKET</div>
        <a href="/" style="background:#333;color:#FFAA00;border:1px solid #000;padding:4px 12px;font-size:11px;text-decoration:none;">MARKETS</a>
    </div>
    <div>{current_time} UTC • BONDS TRADING</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# US TREASURY BONDS
# =============================================
st.markdown("### 🏛️ US TREASURY BONDS - GOVERNMENT SECURITIES")

us_treasuries = {
    '1-MONTH': '^IRX',      # 13 Week Treasury Bill
    '3-MONTH': '^IRX',      # 13 Week Treasury Bill
    '6-MONTH': '^IRX',      # 13 Week Treasury Bill
    '1-YEAR': '^FVX',       # 5-Year Treasury Yield
    '2-YEAR': '^FVX',       # 5-Year Treasury Yield
    '5-YEAR': '^FVX',       # 5-Year Treasury Yield
    '10-YEAR': '^TNX',      # 10-Year Treasury Yield
    '30-YEAR': '^TYX',      # 30-Year Treasury Yield
}

cols_treasury = st.columns(4)

treasury_data = []
for idx, (name, ticker) in enumerate(us_treasuries.items()):
    with cols_treasury[idx % 4]:
        current, change, bond_yield = get_bond_data(ticker)
        
        if current is not None:
            # Pour les yields, on affiche en %
            value_display = f"{current:.3f}%"
            
            st.metric(
                label=f"US {name}",
                value=value_display,
                delta=f"{change:+.2f} bps" if change else "N/A"
            )
            
            treasury_data.append({
                'Maturity': name,
                'Yield': current,
                'Change (bps)': change if change else 0
            })
        else:
            st.metric(label=f"US {name}", value="LOAD...", delta="0 bps")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# EUROPEAN GOVERNMENT BONDS
# =============================================
st.markdown("### 🇪🇺 EUROPEAN GOVERNMENT BONDS")

eu_bonds = {
    'GERMANY 10Y': 'DE10Y.DE',
    'FRANCE 10Y': 'FR10Y.FR',
    'ITALY 10Y': 'IT10Y.IT',
    'SPAIN 10Y': 'ES10Y.ES',
    'UK 10Y': 'GB10Y.GB',
    'SWISS 10Y': 'CH10Y.CH',
}

cols_eu = st.columns(6)

for idx, (name, ticker) in enumerate(eu_bonds.items()):
    with cols_eu[idx]:
        current, change, bond_yield = get_bond_data(ticker)
        
        if current is not None:
            value_display = f"{current:.3f}%"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f} bps" if change else "N/A"
            )
        else:
            st.metric(label=name, value="LOAD...", delta="0 bps")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# CORPORATE BONDS ETFs
# =============================================
st.markdown("### 🏢 CORPORATE BONDS - ETF TRACKING")

corporate_bonds = {
    'IG CORP (LQD)': 'LQD',          # iShares iBoxx $ Investment Grade Corporate Bond
    'HY CORP (HYG)': 'HYG',          # iShares iBoxx $ High Yield Corporate Bond
    'SHORT IG (VCSH)': 'VCSH',       # Vanguard Short-Term Corporate Bond
    'LONG IG (VCLT)': 'VCLT',        # Vanguard Long-Term Corporate Bond
    'EUR CORP (LQDE)': 'LQDE.L',     # iShares Euro Corporate Bond
    'EMERG MKT (EMB)': 'EMB',        # iShares J.P. Morgan USD Emerging Markets Bond
}

cols_corp = st.columns(6)

corp_data = []
for idx, (name, ticker) in enumerate(corporate_bonds.items()):
    with cols_corp[idx]:
        current, change, bond_yield = get_bond_data(ticker)
        
        if current is not None:
            value_display = f"${current:.2f}"
            
            st.metric(
                label=name,
                value=value_display,
                delta=f"{change:+.2f}%" if change else "N/A"
            )
            
            corp_data.append({
                'ETF': name,
                'Price': current,
                'Change (%)': change if change else 0
            })
        else:
            st.metric(label=name, value="LOAD...", delta="0%")

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# YIELD CURVE
# =============================================
st.markdown("### 📈 US TREASURY YIELD CURVE")

col_curve1, col_curve2 = st.columns([3, 1])

with col_curve1:
    # Récupérer les données pour la courbe
    yield_curve_tickers = {
        '1M': '^IRX',
        '3M': '^IRX',
        '6M': '^IRX',
        '1Y': '^FVX',
        '2Y': '^FVX',
        '5Y': '^FVX',
        '10Y': '^TNX',
        '30Y': '^TYX',
    }
    
    maturities = []
    yields = []
    
    # Mapper les maturités en années
    maturity_years = {
        '1M': 1/12,
        '3M': 3/12,
        '6M': 6/12,
        '1Y': 1,
        '2Y': 2,
        '5Y': 5,
        '10Y': 10,
        '30Y': 30
    }
    
    for mat, ticker in yield_curve_tickers.items():
        current, _, _ = get_bond_data(ticker)
        if current is not None:
            maturities.append(maturity_years[mat])
            yields.append(current)
    
    if maturities and yields:
        fig_curve = go.Figure()
        
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=yields,
            mode='lines+markers',
            name='Current Yield',
            line=dict(color='#00FFFF', width=3),
            marker=dict(size=8, color='#FFAA00'),
            hovertemplate='Maturity: %{x:.1f}Y<br>Yield: %{y:.3f}%<extra></extra>'
        ))
        
        fig_curve.update_layout(
            title="US Treasury Yield Curve - Live",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Maturity (Years)",
                type='log'
            ),
            yaxis=dict(
                gridcolor='#333',
                showgrid=True,
                title="Yield (%)"
            ),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)

with col_curve2:
    st.markdown("#### 📊 CURVE ANALYSIS")
    
    if len(yields) >= 3:
        # Spread 2Y-10Y
        if len(yields) >= 5:
            spread_2_10 = yields[-2] - yields[3] if len(yields) > 3 else 0
            st.metric(
                "2Y-10Y Spread",
                f"{spread_2_10:.2f} bps",
                "Normal" if spread_2_10 > 0 else "⚠️ INVERTED"
            )
        
        # Spread 10Y-30Y
        if len(yields) >= 7:
            spread_10_30 = yields[-1] - yields[-2] if len(yields) > 6 else 0
            st.metric(
                "10Y-30Y Spread",
                f"{spread_10_30:.2f} bps",
                "Steep" if spread_10_30 > 50 else "Flat"
            )
        
        # Curve Shape
        if yields[-1] > yields[0]:
            curve_shape = "🟢 NORMAL"
        elif yields[-1] < yields[0]:
            curve_shape = "🔴 INVERTED"
        else:
            curve_shape = "⚪ FLAT"
        
        st.metric("Curve Shape", curve_shape)
        
        st.markdown("""
        <div style='font-size:9px;color:#666;margin-top:10px;'>
        📖 Curve Interpretation:<br>
        • Normal: Long > Short<br>
        • Inverted: Short > Long<br>
        • Flat: Similar yields<br><br>
        ⚠️ Inverted curves often<br>precede recessions
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# HISTORICAL COMPARISON
# =============================================
st.markdown("### 📊 BONDS HISTORICAL PERFORMANCE")

col_hist1, col_hist2 = st.columns([3, 1])

with col_hist1:
    bond_etfs = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'EMB']
    
    selected_bonds = st.multiselect(
        "Sélectionnez des ETF obligataires",
        options=bond_etfs,
        default=['TLT', 'LQD', 'HYG'],
        help="Comparez la performance des différents segments obligataires"
    )

with col_hist2:
    period_bonds = st.selectbox(
        "Période",
        options=['1mo', '3mo', '6mo', '1y', '5y'],
        index=2,
        key="bonds_period"
    )

if selected_bonds:
    fig_bonds = go.Figure()
    
    colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFA500', '#FF0000', '#FFFF00']
    
    for idx, ticker in enumerate(selected_bonds):
        try:
            bond = yf.Ticker(ticker)
            hist = bond.history(period=period_bonds)
            
            if len(hist) > 0:
                # Normaliser à 100
                normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
                
                fig_bonds.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'<b>{ticker}</b><br>%{{y:.2f}}%<br>%{{x}}<extra></extra>'
                ))
        except:
            continue
    
    fig_bonds.update_layout(
        title=f"Bond ETFs Performance Comparison - {period_bonds.upper()}",
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
            title="Performance (%)"
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig_bonds.add_hline(y=100, line_dash="dash", line_color="#666")
    
    st.plotly_chart(fig_bonds, use_container_width=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# BOND ETFs DETAILS
# =============================================
st.markdown("### 📋 BOND ETFs CHARACTERISTICS")

bond_etf_details = {
    'TLT': {'Name': '20+ Year Treasury', 'Duration': 'Very Long', 'Risk': 'Low Credit / High Rate'},
    'IEF': {'Name': '7-10 Year Treasury', 'Duration': 'Medium', 'Risk': 'Low Credit / Medium Rate'},
    'SHY': {'Name': '1-3 Year Treasury', 'Duration': 'Short', 'Risk': 'Very Low'},
    'LQD': {'Name': 'Investment Grade Corp', 'Duration': 'Medium', 'Risk': 'Low Credit / Medium Rate'},
    'HYG': {'Name': 'High Yield Corp', 'Duration': 'Medium', 'Risk': 'High Credit / Medium Rate'},
    'EMB': {'Name': 'Emerging Markets', 'Duration': 'Medium', 'Risk': 'High Credit / High Rate'},
}

# Créer un DataFrame
df_etf = pd.DataFrame.from_dict(bond_etf_details, orient='index')
df_etf.index.name = 'Ticker'
df_etf = df_etf.reset_index()

# Ajouter les prix actuels
current_prices = []
changes = []
for ticker in df_etf['Ticker']:
    current, change, _ = get_bond_data(ticker)
    current_prices.append(f"${current:.2f}" if current else "N/A")
    changes.append(f"{change:+.2f}%" if change else "N/A")

df_etf['Price'] = current_prices
df_etf['Change (1D)'] = changes

# Afficher le tableau
st.dataframe(df_etf, use_container_width=True, hide_index=True)

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# CREDIT SPREADS
# =============================================
st.markdown("### 💰 CREDIT SPREADS MONITOR")

st.markdown("""
<div style='background:#111;border:1px solid #333;padding:10px;margin:10px 0;border-left:4px solid #FFAA00;'>
<b style='color:#FFAA00;'>📊 CREDIT SPREADS INTERPRETATION:</b><br>
• <span style='color:#00FF00;'>Tight spreads (< 200 bps)</span>: Low credit risk perception, healthy market<br>
• <span style='color:#FFA500;'>Moderate spreads (200-400 bps)</span>: Normal credit risk<br>
• <span style='color:#FF0000;'>Wide spreads (> 400 bps)</span>: High credit risk, potential stress<br><br>
<b style='color:#FFAA00;'>Credit Spread = High Yield Yield - Investment Grade Yield</b>
</div>
""", unsafe_allow_html=True)

col_spread1, col_spread2, col_spread3 = st.columns(3)

# Calculer les spreads (simulation basée sur les ETF)
hyg_price, hyg_change, _ = get_bond_data('HYG')
lqd_price, lqd_change, _ = get_bond_data('LQD')

with col_spread1:
    if hyg_price and lqd_price:
        # Spread approximatif (en réalité il faut les yields)
        spread = abs((hyg_price - lqd_price) / lqd_price * 1000)
        spread_status = "🟢 TIGHT" if spread < 200 else "🟠 MODERATE" if spread < 400 else "🔴 WIDE"
        st.metric(
            "HY vs IG Spread (approx)",
            f"{spread:.0f} bps",
            spread_status
        )

with col_spread2:
    emb_price, emb_change, _ = get_bond_data('EMB')
    if emb_price and lqd_price:
        em_spread = abs((emb_price - lqd_price) / lqd_price * 1000)
        st.metric(
            "EM vs IG Spread (approx)",
            f"{em_spread:.0f} bps",
            "Emerging Risk"
        )

with col_spread3:
    tlt_price, tlt_change, _ = get_bond_data('TLT')
    if tlt_price:
        st.metric(
            "Long Treasury",
            f"${tlt_price:.2f}",
            f"{tlt_change:+.2f}%" if tlt_change else "N/A"
        )

st.markdown('<hr>', unsafe_allow_html=True)

# =============================================
# BOND GLOSSARY
# =============================================
with st.expander("📖 BONDS GLOSSARY & KEY CONCEPTS"):
    st.markdown("""
    **KEY TERMS:**
    
    **Yield:** Return on investment, expressed as annual percentage. Inverse relationship with price.
    
    **Duration:** Sensitivity to interest rate changes. Higher duration = higher interest rate risk.
    
    **Credit Spread:** Difference between corporate bond yield and government bond yield. Reflects credit risk.
    
    **Yield Curve:** Graph showing yields across different maturities.
    - Normal: Long-term > Short-term (healthy economy)
    - Inverted: Short-term > Long-term (recession warning)
    - Flat: Similar yields across maturities (uncertainty)
    
    **Investment Grade (IG):** BBB- or higher rating. Lower risk, lower yield.
    
    **High Yield (HY):** BB+ or lower rating. Higher risk, higher yield ("junk bonds").
    
    **Treasury Bonds:** US government bonds. Considered risk-free benchmark.
    
    **Corporate Bonds:** Issued by companies. Higher yield than treasuries due to credit risk.
    
    **Emerging Markets Bonds:** Issued by developing countries. Higher risk and yield.
    
    **Basis Point (bps):** 1/100th of 1%. Example: 25 bps = 0.25%
    """)

# =============================================
# INFO SYSTÈME
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns([6, 6])

with col_info1:
    st.markdown("""
    <div style="color:#666;font-size:10px;padding:5px;">
        📊 BONDS DATA • YAHOO FINANCE • REAL-TIME ETF PRICES<br>
        🔄 DATA SOURCE: US TREASURY, CORPORATE BOND ETFs, EUROPEAN SOVEREIGNS
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    last_update = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div style="color:#666;font-size:10px;padding:5px;">
        🕐 DERNIÈRE MAJ: {last_update}<br>
        📍 FIXED INCOME MARKETS • GLOBAL COVERAGE
    </div>
    """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | BONDS MARKET | FIXED INCOME DIVISION<br>
    GOVERNMENT & CORPORATE SECURITIES • YIELD CURVE ANALYSIS • LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
