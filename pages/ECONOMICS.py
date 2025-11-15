import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Economics Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Votre clé API FRED
FRED_API_KEY = "ce5dbb3d3fcd8669f2fe2cdd9c79a7da"

# CSS Bloomberg style
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
        background-color: #000000;
        color: #FFAA00;
    }
    
    .stApp {
        background-color: #000000;
    }
    
    .bloomberg-header {
        background: #FFAA00;
        padding: 5px 20px;
        color: #000000;
        font-weight: bold;
        font-size: 14px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border-bottom: 2px solid #FFAA00;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
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
        font-size: 20px !important;
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
        font-size: 12px !important;
        font-weight: bold !important;
        font-family: 'Courier New', monospace !important;
    }
    
    .stButton > button {
        background-color: #333;
        color: #FFAA00;
        font-weight: bold;
        border: 1px solid #FFAA00;
        padding: 6px 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 0px;
        font-size: 10px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000;
    }
    
    .stTextInput input, .stSelectbox select {
        background-color: #000;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    hr {
        border-color: #333333;
        margin: 5px 0;
    }
    
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    p, div, span, label {
        font-family: 'Courier New', monospace !important;
        font-size: 11px;
        color: #FFAA00;
    }
    
    .warning-box {
        background-color: #1a0a00;
        border-left: 3px solid #FF6600;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
    .indicator-box {
        background-color: #0a0a0a;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
current_time = datetime.now()
st.markdown(f'''
<div class="bloomberg-header">
    <div>⬛ BLOOMBERG ENS® TERMINAL - ECONOMIC DATA</div>
    <div style="font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold; color: #000;">
        {current_time.strftime("%H:%M:%S")} PARIS
    </div>
</div>
''', unsafe_allow_html=True)

# Fonctions FRED API
@st.cache_data(ttl=3600)
def get_fred_series(series_id, observation_start=None):
    """Récupère une série de données FRED"""
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        if observation_start:
            params['observation_start'] = observation_start
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            
            if observations:
                df = pd.DataFrame(observations)
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                df = df.sort_values('date')
                
                return df
        
        return None
        
    except Exception as e:
        st.error(f"Erreur FRED API pour {series_id}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_fred_series_info(series_id):
    """Récupère les informations sur une série FRED"""
    try:
        url = "https://api.stlouisfed.org/fred/series"
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            series = data.get('seriess', [])
            if series:
                return series[0]
        
        return None
        
    except Exception as e:
        return None

def calculate_yoy_change(df):
    """Calcule la variation Year-over-Year"""
    if df is None or len(df) < 12:
        return None
    
    df = df.copy()
    df['yoy_change'] = df['value'].pct_change(12) * 100
    return df

# Dictionnaire des séries économiques importantes
ECONOMIC_SERIES = {
    'Interest Rates': {
        'FEDFUNDS': 'Fed Funds Rate',
        'DGS10': '10-Year Treasury',
        'DGS2': '2-Year Treasury',
        'DGS30': '30-Year Treasury',
        'MORTGAGE30US': '30Y Mortgage Rate'
    },
    'Inflation': {
        'CPIAUCSL': 'CPI (All Items)',
        'CPILFESL': 'Core CPI',
        'PCEPI': 'PCE Price Index',
        'PCEPILFE': 'Core PCE'
    },
    'Employment': {
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'JTSJOL': 'JOLTS Job Openings',
        'ICSA': 'Initial Claims'
    },
    'Growth': {
        'GDP': 'GDP',
        'GDPC1': 'Real GDP',
        'INDPRO': 'Industrial Production',
        'RSXFS': 'Retail Sales'
    },
    'Markets': {
        'SP500': 'S&P 500',
        'VIXCLS': 'VIX',
        'DEXUSEU': 'EUR/USD',
        'T10Y2Y': '10Y-2Y Spread'
    },
    'Monetary': {
        'M2SL': 'M2 Money Supply',
        'WALCL': 'Fed Balance Sheet'
    }
}

# ONGLETS PRINCIPAUX
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 DASHBOARD", "📈 CUSTOM ANALYSIS", "📐 CONSTRUCTED INDICATORS", "🔍 SERIES SEARCH", "📥 DOWNLOAD DATA"])

# TAB 1: DASHBOARD
with tab1:
    st.markdown("### 📊 ECONOMIC INDICATORS DASHBOARD")
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_dash"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # TAUX D'INTÉRÊT
    st.markdown("#### 💰 INTEREST RATES")
    cols_rates = st.columns(4)
    
    rates_series = [
        ('FEDFUNDS', 'FED FUNDS'),
        ('DGS10', '10Y TREASURY'),
        ('DGS2', '2Y TREASURY'),
        ('MORTGAGE30US', '30Y MORTGAGE')
    ]
    
    for idx, (series_id, label) in enumerate(rates_series):
        with cols_rates[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 1:
                current_value = df['value'].iloc[-1]
                previous_value = df['value'].iloc[-2]
                change = current_value - previous_value
                
                st.metric(
                    label=label,
                    value=f"{current_value:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # INFLATION
    st.markdown("#### 📊 INFLATION")
    cols_inflation = st.columns(4)
    
    inflation_series = [
        ('CPIAUCSL', 'CPI'),
        ('CPILFESL', 'CORE CPI'),
        ('PCEPI', 'PCE'),
        ('PCEPILFE', 'CORE PCE')
    ]
    
    for idx, (series_id, label) in enumerate(inflation_series):
        with cols_inflation[idx]:
            df = get_fred_series(series_id)
            if df is not None and len(df) > 12:
                df_yoy = calculate_yoy_change(df)
                current_yoy = df_yoy['yoy_change'].iloc[-1]
                previous_yoy = df_yoy['yoy_change'].iloc[-2]
                change = current_yoy - previous_yoy
                
                st.metric(
                    label=f"{label} YoY",
                    value=f"{current_yoy:.2f}%",
                    delta=f"{change:+.2f}%"
                )
            else:
                st.metric(label=label, value="N/A")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # YIELD CURVE
    st.markdown("#### 📉 YIELD CURVE")
    
    treasury_rates = {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    }
    
    curve_data = []
    maturities = []
    
    for maturity, series_id in treasury_rates.items():
        df = get_fred_series(series_id)
        if df is not None and len(df) > 0:
            curve_data.append(df['value'].iloc[-1])
            maturities.append(maturity)
    
    if curve_data:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=maturities,
            y=curve_data,
            mode='lines+markers',
            line=dict(color='#FFAA00', width=2),
            marker=dict(size=8, color='#00FF00')
        ))
        
        fig_curve.update_layout(
            title="US Treasury Yield Curve",
            paper_bgcolor='#000',
            plot_bgcolor='#111',
            font=dict(color='#FFAA00', size=10),
            xaxis=dict(title="Maturity", gridcolor='#333'),
            yaxis=dict(title="Yield (%)", gridcolor='#333'),
            height=400
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)

# TAB 2: ANALYSE PERSONNALISÉE
with tab2:
    st.markdown("### 📈 CUSTOM ECONOMIC ANALYSIS")
    
    col_analysis1, col_analysis2 = st.columns([2, 1])
    
    with col_analysis1:
        categories = list(ECONOMIC_SERIES.keys())
        selected_category = st.selectbox(
            "SELECT CATEGORY",
            options=categories,
            key="category_select"
        )
        
        series_in_category = ECONOMIC_SERIES[selected_category]
        
        selected_series = st.multiselect(
            "SELECT SERIES TO COMPARE",
            options=list(series_in_category.keys()),
            default=list(series_in_category.keys())[:2],
            format_func=lambda x: f"{x} - {series_in_category[x]}",
            key="series_multiselect"
        )
    
    with col_analysis2:
        lookback_period = st.selectbox(
            "LOOKBACK PERIOD",
            options=['1Y', '2Y', '5Y', '10Y', 'MAX'],
            index=2,
            key="lookback_select"
        )
    
    if st.button("📊 GENERATE ANALYSIS", use_container_width=True, key="generate_analysis"):
        if selected_series:
            if lookback_period == 'MAX':
                start_date = None
            else:
                years = int(lookback_period[:-1])
                start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
            
            series_data = {}
            for series_id in selected_series:
                df = get_fred_series(series_id, observation_start=start_date)
                if df is not None:
                    series_data[series_id] = df
            
            if series_data:
                fig = go.Figure()
                
                for series_id, df in series_data.items():
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['value'],
                        mode='lines',
                        name=f"{series_id}",
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title=f"{selected_category} Comparison",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Value"),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 3: INDICATEURS CONSTRUITS
with tab3:
    st.markdown("### 📐 CONSTRUCTED MACRO INDICATORS")
    
    st.markdown("""
    <div style="background-color: #0a0a0a; border-left: 3px solid #FFAA00; padding: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 10px; color: #999;">
        Ces indicateurs sont calculés à partir de séries FRED brutes en utilisant des formules économétriques standard.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_constructed"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 8px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 1. INFLATION YoY =====
    st.markdown("#### 📊 1. INFLATION YoY")
    st.caption("Formula: `100 * (CPI_t / CPI_{t-12} - 1)` | Series: `CPIAUCSL`")
    
    df_cpi = get_fred_series('CPIAUCSL')
    
    if df_cpi is not None and len(df_cpi) > 12:
        df_cpi_yoy = calculate_yoy_change(df_cpi)
        
        col_inf1, col_inf2, col_inf3 = st.columns([2, 1, 1])
        
        with col_inf1:
            fig_inf = go.Figure()
            fig_inf.add_trace(go.Scatter(
                x=df_cpi_yoy['date'],
                y=df_cpi_yoy['yoy_change'],
                mode='lines',
                line=dict(color='#FFAA00', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 170, 0, 0.1)',
                name='CPI YoY'
            ))
            
            # Ligne cible Fed 2%
            fig_inf.add_hline(y=2, line_dash="dash", line_color="#00FF00", 
                             annotation_text="Fed Target 2%", annotation_position="right")
            
            fig_inf.update_layout(
                title="CPI Year-over-Year Change (%)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="YoY Change (%)"),
                height=350
            )
            
            st.plotly_chart(fig_inf, use_container_width=True)
        
        with col_inf2:
            current_inf = df_cpi_yoy['yoy_change'].iloc[-1]
            previous_inf = df_cpi_yoy['yoy_change'].iloc[-2]
            
            st.metric(
                label="CURRENT CPI YoY",
                value=f"{current_inf:.2f}%",
                delta=f"{current_inf - previous_inf:+.2f}%"
            )
            
            st.metric(
                label="VS FED TARGET",
                value=f"{current_inf - 2:.2f}%",
                delta="Above" if current_inf > 2 else "Below"
            )
        
        with col_inf3:
            # Stats
            st.metric("MAX (5Y)", f"{df_cpi_yoy['yoy_change'].tail(60).max():.2f}%")
            st.metric("MIN (5Y)", f"{df_cpi_yoy['yoy_change'].tail(60).min():.2f}%")
            st.metric("MEAN (5Y)", f"{df_cpi_yoy['yoy_change'].tail(60).mean():.2f}%")
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 2. REAL YIELD =====
    st.markdown("#### 📊 2. REAL YIELD (10Y)")
    st.caption("Formula: `DGS10 - CPI_YoY` | Series: `DGS10`, `CPIAUCSL`")
    
    df_10y = get_fred_series('DGS10')
    
    if df_10y is not None and df_cpi_yoy is not None:
        # Merger les deux séries
        df_real_yield = pd.merge(
            df_10y[['date', 'value']].rename(columns={'value': 'nominal_10y'}),
            df_cpi_yoy[['date', 'yoy_change']].rename(columns={'yoy_change': 'inflation'}),
            on='date',
            how='inner'
        )
        
        df_real_yield['real_yield'] = df_real_yield['nominal_10y'] - df_real_yield['inflation']
        
        col_ry1, col_ry2, col_ry3 = st.columns([2, 1, 1])
        
        with col_ry1:
            fig_ry = go.Figure()
            
            fig_ry.add_trace(go.Scatter(
                x=df_real_yield['date'],
                y=df_real_yield['real_yield'],
                mode='lines',
                line=dict(color='#00FF00', width=2),
                name='Real Yield'
            ))
            
            fig_ry.add_trace(go.Scatter(
                x=df_real_yield['date'],
                y=df_real_yield['nominal_10y'],
                mode='lines',
                line=dict(color='#FFAA00', width=1, dash='dash'),
                name='Nominal 10Y'
            ))
            
            fig_ry.add_hline(y=0, line_dash="dot", line_color="#FF0000")
            
            fig_ry.update_layout(
                title="Real Yield vs Nominal 10Y Treasury",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="Yield (%)"),
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_ry, use_container_width=True)
        
        with col_ry2:
            current_real = df_real_yield['real_yield'].iloc[-1]
            current_nominal = df_real_yield['nominal_10y'].iloc[-1]
            
            st.metric(
                label="REAL YIELD",
                value=f"{current_real:.2f}%"
            )
            
            st.metric(
                label="NOMINAL 10Y",
                value=f"{current_nominal:.2f}%"
            )
        
        with col_ry3:
            st.metric("MEAN (5Y)", f"{df_real_yield['real_yield'].tail(60).mean():.2f}%")
            
            if current_real < 0:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                    ⚠️ NEGATIVE REAL YIELD
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                    ✅ POSITIVE REAL YIELD
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 3. YIELD CURVE SLOPES =====
    st.markdown("#### 📊 3. YIELD CURVE SLOPES (RECESSION INDICATORS)")
    
    # 10Y-2Y Spread
    st.caption("**A. 10Y-2Y Spread** | Formula: `DGS10 - DGS2` | Classic recession indicator")
    
    df_10y2y = get_fred_series('T10Y2Y')
    
    if df_10y2y is not None:
        col_yc1, col_yc2, col_yc3 = st.columns([2, 1, 1])
        
        with col_yc1:
            fig_10y2y = go.Figure()
            
            # Zones de récession (à ajouter manuellement ou via NBER data)
            fig_10y2y.add_trace(go.Scatter(
                x=df_10y2y['date'],
                y=df_10y2y['value'],
                mode='lines',
                line=dict(color='#FFAA00', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 170, 0, 0.1)',
                name='10Y-2Y Spread'
            ))
            
            fig_10y2y.add_hline(y=0, line_dash="dash", line_color="#FF0000", 
                               annotation_text="Inversion Line", annotation_position="left")
            
            fig_10y2y.update_layout(
                title="10Y-2Y Treasury Spread (Recession Indicator)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="Spread (%)"),
                height=300
            )
            
            st.plotly_chart(fig_10y2y, use_container_width=True)
        
        with col_yc2:
            current_spread = df_10y2y['value'].iloc[-1]
            
            st.metric(
                label="10Y-2Y SPREAD",
                value=f"{current_spread:.2f}%"
            )
            
            # Signal de récession
            if current_spread < 0:
                inversion_duration = (df_10y2y[df_10y2y['value'] < 0].tail(1)['date'].iloc[0] - 
                                     df_10y2y['date'].iloc[-1]).days
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF0000; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF0000; font-weight: bold;">
                    🔴 INVERTED CURVE
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #FF6600;">
                    Recession signal active
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                    ✅ NORMAL CURVE
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_yc3:
            # Historique des inversions
            inversions = df_10y2y[df_10y2y['value'] < 0]
            st.metric("INVERSIONS (HISTORY)", f"{len(inversions)}")
            st.caption("Number of months with inversion")
    
    st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
    
    # 10Y-3M Spread
    st.caption("**B. 10Y-3M Spread** | Formula: `DGS10 - DGS3MO` | Better leading indicator")
    
    df_10y_data = get_fred_series('DGS10')
    df_3m_data = get_fred_series('DGS3MO')
    
    if df_10y_data is not None and df_3m_data is not None:
        df_10y3m = pd.merge(
            df_10y_data[['date', 'value']].rename(columns={'value': 'dgs10'}),
            df_3m_data[['date', 'value']].rename(columns={'value': 'dgs3mo'}),
            on='date',
            how='inner'
        )
        df_10y3m['spread'] = df_10y3m['dgs10'] - df_10y3m['dgs3mo']
        
        col_yc3m1, col_yc3m2 = st.columns([2, 1])
        
        with col_yc3m1:
            fig_10y3m = go.Figure()
            
            fig_10y3m.add_trace(go.Scatter(
                x=df_10y3m['date'],
                y=df_10y3m['spread'],
                mode='lines',
                line=dict(color='#00FFFF', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 255, 0.1)',
                name='10Y-3M Spread'
            ))
            
            fig_10y3m.add_hline(y=0, line_dash="dash", line_color="#FF0000")
            
            fig_10y3m.update_layout(
                title="10Y-3M Treasury Spread (Leading Recession Indicator)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="Spread (%)"),
                height=300
            )
            
            st.plotly_chart(fig_10y3m, use_container_width=True)
        
        with col_yc3m2:
            current_spread_3m = df_10y3m['spread'].iloc[-1]
            
            st.metric(
                label="10Y-3M SPREAD",
                value=f"{current_spread_3m:.2f}%"
            )
            
            if current_spread_3m < 0:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF0000; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF0000; font-weight: bold;">
                    🔴 INVERTED
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #FF6600;">
                    Strong recession signal
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 4. MONEY SUPPLY GROWTH =====
    st.markdown("#### 📊 4. MONEY SUPPLY GROWTH (M2 YoY)")
    st.caption("Formula: `100 * (M2_t / M2_{t-12} - 1)` | Series: `M2SL`")
    
    df_m2 = get_fred_series('M2SL')
    
    if df_m2 is not None and len(df_m2) > 12:
        df_m2_yoy = calculate_yoy_change(df_m2)
        
        col_m2_1, col_m2_2 = st.columns([2, 1])
        
        with col_m2_1:
            fig_m2 = go.Figure()
            
            fig_m2.add_trace(go.Scatter(
                x=df_m2_yoy['date'],
                y=df_m2_yoy['yoy_change'],
                mode='lines',
                line=dict(color='#FF00FF', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 255, 0.1)',
                name='M2 YoY Growth'
            ))
            
            fig_m2.add_hline(y=0, line_dash="dot", line_color="#FF0000")
            
            fig_m2.update_layout(
                title="M2 Money Supply Year-over-Year Growth (%)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="YoY Growth (%)"),
                height=350
            )
            
            st.plotly_chart(fig_m2, use_container_width=True)
        
        with col_m2_2:
            current_m2_growth = df_m2_yoy['yoy_change'].iloc[-1]
            
            st.metric(
                label="M2 GROWTH YoY",
                value=f"{current_m2_growth:.2f}%"
            )
            
            st.metric(
                label="M2 CURRENT",
                value=f"${df_m2['value'].iloc[-1]/1000:.1f}T"
            )
            
            if current_m2_growth < 0:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                    ⚠️ CONTRACTING M2
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                    Money supply shrinking
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 5. REAL GDP GROWTH =====
    st.markdown("#### 📊 5. REAL GDP QUARTER-ON-QUARTER (ANNUALIZED)")
    st.caption("Formula: `400 * (GDPC1_t / GDPC1_{t-1} - 1)` | Series: `GDPC1`")
    
    df_gdp = get_fred_series('GDPC1')
    
    if df_gdp is not None and len(df_gdp) > 1:
        df_gdp['qoq_annualized'] = 400 * df_gdp['value'].pct_change(1)
        
        col_gdp1, col_gdp2 = st.columns([2, 1])
        
        with col_gdp1:
            fig_gdp = go.Figure()
            
            # Barres pour QoQ growth
            colors = ['#00FF00' if x > 0 else '#FF0000' for x in df_gdp['qoq_annualized']]
            
            fig_gdp.add_trace(go.Bar(
                x=df_gdp['date'],
                y=df_gdp['qoq_annualized'],
                marker_color=colors,
                name='GDP QoQ Growth'
            ))
            
            fig_gdp.add_hline(y=0, line_dash="dot", line_color="#FFAA00")
            
            fig_gdp.update_layout(
                title="Real GDP Quarter-on-Quarter Growth (Annualized %)",
                paper_bgcolor='#000',
                plot_bgcolor='#111',
                font=dict(color='#FFAA00', size=10),
                xaxis=dict(gridcolor='#333', title="Date"),
                yaxis=dict(gridcolor='#333', title="QoQ Growth (%)"),
                height=350
            )
            
            st.plotly_chart(fig_gdp, use_container_width=True)
        
        with col_gdp2:
            current_gdp_growth = df_gdp['qoq_annualized'].iloc[-1]
            previous_gdp_growth = df_gdp['qoq_annualized'].iloc[-2]
            
            st.metric(
                label="LATEST GDP GROWTH",
                value=f"{current_gdp_growth:.2f}%",
                delta=f"{current_gdp_growth - previous_gdp_growth:+.2f}%"
            )
            
            # Compter les quarters négatifs récents
            recent_negative = (df_gdp['qoq_annualized'].tail(2) < 0).sum()
            
            if recent_negative >= 2:
                st.markdown("""
                <div style="background-color: #1a0a00; border-left: 3px solid #FF0000; padding: 8px; margin: 5px 0;">
                    <p style="margin: 0; font-size: 10px; color: #FF0000; font-weight: bold;">
                    🔴 TECHNICAL RECESSION
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 9px; color: #FF6600;">
                    2 consecutive negative quarters
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div style="border-bottom: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # ===== 6. TAYLOR RULE =====
    st.markdown("#### 📊 6. TAYLOR RULE IMPLIED RATE")
    st.caption("Formula: `r* + inflation + 0.5*(inflation - target) + 0.5*output_gap`")
    st.caption("Simplified: `r* = 2%, target = 2%, output_gap ≈ 0` → `Taylor Rate = 2 + inflation + 0.5*(inflation - 2)`")
    
    if df_cpi_yoy is not None and df_10y_data is not None:
        # Calculer le taux de Taylor
        r_star = 2.0  # Taux neutre estimé
        inflation_target = 2.0
        
        df_taylor = df_cpi_yoy[['date', 'yoy_change']].copy()
        df_taylor['taylor_rate'] = r_star + df_taylor['yoy_change'] + 0.5 * (df_taylor['yoy_change'] - inflation_target)
        
        # Merger avec Fed Funds
        df_fedfunds = get_fred_series('FEDFUNDS')
        if df_fedfunds is not None:
            df_taylor = pd.merge(
                df_taylor,
                df_fedfunds[['date', 'value']].rename(columns={'value': 'fed_funds'}),
                on='date',
                how='inner'
            )
            
            df_taylor['policy_gap'] = df_taylor['fed_funds'] - df_taylor['taylor_rate']
            
            col_taylor1, col_taylor2 = st.columns([2, 1])
            
            with col_taylor1:
                fig_taylor = go.Figure()
                
                fig_taylor.add_trace(go.Scatter(
                    x=df_taylor['date'],
                    y=df_taylor['taylor_rate'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2),
                    name='Taylor Rule Rate'
                ))
                
                fig_taylor.add_trace(go.Scatter(
                    x=df_taylor['date'],
                    y=df_taylor['fed_funds'],
                    mode='lines',
                    line=dict(color='#00FF00', width=2),
                    name='Actual Fed Funds'
                ))
                
                fig_taylor.update_layout(
                    title="Taylor Rule vs Actual Fed Funds Rate",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333', title="Date"),
                    yaxis=dict(gridcolor='#333', title="Rate (%)"),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_taylor, use_container_width=True)
            
            with col_taylor2:
                current_taylor = df_taylor['taylor_rate'].iloc[-1]
                current_ff = df_taylor['fed_funds'].iloc[-1]
                policy_gap = df_taylor['policy_gap'].iloc[-1]
                
                st.metric(
                    label="TAYLOR RULE RATE",
                    value=f"{current_taylor:.2f}%"
                )
                
                st.metric(
                    label="ACTUAL FED FUNDS",
                    value=f"{current_ff:.2f}%"
                )
                
                st.metric(
                    label="POLICY GAP",
                    value=f"{policy_gap:+.2f}%",
                    delta="Tight" if policy_gap > 0 else "Loose"
                )
                
                if policy_gap > 1:
                    st.markdown("""
                    <div style="background-color: #0a1a00; border-left: 3px solid #00FF00; padding: 8px; margin: 5px 0;">
                        <p style="margin: 0; font-size: 10px; color: #00FF00; font-weight: bold;">
                        📈 RESTRICTIVE POLICY
                        </p>
                        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                        Fed funds above Taylor rule
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif policy_gap < -1:
                    st.markdown("""
                    <div style="background-color: #1a0a00; border-left: 3px solid #FF6600; padding: 8px; margin: 5px 0;">
                        <p style="margin: 0; font-size: 10px; color: #FF6600; font-weight: bold;">
                        📉 ACCOMMODATIVE POLICY
                        </p>
                        <p style="margin: 5px 0 0 0; font-size: 9px; color: #999;">
                        Fed funds below Taylor rule
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# TAB 5: RECHERCHE
with tab5:
    st.markdown("### 🔍 SEARCH FRED SERIES")
    
    st.markdown("#### 📚 AVAILABLE SERIES BY CATEGORY")
    
    for category, series in ECONOMIC_SERIES.items():
        with st.expander(f"📊 {category}", expanded=False):
            for series_id, description in series.items():
                st.markdown(f"**`{series_id}`** - {description}")
    
    st.markdown('<div style="border-top: 1px solid #333; margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔍 CUSTOM SERIES SEARCH")
    
    custom_series_id = st.text_input(
        "Enter FRED Series ID",
        placeholder="Ex: GDP, UNRATE, CPIAUCSL...",
        key="custom_series_input"
    ).upper()
    
    if st.button("🔍 SEARCH", use_container_width=True, key="search_custom"):
        if custom_series_id:
            df = get_fred_series(custom_series_id)
            
            if df is not None:
                st.success(f"✅ Series found: {custom_series_id}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['value'],
                    mode='lines',
                    line=dict(color='#FFAA00', width=2)
                ))
                
                fig.update_layout(
                    title=f"{custom_series_id} - Historical Data",
                    paper_bgcolor='#000',
                    plot_bgcolor='#111',
                    font=dict(color='#FFAA00', size=10),
                    xaxis=dict(gridcolor='#333'),
                    yaxis=dict(gridcolor='#333'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Latest 20 Observations:**")
                st.dataframe(df.tail(20).sort_values('date', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Series '{custom_series_id}' not found")
        else:
            st.warning("⚠️ Please enter a series ID")

# TAB 4: TÉLÉCHARGEMENT
with tab4:
    st.markdown("### 📥 DOWNLOAD ECONOMIC DATA")
    
    download_category = st.selectbox(
        "SELECT CATEGORY",
        options=list(ECONOMIC_SERIES.keys()),
        key="download_category"
    )
    
    series_in_download = ECONOMIC_SERIES[download_category]
    
    download_series = st.multiselect(
        "SELECT SERIES",
        options=list(series_in_download.keys()),
        default=list(series_in_download.keys()),
        format_func=lambda x: f"{x} - {series_in_download[x]}",
        key="download_series"
    )
    
    if st.button("📥 DOWNLOAD CSV", use_container_width=True, key="download_button"):
        if download_series:
            combined_df = pd.DataFrame()
            
            for series_id in download_series:
                df = get_fred_series(series_id)
                if df is not None:
                    df_temp = df[['date', 'value']].copy()
                    df_temp = df_temp.rename(columns={'value': series_id})
                    
                    if combined_df.empty:
                        combined_df = df_temp
                    else:
                        combined_df = pd.merge(combined_df, df_temp, on='date', how='outer')
            
            if not combined_df.empty:
                csv_data = combined_df.to_csv(index=False)
                st.download_button(
                    label="💾 DOWNLOAD CSV",
                    data=csv_data,
                    file_name=f"economic_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success(f"✅ Data prepared! {len(combined_df)} observations")
                st.dataframe(combined_df.head(20), use_container_width=True)

# Footer
st.markdown('<div style="border-top: 1px solid #333; margin: 10px 0;"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 5px;'>
    © 2025 BLOOMBERG ENS® | FEDERAL RESERVE ECONOMIC DATA (FRED) | LAST UPDATE: {datetime.now().strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)
